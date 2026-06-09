from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from app.config import Settings
from app.data_access import ShoppingDataStore, build_data_tools
from app.prompts import (
    DATA_WORKER_PROMPT,
    POLICY_WORKER_PROMPT,
    RESPONSE_WORKER_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.state import ShoppingState
from app.utils import extract_json_payload, get_last_ai_content, serialize_message
from provider import get_chat_model
from rag.embeddings import SentenceTransformerEmbeddings
from rag.vector_store import ChromaPolicyStore


class ShoppingAssistant:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        s = self.settings

        self.llm = get_chat_model(s)
        self.store = ShoppingDataStore(s.orders_path)
        self.data_tools = build_data_tools(self.store)

        embeddings = SentenceTransformerEmbeddings(s.embedding_model_name)
        self.vector_store = ChromaPolicyStore(s.chroma_dir, embeddings)
        self.vector_store.ensure_index(s.policy_path)

        self.graph = build_graph(self.llm, self.vector_store, self.data_tools, s)

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        if rebuild_index:
            self.vector_store.rebuild(self.settings.policy_path)

        initial_state: ShoppingState = {"question": question, "trace": []}
        result = self.graph.invoke(initial_state)

        payload = {
            "question": question,
            "route": result.get("route"),
            "policy_result": result.get("policy_result"),
            "data_result": result.get("data_result"),
            "final_answer": result.get("final_answer"),
            "trace": result.get("trace", []),
        }

        if trace_file:
            trace_path = Path(trace_file)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return payload

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        test_cases = json.loads(Path(test_file).read_text(encoding="utf-8"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, case in enumerate(test_cases):
            question = case.get("question", "")
            trace_file = output_dir / f"trace_{i:03d}.json"
            payload = self.ask(
                question,
                trace_file=trace_file,
                rebuild_index=(rebuild_index and i == 0),
            )
            results.append(
                {
                    "id": case.get("id", f"Q{i+1:02d}"),
                    "question": question,
                    "final_answer": payload["final_answer"],
                    "route": payload["route"],
                }
            )
            print(f"[{i+1}/{len(test_cases)}] {question[:70]}")

        summary = {"total": len(results), "results": results}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary


def build_graph(llm: Any, vector_store: ChromaPolicyStore, data_tools: list, settings: Settings) -> Any:
    data_agent = create_react_agent(llm, data_tools, prompt=DATA_WORKER_PROMPT)

    def _supervisor(state: ShoppingState) -> ShoppingState:
        return supervisor_node(state, llm)

    def _worker_policy(state: ShoppingState) -> ShoppingState:
        return worker_1_policy_node(state, llm, vector_store, settings.top_k)

    def _worker_data(state: ShoppingState) -> ShoppingState:
        return worker_2_data_node(state, data_agent)

    def _worker_response(state: ShoppingState) -> ShoppingState:
        return worker_3_response_node(state, llm)

    def _route_from_supervisor(state: ShoppingState) -> str:
        route = state.get("route") or {}
        if route.get("status") == "clarification_needed":
            return "worker_response"
        if route.get("needs_policy"):
            return "worker_policy"
        if route.get("needs_data"):
            return "worker_data"
        return "worker_response"

    def _route_from_policy(state: ShoppingState) -> str:
        route = state.get("route") or {}
        if route.get("needs_data"):
            return "worker_data"
        return "worker_response"

    graph = StateGraph(ShoppingState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("worker_policy", _worker_policy)
    graph.add_node("worker_data", _worker_data)
    graph.add_node("worker_response", _worker_response)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "worker_policy": "worker_policy",
            "worker_data": "worker_data",
            "worker_response": "worker_response",
        },
    )
    graph.add_conditional_edges(
        "worker_policy",
        _route_from_policy,
        {
            "worker_data": "worker_data",
            "worker_response": "worker_response",
        },
    )
    graph.add_edge("worker_data", "worker_response")
    graph.add_edge("worker_response", END)

    return graph.compile()


def supervisor_node(state: ShoppingState, llm: Any) -> ShoppingState:
    messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    route = extract_json_payload(response.content)
    if not route:
        route = {"status": "ok", "needs_policy": True, "needs_data": False, "clarification_question": None}
    return {
        "route": route,
        "trace": [{"node": "supervisor", "output": route}],
    }


def worker_1_policy_node(
    state: ShoppingState, llm: Any, vector_store: ChromaPolicyStore, top_k: int
) -> ShoppingState:
    question = state["question"]
    hits = vector_store.search(question, top_k=top_k)

    chunks_text = "\n\n".join(
        f"[{h['citation']}]\n{h['content']}" for h in hits
    )

    messages = [
        SystemMessage(content=POLICY_WORKER_PROMPT),
        HumanMessage(content=f"Câu hỏi: {question}\n\nChính sách liên quan:\n{chunks_text}"),
    ]
    response = llm.invoke(messages)
    policy_result = extract_json_payload(response.content)
    if not policy_result:
        policy_result = {
            "status": "ok",
            "summary": response.content,
            "facts": [],
            "citations": [h["citation"] for h in hits],
        }

    return {
        "policy_result": policy_result,
        "trace": [{"node": "worker_policy", "hits": hits, "output": policy_result}],
    }


def worker_2_data_node(state: ShoppingState, data_agent: Any) -> ShoppingState:
    question = state["question"]
    result = data_agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result.get("messages", [])
    last_content = get_last_ai_content(messages)
    data_result = extract_json_payload(last_content)
    if not data_result:
        data_result = {
            "status": "ok",
            "summary": last_content,
            "facts": [],
            "missing_fields": [],
            "not_found_entities": [],
        }

    return {
        "data_result": data_result,
        "trace": [
            {
                "node": "worker_data",
                "messages": [serialize_message(m) for m in messages],
                "output": data_result,
            }
        ],
    }


def worker_3_response_node(state: ShoppingState, llm: Any) -> ShoppingState:
    context = (
        f"Câu hỏi của khách hàng: {state['question']}\n\n"
        f"Routing: {json.dumps(state.get('route'), ensure_ascii=False)}\n\n"
        f"Kết quả Policy Worker: {json.dumps(state.get('policy_result'), ensure_ascii=False)}\n\n"
        f"Kết quả Data Worker: {json.dumps(state.get('data_result'), ensure_ascii=False)}"
    )
    messages = [
        SystemMessage(content=RESPONSE_WORKER_PROMPT),
        HumanMessage(content=context),
    ]
    response = llm.invoke(messages)
    return {
        "final_answer": response.content,
        "trace": [{"node": "worker_response", "output": response.content}],
    }
