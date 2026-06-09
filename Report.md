# Lab Report — Day 09: Multi-Agent Architecture

**Student:** huydunk  
**Date:** 2026-06-09  
**Lab:** Shopping Assistant with LangGraph Multi-Agent System

---

## 1. Overview

This lab implements a multi-agent shopping assistant for the fictional **VinShop Demo** platform using **LangGraph**. The system answers three categories of customer questions: policy-only, order/customer data, and combined questions requiring both policy lookup and data retrieval.

---

## 2. System Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  Supervisor     │  Routes question to appropriate workers
│  Agent          │  Returns: needs_policy, needs_data, clarification_needed
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌────────┐
│Policy  │  │Data    │
│Worker  │  │Worker  │
│(RAG)   │  │(Tools) │
└────┬───┘  └───┬────┘
     │           │
     └─────┬─────┘
           ▼
    ┌─────────────┐
    │  Response   │  Synthesizes final answer with evidence
    │  Worker     │
    └─────────────┘
           │
           ▼
     Final Answer
```

The graph supports three routing paths:
- **Policy only** → Supervisor → Policy Worker → Response Worker
- **Data only** → Supervisor → Data Worker → Response Worker  
- **Combined** → Supervisor → Policy Worker → Data Worker → Response Worker
- **Clarification** → Supervisor → Response Worker (no workers invoked)

---

## 3. Components Implemented

### 3.1 Supervisor Agent (`app/graph.py`)

The supervisor calls the LLM with a structured prompt and returns a JSON routing decision:

```json
{
  "status": "ok",
  "needs_policy": true,
  "needs_data": true,
  "clarification_question": null
}
```

Rules enforced:
- Questions mentioning order/customer IDs → `needs_data = true`
- Policy questions → `needs_policy = true`
- Ambiguous questions missing IDs → `status: clarification_needed`

### 3.2 Policy Worker / RAG Agent (`rag/parser.py`, `rag/vector_store.py`)

**Chunking strategy** (`parser.py`): The policy markdown is split into chunks following the H2 → H3 → content hierarchy. Each chunk contains:

| Field | Example |
|---|---|
| `section_h2` | `5. Chính sách đổi trả và hoàn tiền` |
| `section_h3` | `5.10. Quan hệ giữa trạng thái đơn hàng và quyền trả hàng` |
| `citation` | `5. Chính sách đổi trả... > 5.10. Quan hệ...` |
| `rendered_text` | H2 + H3 + content (full context for embedding) |

**Vector store** (`vector_store.py`):
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: `ChromaDB` (persistent, cosine similarity)
- `ensure_index()` only rebuilds when the collection is empty — avoids re-embedding on every startup

**Retrieval**: Top-K chunks (default K=4) are retrieved and passed to the LLM for summarization with citations.

### 3.3 Data Worker (`app/data_access.py`)

**`ShoppingDataStore`** loads `order_customer_mock_data.json` and builds fast in-memory indexes:
- `customers`: dict keyed by `customer_id`
- `orders`: dict keyed by `order_id`
- `orders_by_customer`: `customer_id → list[order]`
- `vouchers_by_customer`: `customer_id → list[voucher]`

**4 LangChain `@tool` functions** built via `build_data_tools()`:

| Tool | Input | Purpose |
|---|---|---|
| `get_customer_by_id` | `customer_id` | Profile, tier, loyalty points |
| `get_orders_by_customer_id` | `customer_id` | Recent order history |
| `get_order_detail_by_order_id` | `order_id` | Full order detail |
| `get_vouchers_by_customer_id` | `customer_id`, `only_active` | Voucher lookup |

The worker uses `langgraph.prebuilt.create_react_agent` to run the tool-calling loop automatically.

### 3.4 Response Worker (`app/graph.py`)

Receives `route`, `policy_result`, and `data_result` from state, then calls the LLM to produce the final user-facing answer in one of three formats:

```
Answer: ...          ← success
Evidence:
- Policy: ...
- Order data: ...

Status: clarification_needed    ← missing info
Question: ...

Status: not_found               ← data not found
Message: ...
```

### 3.5 Provider Abstraction (`provider/`)

Supports 6 LLM providers with lazy imports (no crash if a provider's package is not installed):

| Provider | Model examples |
|---|---|
| `claude` | `claude-sonnet-4-6`, `claude-haiku-4-5-20251001` |
| `gemini` | `gemini-3.1-flash-lite` |
| `openai` | `gpt-4o`, `gpt-4o-mini` |
| `openrouter` | any OpenRouter model |
| `ollama` | local models |
| `custom` | any OpenAI-compatible endpoint |

Provider is auto-detected from the model name prefix in `.env`.

---

## 4. Test Results

All **22/22** test questions from `data/test.json` answered successfully.

### Routing accuracy

| Category | Questions | Result |
|---|---|---|
| Policy-only (Q01–Q05, Q20) | 6 | ✅ All routed to Policy Worker only |
| Data-only (Q06–Q10, Q17–Q19, Q21) | 9 | ✅ All routed to Data Worker only |
| Combined policy + data (Q11–Q14, Q22) | 5 | ✅ Both workers invoked |
| Clarification needed (Q15–Q16) | 2 | ✅ Correctly returned clarification |

### Notable answers

**Q11 — Combined question:**
> *"Đơn hàng 1971 có được hoàn trả không?"*

The system correctly identified `in_transit` status from data, cross-referenced section 5.10 of policy ("đơn in_transit chưa thể bắt đầu quy trình trả hàng"), and answered that return is not available yet but will be available for 15 days after delivery.

**Q15 — Clarification:**
> *"Voucher của tôi còn dùng được không?"*

Correctly returned `clarification_needed` asking for the customer ID.

**Q17 — Not found:**
> *"Kiểm tra đơn hàng 9999 giúp tôi"*

Correctly returned `not_found` since order 9999 does not exist in the dataset.

---

## 5. Error Handling

| Scenario | Behavior |
|---|---|
| Missing `order_id` / `customer_id` | Supervisor returns `clarification_needed` |
| Non-existent order/customer | Data worker returns `not_found` |
| API rate limit / network error | Each node catches exception, returns `status: error` with message — does not crash |
| Wrong API key | Error shown in Streamlit UI sidebar |

---

## 6. Streamlit UI

A visual debugging interface (`src/streamlit_app.py`) shows the execution flow per question:

- **Flow diagram**: colored nodes showing which agents were activated
- **Supervisor tab**: routing decision with `needs_policy` / `needs_data` flags
- **Policy Worker tab**: RAG hits with citation + relevance score, LLM summary, key facts
- **Data Worker tab**: tool calls made (name + args), tool results, LLM summary
- **Response Worker tab**: final formatted answer

Run with:
```powershell
$env:PYTHONPATH="src"; streamlit run src/streamlit_app.py
```

---

## 7. File Structure

```
src/
├── app/
│   ├── config.py          # Settings loader (multi-provider, .env)
│   ├── state.py           # LangGraph ShoppingState TypedDict
│   ├── data_access.py     # ShoppingDataStore + 4 @tool functions
│   ├── graph.py           # ShoppingAssistant + 4 node functions
│   ├── prompts.py         # Per-agent prompt strings
│   ├── cli.py             # CLI entry point (single + batch)
│   └── utils.py           # JSON parsing, message serialization
├── provider/
│   ├── __init__.py        # Lazy provider dispatch
│   ├── claude.py          # Anthropic ChatAnthropic
│   ├── gemini.py          # Google ChatGoogleGenerativeAI
│   ├── openai.py          # OpenAI ChatOpenAI
│   ├── openrouter.py      # OpenRouter via ChatOpenAI
│   ├── ollama.py          # Ollama ChatOllama
│   └── custom.py          # Custom OpenAI-compatible endpoint
├── rag/
│   ├── embeddings.py      # SentenceTransformerEmbeddings (provided)
│   ├── parser.py          # Markdown → H2/H3 chunks
│   └── vector_store.py    # ChromaPolicyStore (index + search)
├── streamlit_app.py       # Flow visualization UI
└── requirements.txt
data/
├── policy_mock_vi.md           # Knowledge base (10 sections, ~500 lines)
├── order_customer_mock_data.json  # 80 customers, 360 orders, 284 vouchers
└── test.json                   # 22 test questions with expected routes
```

---

## 8. How to Run

```powershell
# Single question
$env:PYTHONPATH="src"; python -m app.cli --question "Đơn hàng 1971 có được hoàn trả không?"

# Batch test (22 questions)
$env:PYTHONPATH="src"; python -m app.cli --batch

# Streamlit UI
$env:PYTHONPATH="src"; streamlit run src/streamlit_app.py
```

`.env` configuration:
```
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=your_key_here
```
