from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from rag.parser import parse_policy_markdown


class ChromaPolicyStore:
    """Chroma-backed policy index using sentence-transformer embeddings."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        self.embedding_model = embedding_model
        self._collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_index(self, markdown_path: Path) -> None:
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        # Drop and recreate so re-indexing is always clean
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        chunks = parse_policy_markdown(markdown_path.read_text(encoding="utf-8"))
        documents = [c["rendered_text"] for c in chunks]
        metadatas = [
            {
                "section_h2": c["section_h2"],
                "section_h3": c["section_h3"],
                "citation": c["citation"],
            }
            for c in chunks
        ]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        embeddings = self.embedding_model.embed_documents(documents)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "citation": results["metadatas"][0][i]["citation"],
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits
