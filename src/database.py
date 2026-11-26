from dataclasses import dataclass
from pathlib import Path

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, DEFAULT_TOP_K


@dataclass(slots=True)
class SearchResult:
    path: Path
    similarity: float
    date_iso: str | None = None
    timestamp: int | None = None
    temporal_score: float = 0.0
    final_score: float = 0.0


class PhotoVectorStore:
    def __init__(
        self,
        db_path: Path = CHROMA_DB_PATH,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def get_existing_ids(self) -> set[str]:
        records = self.collection.get(include=[])
        return set(records["ids"])

    def add_photos(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str | int]],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]:
        total = self.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            include=["metadatas", "distances"],
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        matches: list[SearchResult] = []
        for meta, distance in zip(metadatas, distances):
            sim = 1.0 - float(distance)
            raw_ts = meta.get("timestamp")
            matches.append(
                SearchResult(
                    path=Path(str(meta["path"])),
                    similarity=sim,
                    date_iso=str(meta["date_iso"]) if "date_iso" in meta else None,
                    timestamp=int(raw_ts) if raw_ts is not None else None,
                    final_score=sim,
                )
            )

        return matches