from pathlib import Path

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, DEFAULT_TOP_K


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
        metadatas: list[dict[str, str]],
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
    ) -> list[tuple[str, float]]:
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

        matches: list[tuple[str, float]] = []
        for meta, distance in zip(metadatas, distances):
            similarity = 1.0 - float(distance)
            matches.append((meta["path"], similarity))

        return matches