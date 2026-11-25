from pathlib import Path

from src.config import DEFAULT_TOP_K
from src.database import PhotoVectorStore
from src.encoders import MultimodalEncoder


def search_photos(
    query: str,
    db: PhotoVectorStore,
    encoder: MultimodalEncoder,
    k: int = DEFAULT_TOP_K,
) -> list[tuple[Path, float]]:
    query_embedding = encoder.encode_text(query)
    raw_results = db.search(query_embedding, k=k)
    return [(Path(path_str), score) for path_str, score in raw_results]