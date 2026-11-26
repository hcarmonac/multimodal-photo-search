import calendar
import math
import re
from datetime import datetime, timezone

from src.config import (
    CANDIDATE_POOL_SIZE,
    DEFAULT_TOP_K,
    TEMPORAL_BOOST_WEIGHT,
    TEMPORAL_DECAY_DAYS,
)
from src.database import PhotoVectorStore, SearchResult
from src.encoders import MultimodalEncoder

SECONDS_PER_DAY = 86400.0

MONTHS_ES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def _to_utc_ts(year: int, month: int, day: int, end_of_day: bool = False) -> int:
    if end_of_day:
        dt = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
    else:
        dt = datetime(year, month, day, 0, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp())


def parse_temporal_query(query: str) -> tuple[str, int | None, int | None]:
    """Extrae expresiones temporales en español y devuelve (consulta_limpia, ts_inicio, ts_fin)."""
    # 1. Estaciones del año (ej. "en verano de 2024", "verano 2024")
    season_pattern = re.compile(
        r"\b(?:en\s+(?:el\s+)?)?(primavera|verano|otoño|invierno)\s+(?:de\s+|del\s+)?(20\d{2}|19\d{2})\b",
        re.IGNORECASE,
    )
    match = season_pattern.search(query)
    if match:
        season = match.group(1).lower()
        year = int(match.group(2))
        ranges = {
            "primavera": (_to_utc_ts(year, 3, 20), _to_utc_ts(year, 6, 20, True)),
            "verano": (_to_utc_ts(year, 6, 21), _to_utc_ts(year, 9, 22, True)),
            "otoño": (_to_utc_ts(year, 9, 23), _to_utc_ts(year, 12, 20, True)),
            "invierno": (_to_utc_ts(year - 1, 12, 21), _to_utc_ts(year, 3, 19, True)),
        }
        start_ts, end_ts = ranges[season]
        cleaned = season_pattern.sub("", query).strip(" ,.")
        return cleaned or query, start_ts, end_ts

    # 2. Mes y año (ej. "en julio de 2024", "septiembre del 2026")
    months_regex = "|".join(MONTHS_ES.keys())
    month_pattern = re.compile(
        rf"\b(?:en\s+)?({months_regex})\s+(?:de\s+|del\s+)?(20\d{{2}}|19\d{{2}})\b",
        re.IGNORECASE,
    )
    match = month_pattern.search(query)
    if match:
        month = MONTHS_ES[match.group(1).lower()]
        year = int(match.group(2))
        last_day = calendar.monthrange(year, month)[1]
        cleaned = month_pattern.sub("", query).strip(" ,.")
        return (
            cleaned or query,
            _to_utc_ts(year, month, 1),
            _to_utc_ts(year, month, last_day, True),
        )

    # 3. Año completo (ej. "en 2024", "de 2023")
    year_pattern = re.compile(
        r"\b(?:en|de|del|año)\s+(20\d{2}|19\d{2})\b",
        re.IGNORECASE,
    )
    match = year_pattern.search(query)
    if match:
        year = int(match.group(1))
        cleaned = year_pattern.sub("", query).strip(" ,.")
        return (
            cleaned or query,
            _to_utc_ts(year, 1, 1),
            _to_utc_ts(year, 12, 31, True),
        )

    return query, None, None


def compute_temporal_score(
    photo_ts: int | None,
    start_ts: int,
    end_ts: int,
    decay_days: float = TEMPORAL_DECAY_DAYS,
) -> float:
    """Devuelve 1.0 dentro del rango pedido y decae exponencialmente fuera de él."""
    if photo_ts is None:
        return 0.0

    if start_ts <= photo_ts <= end_ts:
        return 1.0

    delta_seconds = min(abs(photo_ts - start_ts), abs(photo_ts - end_ts))
    delta_days = delta_seconds / SECONDS_PER_DAY
    return math.exp(-delta_days / decay_days)


def search_photos(
    query: str,
    db: PhotoVectorStore,
    encoder: MultimodalEncoder,
    k: int = DEFAULT_TOP_K,
) -> tuple[list[SearchResult], bool]:
    semantic_query, start_ts, end_ts = parse_temporal_query(query)
    has_temporal_hint = start_ts is not None and end_ts is not None

    query_embedding = encoder.encode_text(semantic_query)

    if not has_temporal_hint:
        return db.search(query_embedding, k=k), False

    pool_size = max(k, CANDIDATE_POOL_SIZE)
    candidates = db.search(query_embedding, k=pool_size)

    for item in candidates:
        t_score = compute_temporal_score(item.timestamp, start_ts, end_ts)
        item.temporal_score = t_score
        item.final_score = item.similarity + (TEMPORAL_BOOST_WEIGHT * t_score)

    candidates.sort(key=lambda item: item.final_score, reverse=True)
    return candidates[:k], True