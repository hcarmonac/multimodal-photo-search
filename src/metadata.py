from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

EXIF_SUBIFD_OFFSET = 0x8769
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME_DIGITIZED = 36868
TAG_DATETIME = 306


def _parse_exif_datetime(raw_value: str) -> datetime | None:
    cleaned = raw_value.strip()
    formats = (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    )
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def extract_photo_metadata(
    image: Image.Image,
    file_path: Path,
) -> dict[str, str | int]:
    exif = image.getexif()
    sub_ifd = exif.get_ifd(EXIF_SUBIFD_OFFSET)

    raw_date = (
        sub_ifd.get(TAG_DATETIME_ORIGINAL)
        or sub_ifd.get(TAG_DATETIME_DIGITIZED)
        or exif.get(TAG_DATETIME_ORIGINAL)
        or exif.get(TAG_DATETIME)
    )

    dt = _parse_exif_datetime(str(raw_date)) if raw_date else None

    metadata: dict[str, str | int] = {
        "path": str(file_path.resolve()),
        "filename": file_path.name,
    }

    if dt is not None:
        metadata["timestamp"] = int(dt.timestamp())
        metadata["year"] = dt.year
        metadata["month"] = dt.month
        metadata["date_iso"] = dt.strftime("%Y-%m-%d")

    return metadata