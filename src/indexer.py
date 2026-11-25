import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from src.config import BATCH_SIZE, THUMBNAIL_SIZE, VALID_EXTENSIONS
from src.database import PhotoVectorStore
from src.encoders import MultimodalEncoder

EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306


def _compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_exif_date(image: Image.Image) -> str:
    exif = image.getexif()
    raw_date = exif.get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME)
    return str(raw_date) if raw_date else ""


def index_directory(
    directory: Path,
    db: PhotoVectorStore,
    encoder: MultimodalEncoder,
) -> int:
    candidate_files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    if not candidate_files:
        raise ValueError(f"No encontré fotos compatibles en {directory}")

    existing_ids = db.get_existing_ids()
    seen_hashes: set[str] = set()
    pending_files: list[tuple[str, Path]] = []

    for path in candidate_files:
        file_hash = _compute_sha256(path)
        if file_hash in existing_ids or file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        pending_files.append((file_hash, path))

    if not pending_files:
        return 0

    total_new = len(pending_files)
    indexed_count = 0
    print(f"Indexando {total_new} fotos nuevas...")

    for start in range(0, total_new, BATCH_SIZE):
        batch = pending_files[start : start + BATCH_SIZE]
        batch_ids: list[str] = []
        batch_images: list[Image.Image] = []
        batch_metadatas: list[dict[str, str]] = []

        for file_hash, path in batch:
            try:
                with Image.open(path) as img:
                    date_taken = _extract_exif_date(img)
                    photo = ImageOps.exif_transpose(img).convert("RGB")
                    photo.thumbnail(THUMBNAIL_SIZE)
            except OSError as error:
                print(f"No se pudo abrir {path}: {error}")
                continue

            batch_ids.append(file_hash)
            batch_images.append(photo)
            batch_metadatas.append(
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "date_taken": date_taken,
                }
            )

        if not batch_images:
            continue

        embeddings = encoder.encode_images(batch_images)
        db.add_photos(batch_ids, embeddings, batch_metadatas)
        indexed_count += len(batch_images)
        print(f"Progreso: {indexed_count}/{total_new}")

    return indexed_count