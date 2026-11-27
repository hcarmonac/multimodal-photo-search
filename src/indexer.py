import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from src.config import BATCH_SIZE, THUMBNAIL_SIZE, VALID_EXTENSIONS
from src.database import PhotoVectorStore
from src.encoders import MultimodalEncoder
from src.metadata import extract_photo_metadata


def _compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def index_directory(
    directory: Path,
    db: PhotoVectorStore,
    encoder: MultimodalEncoder,
) -> int:
    # 1. Eliminar de la BD fotos borradas o movidas de su ruta original
    stored_paths = db.get_all_paths()
    missing_ids = [
        photo_id
        for photo_id, path_str in stored_paths.items()
        if not Path(path_str).is_file()
    ]
    if missing_ids:
        db.delete_photos(missing_ids)
        print(f"Registros obsoletos eliminados: {len(missing_ids)}")

    # 2. Escanear fotos actuales en la carpeta
    candidate_files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    if not candidate_files:
        raise ValueError(f"No encontré fotos compatibles en {directory}")

    # 3. Filtrar las que ya están indexadas
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
        batch_metadatas: list[dict[str, str | int]] = []

        for file_hash, path in batch:
            try:
                with Image.open(path) as img:
                    meta = extract_photo_metadata(img, path)
                    photo = ImageOps.exif_transpose(img).convert("RGB")
                    photo.thumbnail(THUMBNAIL_SIZE)
            except OSError as error:
                print(f"No se pudo abrir {path}: {error}")
                continue

            batch_ids.append(file_hash)
            batch_images.append(photo)
            batch_metadatas.append(meta)

        if not batch_images:
            continue

        embeddings = encoder.encode_images(batch_images)
        db.add_photos(batch_ids, embeddings, batch_metadatas)
        indexed_count += len(batch_images)
        print(f"Progreso: {indexed_count}/{total_new}")

    return indexed_count