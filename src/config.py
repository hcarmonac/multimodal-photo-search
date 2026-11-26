from pathlib import Path

from dotenv import load_dotenv

# Ruta raíz del proyecto (sube dos niveles desde src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Persistencia de la base de datos vectorial
CHROMA_DB_PATH = BASE_DIR / ".chroma_db"
CHROMA_COLLECTION_NAME = "galeria_fotos"

# Modelos de embeddings
IMAGE_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# Procesamiento de imágenes y búsqueda
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
THUMBNAIL_SIZE = (640, 640)
BATCH_SIZE = 16
DEFAULT_TOP_K = 5

# Parámetros de re-ranking temporal
CANDIDATE_POOL_SIZE = 150
TEMPORAL_DECAY_DAYS = 45.0
TEMPORAL_BOOST_WEIGHT = 0.07