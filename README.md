# Multimodal Photo Search

Motor de búsqueda semántica e híbrida de imágenes ejecutado en local. Permite localizar fotografías dentro de directorios locales describiendo su contenido y contexto temporal en español (objetos, escenas, acciones o fechas aproximadas) sin necesidad de etiquetado manual previo.

## Cómo funciona

El sistema combina recuperación densa en un **espacio latente multimodal compartido** con un esquema de **re-ranking temporal en dos etapas**:

1. **Encoder visual con carga diferida (`clip-ViT-B-32`):** Extrae embeddings normalizados tras corregir la orientación EXIF y redimensionar las imágenes (incluyendo soporte nativo para archivos `.heic`/`.heif` mediante `pillow-heif`). Solo se carga en memoria cuando existen fotografías nuevas por indexar.
2. **Encoder textual multilingüe (`clip-ViT-B-32-multilingual-v1`):** Proyecta consultas en español al mismo espacio vectorial que el modelo visual.
3. **Persistencia e indexación incremental (`ChromaDB`):** Almacena los vectores en disco usando métrica de similitud coseno junto con el hash SHA-256 y la fecha original de captura extraída de las cabeceras EXIF (`DateTimeOriginal`), evitando recalcular embeddings en ejecuciones posteriores.
4. **Re-ranking híbrido con decaimiento temporal:** Cuando la consulta incluye una referencia temporal en español (ej. *"en verano de 2024"*, *"septiembre del 2026"*), el analizador separa la fecha del concepto visual para no introducir ruido en CLIP, recupera un conjunto amplio de candidatos desde el índice HNSW y reordena los resultados combinando la similitud coseno con un impulso exponencial según la cercanía a la fecha solicitada (sin excluir imágenes que carezcan de metadatos EXIF).

## Estructura del proyecto

```text
multimodal-photo-search/
├── src/
│   ├── __init__.py
│   ├── config.py           # Constantes, modelos, rutas y parámetros de re-ranking
│   ├── encoders.py         # Carga diferida de modelos CLIP y generación de embeddings
│   ├── metadata.py         # Decodificación HEIC y extracción de fechas EXIF reales
│   ├── database.py         # Wrapper de ChromaDB (persistencia y búsqueda coseno)
│   ├── indexer.py          # Escaneo recursivo, hashes SHA-256 e indexación incremental
│   └── search.py           # Parsing temporal en español y re-ranking en dos etapas
├── main.py                 # Interfaz CLI
├── requirements.txt
└── README.md
```

## Instalación y uso

```bash
git clone [https://github.com/hcarmonac/multimodal-photo-search.git](https://github.com/hcarmonac/multimodal-photo-search.git)
cd multimodal-photo-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Opcional)* Crea un archivo `.env` en la raíz con tu token de lectura de Hugging Face (`HF_TOKEN=hf_...`) para evitar límites de descarga anónima al bajar los modelos por primera vez.

### Ejecución

Primera ejecución (indexa las imágenes de la carpeta y las guarda en `.chroma_db/`):

```bash
python main.py ruta/a/tus/fotos
```

Siguientes ejecuciones (arranca directamente contra la base de datos persistente):

```bash
python main.py
```

Ejemplos de consultas soportadas:
* `perro corriendo en el parque` (búsqueda semántica pura)
* `fotos en la playa en verano de 2024` (búsqueda híbrida con re-ranking temporal)

---

## Hoja de ruta (Próximas mejoras)

- [x] **Persistencia con base de datos vectorial local (ChromaDB):** Almacenar los embeddings en disco junto con un identificador único de cada archivo para evitar recalcular vectores en cada ejecución y habilitar indexación incremental.
- [x] **Búsqueda híbrida por metadatos temporales (EXIF / HEIC):** Extraer fechas de captura reales de los metadatos EXIF (incluyendo formato `.heic` de iCloud) y aplicar re-ranking con decaimiento temporal exponencial ante consultas con fechas en lenguaje natural.
- [ ] **Clustering automático y mapa 2D de la galería:** Aplicar reducción de dimensionalidad (UMAP) y agrupamiento no supervisado (HDBSCAN / K-Means) sobre los embeddings para generar álbumes temáticos automáticos y visualizar la colección de forma interactiva.
- [ ] **Búsqueda *Image-to-Image*:** Permitir usar una fotografía de referencia como consulta para encontrar imágenes visualmente o semánticamente similares dentro de la galería.
- [ ] **Carga en streaming por lotes:** Refactorizar la lectura de imágenes mediante generadores por lotes para procesar colecciones masivas sin saturar la memoria RAM.