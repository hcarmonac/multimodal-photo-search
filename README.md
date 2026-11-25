# Multimodal Photo Search

Motor de búsqueda semántica de imágenes ejecutado 100 % en local. Permite localizar fotografías dentro de directorios locales describiendo su contenido en español (objetos, escenas, acciones o contextos) sin necesidad de etiquetado manual previo.

## Cómo funciona

El sistema proyecta imágenes y descripciones textuales en un **espacio latente multimodal compartido** combinado con almacenamiento vectorial persistente:

1. **Encoder visual con carga diferida (`clip-ViT-B-32`):** Extrae embeddings normalizados tras corregir la orientación EXIF y redimensionar las imágenes. Solo se carga en memoria cuando existen fotografías nuevas por indexar.
2. **Encoder textual multilingüe (`clip-ViT-B-32-multilingual-v1`):** Proyecta consultas en español al mismo espacio vectorial que el modelo visual.
3. **Persistencia e indexación incremental (`ChromaDB`):** Almacena los vectores en disco usando métrica de similitud coseno junto con el hash SHA-256 y los metadatos básicos de cada archivo, evitando recalcular embeddings en ejecuciones posteriores.

## Estructura del proyecto

```text
multimodal-photo-search/
├── src/
│   ├── __init__.py
│   ├── config.py           # Constantes, modelos y rutas por defecto
│   ├── encoders.py         # Carga diferida de modelos CLIP y generación de embeddings
│   ├── database.py         # Wrapper de ChromaDB (persistencia y búsqueda coseno)
│   ├── indexer.py          # Escaneo recursivo, hashes SHA-256 e indexación incremental
│   └── search.py           # Lógica de recuperación semántica
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

### Ejecución

Primera ejecución (indexa las imágenes de la carpeta y las guarda en `.chroma_db/`):

```bash
python main.py ruta/a/tus/fotos
```

Siguientes ejecuciones (arranca directamente contra la base de datos persistente):

```bash
python main.py
```

---

## Hoja de ruta (Próximas mejoras)

- [x] **Persistencia con base de datos vectorial local (ChromaDB):** Almacenar los embeddings en disco junto con un identificador único de cada archivo para evitar recalcular vectores en cada ejecución y habilitar indexación incremental.
- [ ] **Búsqueda híbrida por metadatos (EXIF / iCloud):** Extraer fechas de captura y geolocalización de los metadatos de las fotografías (contemplando la exportación de bibliotecas de iCloud) para combinar filtros temporales con búsqueda semántica (ej. *"fotos en la playa en verano de 2024"*).
- [ ] **Clustering automático y mapa 2D de la galería:** Aplicar reducción de dimensionalidad (UMAP) y agrupamiento no supervisado (HDBSCAN / K-Means) sobre los embeddings para generar álbumes temáticos automáticos y visualizar la colección de forma interactiva.
- [ ] **Búsqueda Image-to-Image:** Permitir usar una fotografía de referencia como consulta para encontrar imágenes visualmente o semánticamente similares dentro de la galería.
- [ ] **Carga en streaming por lotes:** Refactorizar la lectura de imágenes mediante generadores por lotes para procesar colecciones masivas sin saturar la memoria RAM.
