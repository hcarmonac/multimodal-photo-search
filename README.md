# Buscador Semántico de Fotos con CLIP Multilingüe

Motor de búsqueda de imágenes en lenguaje natural ejecutado en local. Permite localizar fotografías dentro de directorios locales describiendo su contenido en español (objetos, escenas, acciones o contextos) sin necesidad de etiquetado manual previo.

## Cómo funciona

El sistema proyecta imágenes y descripciones textuales en un **espacio latente multimodal compartido** mediante una arquitectura de doble encoder alineado:

1. **Encoder visual (`clip-ViT-B-32`):** Extrae embeddings normalizados de las imágenes tras corregir su orientación EXIF y redimensionarlas para optimizar el coste computacional.
2. **Encoder textual multilingüe (`clip-ViT-B-32-multilingual-v1`):** Traduce consultas en español al mismo espacio vectorial que el modelo visual original.
3. **Recuperación por similitud coseno:** Calcula el producto escalar entre el vector de la consulta y los embeddings normalizados de la galería para devolver las $k$ coincidencias más cercanas junto con su puntuación de confianza.

## Estructura actual e instalación

```bash
git clone [https://github.com/hcarmonac/](https://github.com/hcarmonac/)<nombre-del-repo>.git
cd <nombre-del-repo>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python buscador_fotos.py /ruta/a/tu/carpeta/de/fotos
```

## Próximas implementaciones

- [ ] **Persistencia con base de datos vectorial local (ChromaDB / Qdrant):** Almacenar los embeddings en disco junto con un identificador único de cada archivo para evitar recalcular vectores en cada ejecución y habilitar indexación incremental.
- [ ] **Búsqueda híbrida por metadatos (EXIF / iCloud):** Extraer fechas de captura y geolocalización de los metadatos de las fotografías (contemplando la exportación de bibliotecas de iCloud) para combinar filtros temporales con búsqueda semántica (ej. *"fotos en la playa en verano de 2024"*).
- [ ] **Clustering automático y mapa 2D de la galería:** Aplicar reducción de dimensionalidad (UMAP) y agrupamiento no supervisado (HDBSCAN / K-Means) sobre los embeddings para generar álbumes temáticos automáticos y visualizar la colección de forma interactiva.
- [ ] **Búsqueda *Image-to-Image*:** Permitir usar una fotografía de referencia como consulta para encontrar imágenes visualmente o semánticamente similares dentro de la galería.
- [ ] **Carga en streaming por lotes:** Refactorizar la lectura de imágenes mediante generadores por lotes para procesar colecciones masivas sin saturar la memoria RAM.
