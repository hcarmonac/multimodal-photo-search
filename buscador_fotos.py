import argparse
from pathlib import Path

from PIL import Image, ImageOps
from sentence_transformers import SentenceTransformer


def indexar(carpeta, encoder_imagen):
    extensiones = {".jpg", ".jpeg", ".png", ".webp"}
    rutas = []
    imagenes = []

    for ruta in sorted(carpeta.rglob("*")):
        if not ruta.is_file() or ruta.suffix.lower() not in extensiones:
            continue

        try:
            with Image.open(ruta) as imagen:
                foto = ImageOps.exif_transpose(imagen).convert("RGB")
                foto.thumbnail((640, 640))
        except OSError as error:
            print(f"No se pudo abrir {ruta}: {error}")
            continue

        rutas.append(ruta)
        imagenes.append(foto)

    if not imagenes:
        raise ValueError(f"No encontré fotos compatibles en {carpeta}")

    vectores_fotos = encoder_imagen.encode(
        imagenes,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return rutas, vectores_fotos

def buscar(descripcion, encoder_texto, rutas, vectores_fotos, k=5):
    vector_texto = encoder_texto.encode(
        [descripcion],
        normalize_embeddings=True,
    )[0]

    similitudes = vectores_fotos @ vector_texto
    indices = similitudes.argsort()[::-1][:k]

    return [(rutas[i], float(similitudes[i])) for i in indices]

def main():
    # Leer y validar la ruta de la carpeta de imágenes
    parser = argparse.ArgumentParser(
        description="Busca fotos en una carpeta a partir de una descripción."
    )
    parser.add_argument(
        "carpeta",
        nargs="?",
        help="Ruta a la carpeta con las fotos",
    )
    args = parser.parse_args()

    ruta_carpeta = args.carpeta
    if ruta_carpeta is None:
        ruta_carpeta = input("¿En qué carpeta están las fotos? ").strip()

    if not ruta_carpeta:
        parser.error("Indica una carpeta.")

    carpeta = Path(ruta_carpeta).expanduser()
    if not carpeta.is_dir():
        parser.error(f"La carpeta no existe: {carpeta}")

    # Indexar las fotos y conservar la ruta asociada a cada embedding
    encoder_imagen = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    try:
        rutas, vectores_fotos = indexar(carpeta, encoder_imagen)
    except ValueError as error:
        parser.exit(1, f"{error}\n")

    # Cargar el encoder de texto multilingüe
    encoder_texto = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )

    print(f"\nFotos indexadas: {len(rutas)}")
    print("Pulsa Intro sin escribir nada para salir.")

    while True:
        descripcion = input("\n¿Qué foto buscas? ").strip()
        if not descripcion:
            break

        # Calcular similitud coseno entre la descripción y las fotos
        resultados = buscar(descripcion, encoder_texto, rutas, vectores_fotos)

        for posicion, (ruta, similitud) in enumerate(resultados, start=1):
            print(f"{posicion}. {ruta} (similitud: {similitud:.3f})")


if __name__ == "__main__":
    main()