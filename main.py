import argparse
from pathlib import Path

from src.database import PhotoVectorStore
from src.encoders import MultimodalEncoder
from src.indexer import index_directory
from src.search import search_photos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca fotos localmente a partir de una descripción en lenguaje natural."
    )
    parser.add_argument(
        "carpeta",
        nargs="?",
        help="Ruta a la carpeta con fotos para indexar (opcional si ya existen fotos en la BD)",
    )
    args = parser.parse_args()

    db = PhotoVectorStore()
    encoder = MultimodalEncoder()

    ruta_carpeta = args.carpeta
    if ruta_carpeta is None and db.count() == 0:
        ruta_carpeta = input("La base de datos está vacía. ¿Qué carpeta quieres indexar? ").strip()
        if not ruta_carpeta:
            parser.error("Indica una carpeta con fotos para inicializar la base de datos.")

    if ruta_carpeta is not None:
        carpeta = Path(ruta_carpeta).expanduser()
        if not carpeta.is_dir():
            parser.error(f"La carpeta no existe: {carpeta}")

        try:
            nuevas = index_directory(carpeta, db, encoder)
            print(f"Fotos nuevas indexadas: {nuevas}")
        except ValueError as error:
            if db.count() == 0:
                parser.exit(1, f"{error}\n")
            print(error)

    print(f"\nTotal de fotos disponibles en base de datos: {db.count()}")
    print("Pulsa Intro sin escribir nada para salir.")

    while True:
        descripcion = input("\n¿Qué foto buscas? ").strip()
        if not descripcion:
            break

        resultados, tiene_fecha = search_photos(descripcion, db, encoder)

        if not resultados:
            print("No hay fotos indexadas en la base de datos.")
            continue

        for posicion, item in enumerate(resultados, start=1):
            if tiene_fecha:
                detalles = (
                    f"score: {item.final_score:.3f} "
                    f"[sem: {item.similarity:.3f} | temp: {item.temporal_score:.2f}]"
                )
            else:
                detalles = f"similitud: {item.similarity:.3f}"

            if item.date_iso:
                detalles += f" | fecha: {item.date_iso}"

            print(f"{posicion}. {item.path.name} ({detalles})")


if __name__ == "__main__":
    main()