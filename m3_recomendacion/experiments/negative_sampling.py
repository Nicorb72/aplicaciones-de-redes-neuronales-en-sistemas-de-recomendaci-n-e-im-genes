
"""
Módulo 3: Diagnóstico para muestreo de negativos.
Responsable: Persona 4.

Analiza interacciones observadas y no observadas.

No modifica train.csv ni genera negativos todavía.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PROCESSED = (
    ROOT / "data" / "m3_recomendacion" / "processed"
)


def main():

    train = pd.read_csv(
        PROCESSED / "train.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    usuarios_train = sorted(
        train["UserID"].unique()
    )

    destinos = sorted(
        catalogo["ItemID"].unique()
    )

    # Todos los pares posibles para usuarios conocidos.
    total_posibles = (
        len(usuarios_train) * len(destinos)
    )

    pares_observados = set(
        zip(
            train["UserID"].astype(int),
            train["ItemID"].astype(int)
        )
    )

    # Interacciones observadas y etiquetadas.
    positivos = int(
        train["Positiva"].sum()
    )

    no_positivos = int(
        (~train["Positiva"].astype(bool)).sum()
    )

    # Pares ausentes del entrenamiento.
    no_observados = (
        total_posibles - len(pares_observados)
    )

    # Separar los pares retenidos en validación,
    # prueba y evaluación de usuarios nuevos.
    reservados = set()

    for nombre in [
        "val.csv",
        "test.csv",
        "cold_val.csv",
        "cold_test.csv"
    ]:

        df = pd.read_csv(
            PROCESSED / nombre
        )

        reservados.update(
            zip(
                df["UserID"].astype(int),
                df["ItemID"].astype(int)
            )
        )

    # Posibles candidatos desconocidos:
    # excluir cualquier par retenido en evaluación.
    candidatos = []

    for usuario in usuarios_train:
        for destino in destinos:

            par = (int(usuario), int(destino))

            if (
                par not in pares_observados
                and par not in reservados
            ):
                candidatos.append(par)

    print("\n" + "=" * 60)
    print("DIAGNÓSTICO DE MUESTREO DE NEGATIVOS")
    print("=" * 60)

    print("Usuarios de entrenamiento:", len(usuarios_train))
    print("Destinos:", len(destinos))

    print("\nPares posibles:", total_posibles)
    print("Pares observados en train:", len(pares_observados))
    print("Pares positivos:", positivos)
    print("Pares no positivos observados:", no_positivos)
    print("Pares no observados en train:", no_observados)

    print(
        "\nDensidad de entrenamiento:",
        f"{len(pares_observados) / total_posibles:.2%}"
    )

    print(
        "\nCandidatos no observados sin pares reservados:",
        len(candidatos)
    )

    print("\nDIAGNÓSTICO COMPLETADO")


if __name__ == "__main__":
    main()
