
"""
Cinco ejemplos de recomendaciones.

Utiliza únicamente usuarios conocidos en entrenamiento.
No utiliza las particiones de validación o prueba
para seleccionar ejemplos.
"""

from pathlib import Path

import pandas as pd

from m3_recomendacion.experiments.train import NeuralRecommendationSystem


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"


def main():
    train = pd.read_csv(PROCESSED / "train.csv")

    val = pd.read_csv(PROCESSED / "val.csv")

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )
    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    # Nombres de destinos a partir del catálogo.
    nombres = dict(zip(
        catalogo["ItemID"].astype(int),
        catalogo["Name"]
    ))

    perfiles = dict(zip(
        usuarios["UserID"].astype(int),
        usuarios["Preferences"]
    ))
    
    # Arquitectura C5 seleccionada durante la búsqueda.
    # Utilizamos validación para recuperar los mejores pesos.

    modelo = NeuralRecommendationSystem(
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.1,
        lr=0.001,
        epochs=100,
        seed=42,
        patience=15,
        negative_ratio=None
    )

    modelo.fit(
        train,
        catalogo,
        usuarios,
        val=val,
        k=3
    )

    historial = (
        train.groupby("UserID")["ItemID"]
        .apply(lambda s: sorted(set(s.astype(int))))
        .to_dict()
    )

    # Selección reproducible, con diversidad de perfiles
    # y de cantidad de destinos observados.
    candidatos = []

    for user_id, vistos in historial.items():
        if len(vistos) >= 4:
            continue

        candidatos.append({
            "UserID": int(user_id),
            "Preferencias": perfiles[int(user_id)],
            "CantidadVistos": len(vistos)
        })

    tabla = pd.DataFrame(candidatos).sort_values("UserID")

    seleccion = []
    usados = set()

    objetivos = [
        ("Beaches, Historical", 1),
        ("Nature, Adventure", 1),
        ("City, Historical", 2),
        ("Nature, Adventure", 2),
        ("Beaches, Historical", 3)
    ]

    for perfil, cantidad in objetivos:
        opciones = tabla.loc[
            (tabla["Preferencias"] == perfil)
            & (tabla["CantidadVistos"] == cantidad)
            & (~tabla["UserID"].isin(usados))
        ]

        if opciones.empty:
            print(
                "No se encontró ejemplo para:",
                perfil, cantidad
            )
            continue

        user_id = int(opciones.iloc[0]["UserID"])
        usados.add(user_id)

        vistos = historial[user_id]
        ranking = modelo.recommend(user_id, k=3)

        seleccion.append({
            "UserID": user_id,
            "Preferencias": perfil,
            "Historial": "; ".join(
                nombres[i] for i in vistos
            ),
            "Recomendaciones": "; ".join(
                nombres[i] for i, _ in ranking
            ),
            "Puntuaciones": "; ".join(
                f"{score:.4f}" for _, score in ranking
            )
        })

    resultado = pd.DataFrame(seleccion)

    ruta = PROCESSED / "recommendation_examples.csv"
    resultado.to_csv(ruta, index=False)

    print("\n" + "=" * 60)
    print("CINCO EJEMPLOS DE RECOMENDACIONES")
    print("=" * 60)

    print(resultado.to_string(index=False))

    print("\nArchivo guardado en:", ruta)
    print(
        "\nNota: son recomendaciones ilustrativas "
        "generadas con datos de entrenamiento."
    )


if __name__ == "__main__":
    main()
