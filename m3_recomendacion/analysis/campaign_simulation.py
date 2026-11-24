
"""
Persona 4: simulación comercial de una campaña.

Escenario:
Promocionar Kerala Backwaters entre usuarios del perfil
Nature, Adventure cuando ya aparece en su Top 3.

Las conversiones son supuestos, no datos reales.
No se utiliza el conjunto de prueba.
"""

from pathlib import Path
from math import ceil

import pandas as pd

from m3_recomendacion.train import NeuralRecommendationSystem


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"

DESTINO = 3
PERFIL = "Nature, Adventure"
FRACCION_CAMPANA = 0.10
K = 3


def main():

    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )
    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

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
        train, catalogo, usuarios, val=val, k=K
    )

    perfiles = dict(zip(
        usuarios["UserID"].astype(int),
        usuarios["Preferences"]
    ))

    candidatos = []
    usuarios_segmento = 0
    usuarios_disponibles = 0
    top1_actual = 0

    for user_id in sorted(modelo.user_map):

        if perfiles[user_id] != PERFIL:
            continue

        usuarios_segmento += 1

        if DESTINO in modelo.historial.get(user_id, set()):
            continue

        usuarios_disponibles += 1

        ranking = modelo.recommend(user_id, k=K)
        ids = [int(item) for item, _ in ranking]

        if ids and ids[0] == DESTINO:
            top1_actual += 1

        # Solamente usuarios para quienes Kerala ya
        # aparece en segunda o tercera posición.
        if DESTINO in ids[1:]:
            candidatos.append({
                "UserID": user_id,
                "PosicionActual": ids.index(DESTINO) + 1
            })

    cantidad_intervenida = ceil(
        len(candidatos) * FRACCION_CAMPANA
    )

    # Selección reproducible por UserID.
    intervenidos = candidatos[:cantidad_intervenida]

    # Cada intervención desplaza Kerala al primer puesto.
    exposiciones_top1_nuevas = len(intervenidos)

    top1_despues = (
        top1_actual + exposiciones_top1_nuevas
    )

    escenarios = [
        ("Conservador", 0.01, 0.02),
        ("Intermedio", 0.01, 0.03),
        ("Optimista", 0.01, 0.05)
    ]

    filas = []

    for nombre, tasa_base, tasa_promocion in escenarios:

        conversiones_antes = (
            cantidad_intervenida * tasa_base
        )

        conversiones_despues = (
            cantidad_intervenida * tasa_promocion
        )

        filas.append({
            "Escenario": nombre,
            "UsuariosIntervenidos": cantidad_intervenida,
            "TasaBase": tasa_base,
            "TasaPromocion": tasa_promocion,
            "ConversionesBaseEsperadas": conversiones_antes,
            "ConversionesPromocionEsperadas": conversiones_despues,
            "ConversionesIncrementalesEsperadas": (
                conversiones_despues - conversiones_antes
            )
        })

    tabla = pd.DataFrame(filas)

    ruta = PROCESSED / "campaign_simulation.csv"
    tabla.to_csv(ruta, index=False)

    print("\n" + "=" * 60)
    print("SIMULACIÓN DE CAMPAÑA TURÍSTICA")
    print("=" * 60)

    print("Destino:", "Kerala Backwaters")
    print("Perfil:", PERFIL)
    print("Usuarios del perfil en train:", usuarios_segmento)
    print("Usuarios con Kerala disponible:", usuarios_disponibles)
    print("Kerala en Top 1 antes:", top1_actual)

    print(
        "Candidatos con Kerala en posición 2 o 3:",
        len(candidatos)
    )
    print(
        "Usuarios intervenidos:",
        cantidad_intervenida
    )
    print(
        "Kerala en Top 1 después del escenario:",
        top1_despues
    )

    print("\nESCENARIOS DE CONVERSIÓN HIPOTÉTICOS")
    print(tabla.round(4).to_string(index=False))

    print("\nNota:")
    print(
        "Las tasas de conversión son supuestos "
        "para análisis de sensibilidad."
    )
    print(
        "No corresponden a reservas ni compras observadas."
    )

    print("\nResultados guardados en:", ruta)


if __name__ == "__main__":
    main()
