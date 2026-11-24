
"""
Análisis de concentración de primeras posiciones.

Distingue la exposición general del primer lugar
y considera cuántos usuarios tenían cada destino
disponible para recibir una recomendación.
"""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


def main():
    train = pd.read_csv(PROCESSED / "train.csv")

    datos = pd.read_csv(
        PROCESSED / "recommendation_concentration.csv"
    )

    usuarios = set(train["UserID"].astype(int))
    total_usuarios = len(usuarios)

    disponibles = []

    for item in datos["ItemID"].astype(int):
        usuarios_que_lo_vieron = set(
            train.loc[
                train["ItemID"] == item, "UserID"
            ].astype(int)
        )

        cantidad = total_usuarios - len(usuarios_que_lo_vieron)
        disponibles.append(cantidad)

    datos["UsuariosDisponibles"] = disponibles

    total_primeros = int(
        datos["PrimerasPosiciones"].sum()
    )

    datos["ParticipacionTop1"] = (
        datos["PrimerasPosiciones"] / total_primeros
    )

    datos["TasaTop1EntreDisponibles"] = (
        datos["PrimerasPosiciones"]
        / datos["UsuariosDisponibles"]
    )

    hhi_top1 = float(
        (datos["ParticipacionTop1"] ** 2).sum()
    )

    top2 = (
        datos.nlargest(2, "PrimerasPosiciones")[
            "PrimerasPosiciones"
        ].sum() / total_primeros
    )

    columnas = [
        "Destino",
        "UsuariosDisponibles",
        "PrimerasPosiciones",
        "ParticipacionTop1",
        "TasaTop1EntreDisponibles"
    ]

    print("\n" + "=" * 60)
    print("CONCENTRACIÓN DE PRIMERAS POSICIONES")
    print("=" * 60)

    print(
        datos[columnas]
        .sort_values("PrimerasPosiciones", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    print("\nINDICADORES TOP 1")
    print("Rankings emitidos:", total_primeros)
    print(f"HHI del primer lugar: {hhi_top1:.4f}")
    print(f"Participación de los dos líderes: {top2:.2%}")
    print(
        "HHI uniforme de referencia:",
        f"{1 / len(datos):.4f}"
    )

    ruta = PROCESSED / "first_position_analysis.csv"
    datos.to_csv(ruta, index=False)

    print("\nArchivo guardado en:", ruta)


if __name__ == "__main__":
    main()
