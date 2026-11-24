
"""
Diagnóstico por usuario de estrategias cold-start.

Compara recomendaciones y NDCG@3 en cold_val.
No utiliza el conjunto de prueba.
"""

from pathlib import Path
import math
import pandas as pd

from m3_recomendacion.models.cold_start import ColdStartRecommender


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"

K = 3


def ndcg(recomendaciones, relevantes):
    if not relevantes:
        return 0.0

    dcg = sum(
        1 / math.log2(pos + 2)
        for pos, item in enumerate(recomendaciones[:K])
        if item in relevantes
    )

    ideal = sum(
        1 / math.log2(pos + 2)
        for pos in range(min(K, len(relevantes)))
    )

    return dcg / ideal


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    cold_val = pd.read_csv(PROCESSED / "cold_val.csv")
    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )
    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    global_model = ColdStartRecommender(
        strategy="global"
    ).fit(train, catalogo, usuarios)

    profile_model = ColdStartRecommender(
        strategy="preferences"
    ).fit(train, catalogo, usuarios)

    perfiles = dict(zip(
        usuarios["UserID"].astype(int),
        usuarios["Preferences"]
    ))

    filas = []

    for user_id, grupo in cold_val.groupby("UserID"):
        user_id = int(user_id)

        relevantes = set(
            grupo.loc[
                grupo["Positiva"].astype(int) == 1,
                "ItemID"
            ].astype(int)
        )

        if not relevantes:
            continue

        global_rec = global_model.recommend(user_id, k=K)
        profile_rec = profile_model.recommend(user_id, k=K)

        ndcg_global = ndcg(global_rec, relevantes)
        ndcg_profile = ndcg(profile_rec, relevantes)

        filas.append({
            "UserID": user_id,
            "Preferencias": perfiles[user_id],
            "Relevantes": str(sorted(relevantes)),
            "RankingGlobal": str(global_rec),
            "RankingPreferencias": str(profile_rec),
            "NDCG_Global": ndcg_global,
            "NDCG_Preferencias": ndcg_profile,
            "Diferencia": ndcg_profile - ndcg_global
        })

    df = pd.DataFrame(filas)

    if df.empty:
        raise ValueError(
            "No hay usuarios evaluables en cold_val."
        )

    ruta = PROCESSED / "cold_start_diagnostics.csv"
    df.to_csv(ruta, index=False)

    print("\n" + "=" * 60)
    print("DIAGNÓSTICO COLD-START POR USUARIO")
    print("=" * 60)

    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("BALANCE ENTRE ESTRATEGIAS")
    print("=" * 60)

    print(
        "Usuarios donde mejoran las preferencias:",
        int((df["Diferencia"] > 1e-10).sum())
    )

    print(
        "Usuarios donde empeoran las preferencias:",
        int((df["Diferencia"] < -1e-10).sum())
    )

    print(
        "Usuarios sin cambio en NDCG:",
        int((df["Diferencia"].abs() <= 1e-10).sum())
    )

    print("\nResultados por perfil:")
    print(
        df.groupby("Preferencias")[
            ["NDCG_Global", "NDCG_Preferencias", "Diferencia"]
        ].agg(["mean", "count"]).round(4).to_string()
    )

    print("\nArchivo guardado en:", ruta)


if __name__ == "__main__":
    main()
