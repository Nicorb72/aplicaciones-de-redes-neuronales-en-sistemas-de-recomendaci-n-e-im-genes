
"""
Módulo 3: Estrategias de recomendación cold-start.

Compara:
1. Popularidad global.
2. Popularidad por perfil de preferencias.

Aprende con train.csv y evalúa con cold_val.csv.
No utiliza cold_test.csv ni test.csv.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"


def precision_at_k(recomendados, relevante, k):
    return (
        len(set(recomendados[:k]) & {relevante}) / k
    )


def recall_at_k(recomendados, relevante, k):
    return float(relevante in recomendados[:k])


def ndcg_at_k(recomendados, relevante, k):
    for posicion, item in enumerate(recomendados[:k], start=1):
        if item == relevante:
            import math
            return 1.0 / math.log2(posicion + 1)
    return 0.0


class ColdStartRecommender:

    def __init__(self, strategy="global"):
        self.strategy = strategy
        self.catalogo = []
        self.global_scores = {}
        self.profile_scores = {}
        self.user_preferences = {}

    def fit(self, train, catalogo, usuarios):

        self.catalogo = sorted(
            catalogo["ItemID"].astype(int).unique()
        )

        self.user_preferences = dict(zip(
            usuarios["UserID"].astype(int),
            usuarios["Preferences"]
        ))

        positivos = train.loc[
            train["Positiva"].astype(int) == 1
        ].copy()

        self.global_scores = (
            positivos.groupby("ItemID")
            .size()
            .to_dict()
        )

        positivos["Preferences"] = (
            positivos["UserID"].map(self.user_preferences)
        )

        conteos = (
            positivos.groupby(["Preferences", "ItemID"])
            .size()
        )

        self.profile_scores = conteos.to_dict()

        return self

    def recommend(self, user_id, k=3):

        user_id = int(user_id)

        if self.strategy == "global":

            ranking = sorted(
                self.catalogo,
                key=lambda item: (
                    -self.global_scores.get(item, 0),
                    item
                )
            )

        elif self.strategy == "preferences":

            perfil = self.user_preferences.get(user_id)

            ranking = sorted(
                self.catalogo,
                key=lambda item: (
                    -self.profile_scores.get(
                        (perfil, item), 0
                    ),
                    -self.global_scores.get(item, 0),
                    item
                )
            )

        else:
            raise ValueError("Estrategia desconocida.")

        return ranking[:k]


def main():

    train = pd.read_csv(PROCESSED / "train.csv")
    cold_val = pd.read_csv(
        PROCESSED / "cold_val.csv"
    )

    catalogo = pd.read_csv(
        PROCESSED / "catalogo_destinos.csv"
    )

    usuarios = pd.read_csv(
        RAW / "Final_Updated_Expanded_Users.csv"
    )

    estrategias = ["global", "preferences"]
    resultados = []

    print("\n" + "=" * 60)
    print("EVALUACIÓN COLD-START")
    print("=" * 60)

    for estrategia in estrategias:

        modelo = ColdStartRecommender(
            strategy=estrategia
        ).fit(train, catalogo, usuarios)

        metricas = []

        # Solo usuarios nuevos con al menos una interacción
        # positiva en cold_val.
        for user_id, grupo in cold_val.groupby("UserID"):

            relevantes = set(
                grupo.loc[
                    grupo["Positiva"].astype(int) == 1,
                    "ItemID"
                ].astype(int)
            )

            if not relevantes:
                continue

            recomendados = modelo.recommend(
                user_id, k=3
            )

            # Métricas de ranking con múltiples
            # relevantes posibles por usuario.
            hits = len(
                set(recomendados) & relevantes
            )

            precision = hits / 3
            recall = hits / len(relevantes)

            import math

            dcg = sum(
                1 / math.log2(pos + 2)
                for pos, item in enumerate(recomendados)
                if item in relevantes
            )

            ideal = sum(
                1 / math.log2(pos + 2)
                for pos in range(
                    min(3, len(relevantes))
                )
            )

            ndcg = dcg / ideal if ideal > 0 else 0

            metricas.append({
                "UserID": user_id,
                "Precision": precision,
                "Recall": recall,
                "NDCG": ndcg
            })

        tabla = pd.DataFrame(metricas)

        if tabla.empty:
            raise ValueError(
                "No hay usuarios evaluables en cold_val."
            )

        resultados.append({
            "Estrategia": estrategia,
            "Usuarios": len(tabla),
            "Precision@3": tabla["Precision"].mean(),
            "Recall@3": tabla["Recall"].mean(),
            "NDCG@3": tabla["NDCG"].mean()
        })

    resumen = pd.DataFrame(resultados)

    print(
        resumen.round(4).to_string(index=False)
    )

    ruta = PROCESSED / "cold_start_results.csv"
    resumen.to_csv(ruta, index=False)

    print("\nResultados guardados en:", ruta)


if __name__ == "__main__":
    main()
