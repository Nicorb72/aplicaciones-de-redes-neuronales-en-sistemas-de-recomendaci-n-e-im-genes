"""Evaluación FINAL cold-start (M3) con usuarios nuevos de cold_test.csv.

Mantiene el protocolo de cold_start.py: aprende exclusivamente de train.csv,
utiliza las preferencias declaradas de los usuarios como metadatos disponibles
en inferencia y evalúa los positivos observados de cold_test.csv a K=3.
No usa cold_val.csv para reajustar las estrategias ni utiliza test.csv.

Ejecutar desde la raíz del repositorio:
    python -m m3_recomendacion.evaluation.cold_test_final
"""

from pathlib import Path
import math

import pandas as pd

from m3_recomendacion.models.cold_start import ColdStartRecommender


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"
K = 3


def evaluar_estrategia(modelo, cold_test, k=K):
    """Métricas por usuario; excluye quienes no tengan positivos observados."""
    filas = []
    for user_id, grupo in cold_test.groupby("UserID", sort=True):
        relevantes = set(
            grupo.loc[grupo["Positiva"].astype(int) == 1, "ItemID"].astype(int)
        )
        if not relevantes:
            continue

        recomendados = [int(item) for item in modelo.recommend(int(user_id), k=k)]
        if len(recomendados) != len(set(recomendados)):
            raise ValueError(f"Recomendaciones duplicadas para usuario {user_id}.")

        aciertos = len(set(recomendados) & relevantes)
        dcg = sum(
            1.0 / math.log2(posicion + 2)
            for posicion, item in enumerate(recomendados[:k])
            if item in relevantes
        )
        idcg = sum(
            1.0 / math.log2(posicion + 2)
            for posicion in range(min(k, len(relevantes)))
        )
        filas.append({
            "UserID": int(user_id),
            "Relevantes": len(relevantes),
            "Recomendados": ",".join(map(str, recomendados)),
            "Precision@3": aciertos / k,
            "Recall@3": aciertos / len(relevantes),
            "NDCG@3": dcg / idcg if idcg else 0.0,
        })

    resultado = pd.DataFrame(filas)
    if resultado.empty:
        raise ValueError("cold_test.csv no contiene usuarios con interacciones positivas.")
    return resultado


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    cold_test = pd.read_csv(PROCESSED / "cold_test.csv")
    catalogo = pd.read_csv(PROCESSED / "catalogo_destinos.csv")
    usuarios = pd.read_csv(RAW / "Final_Updated_Expanded_Users.csv")

    # Asegurarse de que los usuarios evaluados son realmente nuevos.
    repetidos = set(train["UserID"].astype(int)) & set(cold_test["UserID"].astype(int))
    if repetidos:
        raise ValueError(f"Hay {len(repetidos)} usuarios de test presentes en train.")

    # Las preferencias son información declarada que debe estar disponible
    # cuando llegue un usuario nuevo. No se derivan de sus ratings de test.
    sin_perfil = set(cold_test["UserID"].astype(int)) - set(usuarios["UserID"].astype(int))
    if sin_perfil:
        raise ValueError(f"Faltan preferencias para {len(sin_perfil)} usuarios nuevos.")

    print("\n" + "=" * 60)
    print("EVALUACIÓN FINAL COLD-START — USUARIOS NUEVOS")
    print("=" * 60)
    print(f"Interacciones de entrenamiento: {len(train)}")
    print(f"Interacciones cold_test: {len(cold_test)}")
    print(f"Usuarios nuevos en cold_test: {cold_test['UserID'].nunique()}")
    print(f"K: {K}")

    resumen = []
    detalles = []
    catalogo_items = set(catalogo["ItemID"].astype(int))

    for estrategia in ("global", "preferences"):
        modelo = ColdStartRecommender(strategy=estrategia).fit(train, catalogo, usuarios)
        tabla = evaluar_estrategia(modelo, cold_test, k=K)
        tabla.insert(0, "Estrategia", estrategia)
        detalles.append(tabla)
        items_recomendados = {
            int(item)
            for cadena in tabla["Recomendados"]
            for item in cadena.split(",") if item
        }
        resumen.append({
            "Estrategia": estrategia,
            "Usuarios": len(tabla),
            "Precision@3": tabla["Precision@3"].mean(),
            "Recall@3": tabla["Recall@3"].mean(),
            "NDCG@3": tabla["NDCG@3"].mean(),
            "Cobertura": len(items_recomendados & catalogo_items) / len(catalogo_items),
        })

    resumen_df = pd.DataFrame(resumen)
    detalle_df = pd.concat(detalles, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ruta_resumen = PROCESSED / "cold_test_final_results.csv"
    ruta_usuarios = PROCESSED / "cold_test_final_users.csv"
    resumen_df.to_csv(ruta_resumen, index=False)
    detalle_df.to_csv(ruta_usuarios, index=False)

    print("\nRESULTADOS FINALES COLD-START (K=3)")
    print(resumen_df.round(4).to_string(index=False))
    print("\nResumen guardado:", ruta_resumen)
    print("Detalle por usuario:", ruta_usuarios)
    print("\nLos resultados de test no deben utilizarse para reajustar estrategias.")


if __name__ == "__main__":
    main()
