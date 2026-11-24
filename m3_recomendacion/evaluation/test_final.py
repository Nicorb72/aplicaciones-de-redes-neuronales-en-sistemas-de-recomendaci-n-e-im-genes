"""Evaluación final (holdout) de los recomendadores del Módulo 3.

Se ejecuta desde la raíz del repositorio:
    python -m m3_recomendacion.evaluation.test_final

El entrenamiento utiliza train.csv; el sistema neuronal usa val.csv solo
para early stopping. Ningún modelo se entrena ni se selecciona con test.csv.
Las métricas finales se calculan una sola vez sobre test.csv.
"""

from pathlib import Path

import pandas as pd

from m3_recomendacion.evaluation.evaluate import evaluar_modelo
from m3_recomendacion.evaluation.metrics_by_k import entrenar_modelos


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
RAW = ROOT / "data" / "m3_recomendacion" / "raw"


def cargar_datos():
    train = pd.read_csv(PROCESSED / "train.csv")
    val = pd.read_csv(PROCESSED / "val.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    catalogo = pd.read_csv(PROCESSED / "catalogo_destinos.csv")
    usuarios = pd.read_csv(RAW / "Final_Updated_Expanded_Users.csv")
    return train, val, test, catalogo, usuarios


def comprobar_particiones(train, val, test):
    """Detecta errores de particionado antes de evaluar el conjunto final."""
    def pares(df):
        return set(zip(df["UserID"].astype(int), df["ItemID"].astype(int)))

    p_train, p_val, p_test = map(pares, (train, val, test))
    if p_train & p_val or p_train & p_test or p_val & p_test:
        raise ValueError("Hay pares usuario-destino repetidos entre particiones.")
    if not set(test["UserID"].astype(int)).issubset(
        set(train["UserID"].astype(int))
    ):
        raise ValueError("test.csv contiene usuarios no presentes en train.csv.")
    if not test["Positiva"].astype(bool).any():
        raise ValueError("No hay interacciones positivas evaluables en test.csv.")


def main():
    train, val, test, catalogo, usuarios = cargar_datos()
    comprobar_particiones(train, val, test)

    print("\n" + "=" * 60)
    print("EVALUACIÓN FINAL EN TEST — USUARIOS CONOCIDOS")
    print("=" * 60)
    print(f"Interacciones de entrenamiento: {len(train)}")
    print(f"Interacciones de validación: {len(val)}")
    print(f"Interacciones de prueba: {len(test)}")
    print(f"Usuarios de prueba: {test['UserID'].nunique()}")
    print("K = 3 | Semilla = 42")

    # Reutiliza EXACTAMENTE las configuraciones fijadas en validación.
    modelos = entrenar_modelos(train, val, catalogo, usuarios)

    resumen = []
    detalles = []
    for nombre, modelo in modelos.items():
        metricas, por_usuario = evaluar_modelo(modelo, test, catalogo, k=3)
        resumen.append({
            "Modelo": nombre,
            "K": 3,
            "Precision@3": metricas["Precision@K"],
            "Recall@3": metricas["Recall@K"],
            "NDCG@3": metricas["NDCG@K"],
            "Cobertura": metricas["Cobertura"],
            "Usuarios": metricas["Usuarios_evaluados"],
        })
        detalle = por_usuario.copy()
        detalle.insert(0, "Modelo", nombre)
        detalles.append(detalle)

    tabla = pd.DataFrame(resumen).sort_values("NDCG@3", ascending=False)
    detalle_total = pd.concat(detalles, ignore_index=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ruta_resumen = PROCESSED / "test_final_results.csv"
    ruta_usuarios = PROCESSED / "test_final_users.csv"
    tabla.to_csv(ruta_resumen, index=False)
    detalle_total.to_csv(ruta_usuarios, index=False)

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES EN TEST (K=3)")
    print("=" * 60)
    print(tabla.round(4).to_string(index=False))
    print(f"\nResumen guardado: {ruta_resumen}")
    print(f"Resultados por usuario: {ruta_usuarios}")
    print("\nEstos resultados son de TEST; no deben usarse para reajustar modelos.")


if __name__ == "__main__":
    main()
