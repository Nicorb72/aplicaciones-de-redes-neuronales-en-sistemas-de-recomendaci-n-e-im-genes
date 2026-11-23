
"""
Módulo 3: Diagnóstico previo a la partición de datos.
Responsable: Persona 4.

Objetivo:
Determinar qué usuarios pueden participar en entrenamiento,
validación y prueba, evitando fugas de información.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"


def diagnosticar_particiones():
    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    usuarios = pd.read_csv(
        ROOT / "data" / "m3_recomendacion" / "raw"
        / "Final_Updated_Expanded_Users.csv"
    )

    # Resumir calificaciones por usuario y destino
    pares = (
        eventos.groupby(["UserID", "ItemID"], as_index=False)
        .agg(
            Rating=("Rating", "mean"),
            NumEventos=("Rating", "size")
        )
    )

    # Cantidad de destinos diferentes por usuario
    actividad = (
        pares.groupby("UserID")["ItemID"]
        .nunique()
    )

    # Interacciones relevantes
    pares["Positiva"] = pares["Rating"] >= 4

    print("\n" + "=" * 55)
    print("DIAGNÓSTICO PARA PARTICIONES")
    print("=" * 55)

    print("\nUsuarios registrados:", usuarios["UserID"].nunique())
    print("Usuarios activos:", len(actividad))

    usuarios_nuevos = (
        set(usuarios["UserID"])
        - set(actividad.index)
    )

    print("Usuarios sin historial:", len(usuarios_nuevos))

    print("\nUsuarios según destinos observados:")
    for cantidad in range(1, 6):
        n = (actividad == cantidad).sum()
        print(f"{cantidad} destinos: {n} usuarios")

    print("\nUsuarios con al menos 2 destinos:",
          (actividad >= 2).sum())

    print("Usuarios con al menos 3 destinos:",
          (actividad >= 3).sum())

    print("\nPares usuario-destino:", len(pares))
    print("Pares positivos (promedio >= 4):",
          pares["Positiva"].sum())

    print("Pares no positivos:",
          (~pares["Positiva"]).sum())

    usuarios_con_positivos = (
        pares.loc[pares["Positiva"], "UserID"].nunique()
    )

    print("Usuarios con alguna interacción positiva:",
          usuarios_con_positivos)

    print("\nDIAGNÓSTICO FINALIZADO")

def diagnosticar_positivos():
    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    # Una calificación promedio por usuario y destino
    pares = (
        eventos.groupby(["UserID", "ItemID"], as_index=False)
        .agg(Rating=("Rating", "mean"))
    )

    print("\n" + "=" * 55)
    print("VIABILIDAD DE EVALUACIÓN POR USUARIO")
    print("=" * 55)

    for umbral in [3, 4]:
        positivos = pares[pares["Rating"] >= umbral]

        cantidad = (
            positivos.groupby("UserID")
            .size()
        )

        print(f"\nUmbral positivo: Rating >= {umbral}")

        print("Pares positivos:", len(positivos))

        print("Usuarios con al menos 1 positivo:",
              (cantidad >= 1).sum())

        print("Usuarios con al menos 2 positivos:",
              (cantidad >= 2).sum())

        print("Usuarios con al menos 3 positivos:",
              (cantidad >= 3).sum())

        print("\nDistribución de positivos por usuario:")
        print(
            cantidad.value_counts()
            .sort_index()
            .to_string()
        )

    print("\nDIAGNÓSTICO COMPLETADO")


import numpy as np


def crear_particiones(seed=42):
    rng = np.random.default_rng(seed)

    eventos = pd.read_csv(
        PROCESSED / "eventos_interaccion.csv"
    )

    # Consolidamos las calificaciones de cada par.
    pares = (
        eventos.groupby(["UserID", "ItemID"], as_index=False)
        .agg(
            Rating=("Rating", "mean"),
            NumEventos=("Rating", "size")
        )
    )

    pares["Positiva"] = pares["Rating"] >= 4

    # Usuarios con al menos una interacción positiva.
    usuarios_positivos = (
        pares.loc[pares["Positiva"], "UserID"]
        .unique()
    )

    # Reservamos aproximadamente el 10 % para
    # simular usuarios nuevos.
    n_cold = round(0.10 * len(usuarios_positivos))

    cold_users = rng.choice(
        usuarios_positivos,
        size=n_cold,
        replace=False
    )

    cold = pares[pares["UserID"].isin(cold_users)].copy()
    warm = pares[~pares["UserID"].isin(cold_users)].copy()

    # Separamos usuarios nuevos para validar y probar.
    cold_ids = rng.permutation(cold_users)
    n_cold_val = max(1, round(0.20 * len(cold_ids)))

    cold_val_ids = cold_ids[:n_cold_val]
    cold_test_ids = cold_ids[n_cold_val:]

    cold_val = cold[
        cold["UserID"].isin(cold_val_ids)
    ].copy()

    cold_test = cold[
        cold["UserID"].isin(cold_test_ids)
    ].copy()

    # Usuarios conocidos con al menos dos positivos.
    conteo_positivos = (
        warm.loc[warm["Positiva"]]
        .groupby("UserID")
        .size()
    )

    elegibles = conteo_positivos[
        conteo_positivos >= 2
    ].index.to_numpy()

    elegibles = rng.permutation(elegibles)

    # Mitad para validación, mitad para prueba.
    n_val = len(elegibles) // 2

    usuarios_val = elegibles[:n_val]
    usuarios_test = elegibles[n_val:]

    # Seleccionamos una interacción positiva por usuario.
    indices_val = []
    indices_test = []

    for user_id in usuarios_val:
        candidatos = warm.index[
            (warm["UserID"] == user_id)
            & warm["Positiva"]
        ].to_numpy()

        indices_val.append(rng.choice(candidatos))

    for user_id in usuarios_test:
        candidatos = warm.index[
            (warm["UserID"] == user_id)
            & warm["Positiva"]
        ].to_numpy()

        indices_test.append(rng.choice(candidatos))

    val = warm.loc[indices_val].copy()
    test = warm.loc[indices_test].copy()

    train = warm.drop(
        index=indices_val + indices_test
    ).copy()

    # Comprobar ausencia de fugas.
    def claves(df):
        return set(zip(df["UserID"], df["ItemID"]))

    particiones = [train, val, test, cold_val, cold_test]

    for i in range(len(particiones)):
        for j in range(i + 1, len(particiones)):
            assert claves(particiones[i]).isdisjoint(
                claves(particiones[j])
            )

    assert sum(len(df) for df in particiones) == len(pares)

    # Guardar resultados.
    archivos = {
        "train.csv": train,
        "val.csv": val,
        "test.csv": test,
        "cold_val.csv": cold_val,
        "cold_test.csv": cold_test,
    }

    for nombre, df in archivos.items():
        df.to_csv(PROCESSED / nombre, index=False)

    print("\n" + "=" * 55)
    print("PARTICIONES GENERADAS")
    print("=" * 55)

    for nombre, df in archivos.items():
        print(
            f"{nombre}: {len(df)} pares, "
            f"{df['UserID'].nunique()} usuarios"
        )

    print("\nSemilla:", seed)
    print("Total de pares:", len(pares))
    print("Fugas entre particiones: 0")
    print("PARTICIONES GUARDADAS CORRECTAMENTE")


def auditar_particiones():
    archivos = {
        nombre: pd.read_csv(PROCESSED / f"{nombre}.csv")
        for nombre in [
            "train", "val", "test", "cold_val", "cold_test"
        ]
    }

    train = archivos["train"]
    val = archivos["val"]
    test = archivos["test"]
    cold_val = archivos["cold_val"]
    cold_test = archivos["cold_test"]

    print("\n" + "=" * 55)
    print("AUDITORÍA DE PARTICIONES")
    print("=" * 55)

    usuarios_train = set(train["UserID"])
    usuarios_val = set(val["UserID"])
    usuarios_test = set(test["UserID"])
    usuarios_cold_val = set(cold_val["UserID"])
    usuarios_cold_test = set(cold_test["UserID"])

    # 1. Los usuarios nuevos no deben aparecer en entrenamiento.
    assert usuarios_cold_val.isdisjoint(usuarios_train)
    assert usuarios_cold_test.isdisjoint(usuarios_train)

    # 2. Validación y prueba de usuarios nuevos son independientes.
    assert usuarios_cold_val.isdisjoint(usuarios_cold_test)

    # 3. Los usuarios conocidos tienen historial de entrenamiento.
    assert usuarios_val.issubset(usuarios_train)
    assert usuarios_test.issubset(usuarios_train)

    # 4. Los usuarios conocidos conservan positivos en entrenamiento.
    positivos_train = set(
        train.loc[train["Positiva"], "UserID"]
    )

    assert usuarios_val.issubset(positivos_train)
    assert usuarios_test.issubset(positivos_train)

    # 5. Las interacciones retenidas son positivas.
    for nombre in ["val", "test"]:
        assert archivos[nombre]["Positiva"].all()

    # 6. Ningún par se repite entre particiones.
    conjuntos_pares = {
        nombre: set(zip(df["UserID"], df["ItemID"]))
        for nombre, df in archivos.items()
    }

    nombres = list(conjuntos_pares)

    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            assert conjuntos_pares[nombres[i]].isdisjoint(
                conjuntos_pares[nombres[j]]
            )

    print("Usuarios nuevos excluidos del entrenamiento: OK")
    print("Usuarios conocidos presentes en entrenamiento: OK")
    print("Positivos conservados en entrenamiento: OK")
    print("Interacciones de evaluación positivas: OK")
    print("Pares compartidos entre particiones: 0")
    print("\nAUDITORÍA COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    diagnosticar_particiones()
    diagnosticar_positivos()
    crear_particiones(seed=42)
    auditar_particiones()   

