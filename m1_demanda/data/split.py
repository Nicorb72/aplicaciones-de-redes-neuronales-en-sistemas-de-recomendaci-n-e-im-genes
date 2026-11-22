"""Particiones temporales fijas del Módulo 1.

Responsable: Persona 1.

Diseño
------
* Las particiones son por TIEMPO (nunca aleatorias): el modelo solo ve el pasado.
* Se usa el bloque continuo más largo con reporte completo (2021-01-01 a 2023-12-31):
    - entrenamiento: 2021-01-01 a 2023-06-30 (incluye 2 Semanas Santas, 2 fines de año
      y la recuperación pospandemia),
    - validación:    2023-07-01 a 2023-09-30 (selección de hiperparámetros y parada temprana),
    - prueba:        2023-10-01 a 2023-12-31 (temporada alta: receso de octubre,
      puentes de noviembre, Navidad y fin de año) -> cumple "al menos una temporada
      alta en el conjunto de prueba".
* ``post_hueco`` (2025-01-01 a 2026-03-30) queda como evaluación adicional fuera de
  tiempo y para el pronóstico "en producción" de la aplicación. No se usa para elegir
  modelos.
* Orígenes rodantes de prueba: cada 7 días desde el inicio de prueba, mientras
  origen + horizonte - 1 <= fin de prueba. En cada origen el modelo puede usar todos
  los datos anteriores al origen (incluida la parte de prueba ya "revelada").
  Se agrega el último origen posible (fin de prueba - horizonte + 1) para que el
  pronóstico cubra hasta el 31 de diciembre.

Todas las fechas se leen de ``configs/m1_datos.yaml`` (sección ``particiones``).
"""
from __future__ import annotations

import json

import pandas as pd


def asignar_particion(fechas: pd.Series, cfg: dict) -> pd.Series:
    p = cfg["particiones"]
    out = pd.Series("fuera", index=fechas.index, dtype=object)
    for nombre in ["train", "val", "test", "post_hueco"]:
        a, b = p[nombre]
        out[(fechas >= a) & (fechas <= b)] = nombre
    return out


def origenes_rodantes(cfg: dict, paso_dias: int = 7) -> list[str]:
    p = cfg["particiones"]
    h = p["horizonte_dias"]
    ini, fin = pd.Timestamp(p["test"][0]), pd.Timestamp(p["test"][1])
    out, o = [], ini
    while o + pd.Timedelta(days=h - 1) <= fin:
        out.append(o.strftime("%Y-%m-%d"))
        o += pd.Timedelta(days=paso_dias)
    # Se agrega el último origen posible para que el fin de año quede cubierto.
    ultimo = (fin - pd.Timedelta(days=h - 1)).strftime("%Y-%m-%d")
    if out and out[-1] != ultimo:
        out.append(ultimo)
    return out


def guardar_particiones(cfg: dict, ruta) -> dict:
    p = cfg["particiones"]
    info = {
        "descripcion": "Particiones temporales fijas del Módulo 1 (ver m1_demanda/data/split.py)",
        "train": p["train"], "val": p["val"], "test": p["test"],
        "post_hueco": p["post_hueco"],
        "horizonte_dias": p["horizonte_dias"],
        "origenes_rodantes_test": origenes_rodantes(cfg),
    }
    ruta.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info
