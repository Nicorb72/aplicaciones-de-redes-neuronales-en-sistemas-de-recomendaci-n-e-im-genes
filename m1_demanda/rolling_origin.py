"""Propósito: Orígenes de pronóstico (rodantes) para entrenamiento, validación y prueba del Módulo 1.
Responsable: Persona 2.

Regla (la misma que usa ``m1_demanda/data/split.py`` para la prueba)
--------------------------------------------------------------------
Desde el inicio de la partición, un origen cada ``paso_dias`` días mientras
``origen + horizonte - 1 <= fin de la partición``; se agrega el último origen posible
(``fin - horizonte + 1``) para que el pronóstico cubra hasta el último día de la partición.

* prueba, ``paso_dias = 7``: reproduce los 10 orígenes de ``particiones.json``
  (hay una prueba que lo verifica).
* validación, ``paso_dias = 7``: orígenes semanales para elegir hiperparámetros (nunca prueba).
* entrenamiento, ``paso_dias = 1``: una ventana por día, con los 30 días objetivo
  dentro de la partición (el entrenamiento no ve la validación ni la prueba).

Todas las fechas salen de ``configs/m1_datos.yaml`` (sección ``particiones``), que es de la
Persona 1 y no se modifica aquí.
"""
from __future__ import annotations

import pandas as pd

PARTICIONES = ("train", "val", "test", "post_hueco")


def origenes_particion(cfg_datos: dict, particion: str, paso_dias: int = 7,
                       horizonte: int | None = None) -> list[str]:
    """Devuelve los orígenes (``YYYY-MM-DD``) de una partición según la regla del módulo."""
    if particion not in PARTICIONES:
        raise ValueError(f"Partición {particion!r} inválida; use una de {PARTICIONES}.")
    if paso_dias < 1:
        raise ValueError("paso_dias debe ser al menos 1.")
    p = cfg_datos["particiones"]
    h = int(horizonte if horizonte is not None else p["horizonte_dias"])
    ini, fin = pd.Timestamp(p[particion][0]), pd.Timestamp(p[particion][1])
    ultimo = fin - pd.Timedelta(days=h - 1)
    if ultimo < ini:
        return []
    out, o = [], ini
    while o <= ultimo:
        out.append(o)
        o += pd.Timedelta(days=paso_dias)
    if out[-1] != ultimo:
        out.append(ultimo)
    return [o.strftime("%Y-%m-%d") for o in out]
