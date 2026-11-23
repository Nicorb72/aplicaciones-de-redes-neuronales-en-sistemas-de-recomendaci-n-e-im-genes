"""Propósito: Ejecución multisemilla y resumen estadístico.
Responsable: Persona 5 (versión mínima escrita por la Persona 2 para el Módulo 1; pendiente de revisión).

``ejecutar_multisemilla(fn, semillas)`` llama ``fn(semilla)`` por cada semilla. ``fn`` debe
devolver un diccionario ``{metrica: valor}``. Se devuelve:

* ``por_semilla``: DataFrame con una fila por semilla (columna ``semilla`` + métricas).
* ``resumen``: DataFrame indexado por métrica con ``media``, ``desv`` (desviación estándar
  muestral, ddof=1), ``mejor`` y ``peor``. Por defecto menor es mejor (errores, pérdidas);
  se puede invertir por métrica con ``mayor_es_mejor``.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from common.seeds import fijar_semillas


def resumir(por_semilla: pd.DataFrame, mayor_es_mejor: Iterable[str] = ()) -> pd.DataFrame:
    """Resume las métricas por semilla: media, desviación estándar, mejor y peor."""
    mayor = set(mayor_es_mejor)
    metricas = [c for c in por_semilla.columns if c != "semilla"]
    filas = {}
    for m in metricas:
        v = por_semilla[m].to_numpy(dtype=float)
        mejor, peor = (np.nanmax(v), np.nanmin(v)) if m in mayor else (np.nanmin(v), np.nanmax(v))
        filas[m] = {
            "media": float(np.nanmean(v)),
            "desv": float(np.nanstd(v, ddof=1)) if np.sum(~np.isnan(v)) > 1 else float("nan"),
            "mejor": float(mejor),
            "peor": float(peor),
        }
    return pd.DataFrame.from_dict(filas, orient="index")


def ejecutar_multisemilla(
    fn: Callable[[int], dict],
    semillas: Iterable[int],
    mayor_es_mejor: Iterable[str] = (),
    fijar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta ``fn`` con cada semilla y devuelve ``(por_semilla, resumen)``.

    Si ``fijar`` es True se llama ``fijar_semillas`` antes de cada ejecución.
    """
    semillas = list(semillas)
    if not semillas:
        raise ValueError("Se necesita al menos una semilla.")
    filas = []
    for s in semillas:
        if fijar:
            fijar_semillas(s)
        resultado = fn(s)
        if not isinstance(resultado, dict):
            raise TypeError("fn debe devolver un diccionario {metrica: valor}.")
        filas.append({"semilla": int(s), **resultado})
    por_semilla = pd.DataFrame(filas)
    return por_semilla, resumir(por_semilla, mayor_es_mejor)
