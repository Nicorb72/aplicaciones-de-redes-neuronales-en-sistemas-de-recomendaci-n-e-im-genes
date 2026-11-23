"""Propósito: Sensibilidad de ±10 % del dimensionamiento a los parámetros de operación y a la demanda pronosticada.
Responsable: Persona 2.

Se varía UN parámetro a la vez en ``−δ`` y ``+δ`` (``sensitivity.delta``, 10 %) alrededor del caso base y se mide
el cambio relativo de la flota total, la plantilla de conductores y el costo total (contra la demanda real,
promedio de los 10 orígenes de prueba). Parámetros:

* operación: ``capacity_by_class``, ``target_occupancy``, ``cycle_time`` (viaje y alistamiento de todos los
  corredores), ``operating_hours``, ``technical_reserve``;
* conductores: ``workday.jornada_semanal_h``, ``workday.max_conduccion_dia_h``, ``absenteeism``;
* costos: ``costs.c_veh_dia``, ``costs.c_despacho``, ``costs.c_no_atendido``, ``costs.smmlv`` (salario del conductor);
* la **demanda pronosticada** (el plan se multiplica por ``1 ± δ``; la demanda real con la que se mide no cambia).

Como casi todos los parámetros son supuestos [S] (``configs/m1_operacion.yaml``), esta tabla dice cuáles
importan: es la guía de qué dato real conseguir primero.

Nota sobre la interpretación: variar ``technical_reserve`` o ``absenteeism`` en ±10 % *relativo* (p. ej. 0,08 →
0,088) mueve poco la plantilla; no es lo mismo que variarlos ±10 puntos porcentuales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from m1_demanda.operacion import forecast_value as fv
from m1_demanda.operacion import parametros as pa
from m1_demanda.ventanas import Panel, Ventanas

PARAMETROS = (
    "capacity_by_class", "target_occupancy", "cycle_time", "operating_hours", "technical_reserve",
    "workday.jornada_semanal_h", "workday.max_conduccion_dia_h", "absenteeism",
    "costs.c_veh_dia", "costs.c_despacho", "costs.c_no_atendido", "costs.smmlv",
)
DEMANDA = "demanda_pronosticada"
METRICAS = ("flota_total", "plantilla_total", "costo_total")


def _medir(panel: Panel, v: Ventanas, plan: np.ndarray, params: dict, mezcla: pd.DataFrame) -> dict:
    s = fv.simular(panel, v, {"plan": plan}, params, mezcla)
    return {m: float(s[m].mean()) for m in METRICAS}


def sensibilidad(panel: Panel, v: Ventanas, plan: np.ndarray, params: dict | None = None,
                 delta: float | None = None, parametros: tuple[str, ...] = PARAMETROS + (DEMANDA,),
                 mezcla: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de sensibilidad: una fila por (parámetro, signo) con el valor de las métricas y su cambio relativo.

    ``plan`` es la demanda de planeación (N, H) alineada con ``v`` (por ejemplo el P90 de una red). Columnas:
    ``parametro, variacion (−δ o +δ), flota_total, plantilla_total, costo_total`` y ``d_<metrica>_pct``.
    La primera fila (``parametro = "base"``) es el caso sin variar.
    """
    params = params or pa.cargar_parametros()
    delta = float(params["sensitivity"]["delta"] if delta is None else delta)
    if not 0 <= delta < 1:
        raise ValueError("delta debe estar en [0, 1).")
    mezcla = mezcla if mezcla is not None else pa.cargar_mezcla(params)
    base = _medir(panel, v, plan, params, mezcla)
    filas = [{"parametro": "base", "variacion": 0.0, **base, **{f"d_{m}_pct": 0.0 for m in METRICAS}}]
    for nombre in parametros:
        for signo in (-1.0, 1.0):
            factor = 1.0 + signo * delta
            if nombre == DEMANDA:
                med = _medir(panel, v, plan * factor, params, mezcla)
            else:
                med = _medir(panel, v, plan, pa.escalar_parametro(params, nombre, factor), mezcla)
            filas.append({"parametro": nombre, "variacion": signo * delta, **med,
                          **{f"d_{m}_pct": 100.0 * (med[m] / base[m] - 1.0) for m in METRICAS}})
    return pd.DataFrame(filas)


def orden_por_efecto(tabla: pd.DataFrame, metrica: str = "costo_total") -> list[str]:
    """Parámetros ordenados de mayor a menor efecto absoluto máximo sobre ``metrica``."""
    t = tabla[tabla["parametro"] != "base"]
    efecto = t.groupby("parametro")[f"d_{metrica}_pct"].apply(lambda s: s.abs().max())
    return list(efecto.sort_values(ascending=False).index)
