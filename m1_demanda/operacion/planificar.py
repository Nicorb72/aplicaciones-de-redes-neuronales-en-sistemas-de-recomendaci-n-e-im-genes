"""Propósito: Encadena flota, conductores y costo en una sola llamada.
Responsable: Persona 2.

Lo usan ``predict.py`` (contrato con la app), ``forecast_value.py`` (valor del pronóstico),
``peak_season.py`` y ``sensitivity.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from m1_demanda.operacion import drivers, fleet
from m1_demanda.operacion import parametros as pa


@dataclass
class ResultadoOperacion:
    por_ruta_dia: pd.DataFrame        # fecha, ruta_id, corredor, sentido, demanda, capacidad_media, despachos
    por_corredor_dia: pd.DataFrame    # ciclos, vehículos, horas, conductores-día por corredor y día
    flota: pd.DataFrame               # una fila por corredor
    plantilla: pd.DataFrame           # una fila por corredor
    parametros: dict

    def resumen_corredor(self) -> pd.DataFrame:
        r = self.flota.merge(self.plantilla, on="corredor")
        sub = self.por_corredor_dia.groupby("corredor")["despachos_subutilizados"].sum().rename("despachos_subutilizados")
        return r.merge(sub, on="corredor")


def planificar(plan: pd.DataFrame, rutas: pd.DataFrame, p: dict | None = None,
               mezcla: pd.DataFrame | None = None) -> ResultadoOperacion:
    """Dimensiona flota y conductores. ``plan``: fecha, ruta_id, demanda (de planeación)."""
    p = p or pa.cargar_parametros()
    mezcla = mezcla if mezcla is not None else pa.cargar_mezcla(p)
    por_ruta, por_corr, flota = fleet.dimensionar_flota(plan, rutas, p, mezcla)
    por_corr, plantilla = drivers.dimensionar_conductores(por_corr, p)
    return ResultadoOperacion(por_ruta, por_corr, flota, plantilla, p)


def costo_con_demanda_real(res: ResultadoOperacion, real: pd.DataFrame) -> dict:
    """Costo total del plan evaluado contra la demanda real observada ``real`` (fecha, ruta_id, y)."""
    m = fleet.pasajeros_no_atendidos(res.por_ruta_dia, res.por_corredor_dia, real)
    n_dias = res.por_corredor_dia["fecha"].nunique()
    c = fleet.costo_operacion(res.por_corredor_dia, res.plantilla, float(m["no_atendidos"].sum()), n_dias,
                              res.parametros)
    c["pasajeros_no_atendidos"] = float(m["no_atendidos"].sum())
    c["demanda_real"] = float(m["y"].sum())
    return c
