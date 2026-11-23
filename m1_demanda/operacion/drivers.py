"""Propósito: Dimensionamiento de conductores por corredor.
Responsable: Persona 2.

A partir de los ciclos diarios ``R_{c,t}`` de ``fleet.dimensionar_flota``:

    H_{c,t}  = R_{c,t}·2·viaje_c                       horas de conducción del día
    CD_{c,t} = ⌈ H_{c,t} / h_dia_max ⌉                 conductores-día (máximo de conducción por día)
    K_c      = ⌈ max( max_t CD_{c,t}·(7/d),  Σ_t H_{c,t} / (h_semana·n/7) ) / (1 − a) ⌉

``n`` es el número de días del horizonte (30), ``h_semana`` la jornada semanal vigente el último día del
horizonte (Ley 2101 de 2021, la más baja del horizonte y por tanto la más exigente), ``d`` los días de trabajo
por semana por conductor (``workday.dias_trabajo_por_semana`` = 6: descanso semanal de 24 h, CST art. 172) y
``a`` el ausentismo. El primer término cubre el día pico CON descansos (para tener ``CD`` conductores
trabajando un día cualquiera, la plantilla debe ser ``CD·7/d``); el segundo, que el total de horas del horizonte
quepa en la jornada semanal. Limitaciones (se reportan en el informe): ``h_semana`` incluye tareas distintas
de conducir que aquí no se modelan, y no se programan turnos, descansos semanales ni pernocta.

El máximo de conducción diaria (9 h) viene del Convenio 153 de la OIT, art. 6, que no se pudo confirmar
como norma ratificada por Colombia (ver ``configs/m1_operacion.yaml``).
"""
from __future__ import annotations

import pandas as pd

from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.fleet import techo


def dimensionar_conductores(por_corredor_dia: pd.DataFrame, p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula horas de conducción, conductores-día y plantilla.

    Devuelve ``(por_corredor_dia_con_horas, plantilla)``. La plantilla tiene, por corredor:
    ``horas_totales, conductores_dia_pico, por_horas_semanales, plantilla, jornada_semanal_h``.
    """
    cd = por_corredor_dia.copy()
    if cd.empty:
        raise ValueError("No hay ciclos para dimensionar conductores.")
    viaje = {int(c): pa.tiempos_ciclo(p, int(c))["viaje_h"] for c in cd["corredor"].unique()}
    cd["horas_conduccion"] = cd["ciclos"] * 2.0 * cd["corredor"].map(lambda c: viaje[int(c)])
    cd["conductores_dia"] = techo(cd["horas_conduccion"] / p["workday"]["max_conduccion_dia_h"])

    n_dias = cd["fecha"].nunique()
    ultimo = pd.to_datetime(cd["fecha"]).max()
    h_sem = pa.jornada_semanal_h(p, ultimo)
    a = p["absenteeism"]
    dias_trab = float(p["workday"]["dias_trabajo_por_semana"])
    filas = []
    for c, g in cd.groupby("corredor"):
        horas = float(g["horas_conduccion"].sum())
        pico = int(g["conductores_dia"].max())
        por_horas = horas / (h_sem * n_dias / 7.0)
        pico_con_descanso = pico * 7.0 / dias_trab           # el descanso semanal exige más gente que el pico diario
        base = max(pico_con_descanso, por_horas)
        filas.append({
            "corredor": int(c), "horas_totales": horas, "conductores_dia_pico": pico,
            "pico_con_descanso": pico_con_descanso,
            "por_horas_semanales": por_horas, "jornada_semanal_h": h_sem,
            "plantilla": int(techo(base / (1.0 - a))),
        })
    return cd, pd.DataFrame(filas)
