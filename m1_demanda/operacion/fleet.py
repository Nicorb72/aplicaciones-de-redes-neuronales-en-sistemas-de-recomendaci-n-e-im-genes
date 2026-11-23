"""Propósito: Dimensionamiento de despachos y flota por corredor a partir de la demanda planeada.
Responsable: Persona 2.

Notación: r ruta, r′ su regreso, c corredor, t día. ``D_{r,t}`` es la demanda de planeación:
la media μ (plan puntual) o el cuantil alto q (plan P90).

    C_r      = Σ_k w_{r,k}·cap_k                       capacidad media (w = participación de despachos, entrenamiento)
    N_{r,t}  = ⌈ D_{r,t} / (φ·C_r) ⌉                   despachos requeridos por ruta (φ = ocupación objetivo)
    R_{c,t}  = max(N_{r,t}, N_{r′,t})                  ciclos (viajes redondos) del corredor: NO la suma (decisión D10)
    v_c      = ⌊ H_op / T_c ⌋,  T_c = 2·viaje + 2·alistamiento
    V_{c,t}  = ⌈ R_{c,t} / v_c ⌉                       vehículos necesarios el día t
    F_c      = ⌈ max_t V_{c,t} · (1 + ρ_m) ⌉           flota del horizonte (ρ_m = reserva técnica)

Un vehículo que va debe volver, así que cada ciclo despacha un vehículo en CADA sentido. Los
despachos ejecutados del corredor son ``2·R_{c,t}``; los del sentido más liviano que superan lo
requerido (``R − N``) se reportan como ``despachos_subutilizados`` y se cobran (D10).

Capacidad ofrecida a la ruta r el día t: ``R_{c,t}·C_r`` (todos los vehículos del ciclo salen en ambos
sentidos). Pasajeros no atendidos: ``U_{r,t} = max(0, y_{r,t} − R_{c,t}·C_r)`` contra la demanda real
observada ``y`` (capacidad plena, sin φ).

Costo del horizonte (ver ``costo_operacion``):

    c_veh·Σ V  +  c_desp·Σ (2R)  +  c_cond·K·(días/30)  +  c_u·Σ U
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from m1_demanda.operacion import parametros as pa

EPS = 1e-9


def techo(x):
    """Techo con tolerancia a ruido de punto flotante (3.0000000001 -> 3)."""
    return np.ceil(np.asarray(x, dtype=float) - EPS).astype(int)


def despachos_requeridos(demanda, capacidad_media: float, ocupacion: float):
    """N = ⌈D / (φ·C)⌉ (0 si D = 0). Más capacidad nunca da más despachos."""
    d = np.maximum(np.asarray(demanda, dtype=float), 0.0)
    return techo(d / (ocupacion * capacidad_media))


def dimensionar_flota(plan: pd.DataFrame, rutas: pd.DataFrame, p: dict, mezcla: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dimensiona despachos, ciclos y vehículos.

    ``plan``: columnas ``fecha, ruta_id, demanda`` (demanda de planeación D). Debe incluir, por cada
    corredor presente, la ida y el regreso para las mismas fechas.

    Devuelve ``(por_ruta_dia, por_corredor_dia, flota)``:

    * ``por_ruta_dia``: fecha, ruta_id, corredor, sentido, demanda, capacidad_media, despachos (N).
    * ``por_corredor_dia``: fecha, corredor, ciclos (R), vehiculos (V), despachos_ejecutados (2R),
      despachos_subutilizados, ciclos_por_vehiculo.
    * ``flota``: corredor, ciclo_h, ciclos_por_vehiculo, vehiculos_pico, flota (F_c).
    """
    faltan = {"fecha", "ruta_id", "demanda"} - set(plan.columns)
    if faltan:
        raise ValueError(f"El plan no tiene las columnas {sorted(faltan)}.")
    cat = rutas[["ruta_id", "corredor", "sentido"]]
    d = plan[["fecha", "ruta_id", "demanda"]].merge(cat, on="ruta_id", how="left")
    if d["corredor"].isna().any():
        raise ValueError("El plan tiene rutas que no están en rutas.csv.")
    d["corredor"] = d["corredor"].astype(int)
    if (d["demanda"] < 0).any() or not np.isfinite(d["demanda"]).all():
        raise ValueError("La demanda de planeación debe ser finita y no negativa.")
    d["capacidad_media"] = d["ruta_id"].map({r: pa.capacidad_media(p, mezcla, r) for r in d["ruta_id"].unique()})
    d["despachos"] = despachos_requeridos(d["demanda"].to_numpy(), d["capacidad_media"].to_numpy(),
                                          p["target_occupancy"])
    d = d.sort_values(["corredor", "fecha", "sentido"]).reset_index(drop=True)

    ancho = d.pivot_table(index=["corredor", "fecha"], columns="sentido", values="despachos", aggfunc="sum")
    for s in ("ida", "regreso"):
        if s not in ancho.columns:
            raise ValueError(f"Falta el sentido {s!r} en el plan: un corredor necesita ida y regreso.")
    if ancho[["ida", "regreso"]].isna().any().any():
        raise ValueError("Cada corredor debe tener ida y regreso en todas las fechas del plan.")
    cd = ancho.reset_index()
    cd["ciclos"] = cd[["ida", "regreso"]].max(axis=1).astype(int)
    cd["despachos_ejecutados"] = 2 * cd["ciclos"]
    cd["despachos_subutilizados"] = (cd["ciclos"] - cd["ida"]) + (cd["ciclos"] - cd["regreso"])
    tc = {int(c): pa.tiempos_ciclo(p, int(c)) for c in cd["corredor"].unique()}
    cd["ciclos_por_vehiculo"] = cd["corredor"].map(lambda c: tc[int(c)]["ciclos_por_vehiculo"])
    cd["vehiculos"] = techo(cd["ciclos"] / cd["ciclos_por_vehiculo"])
    cd = cd.rename(columns={"ida": "despachos_ida", "regreso": "despachos_regreso"})

    filas = []
    for c, g in cd.groupby("corredor"):
        pico = int(g["vehiculos"].max())
        filas.append({
            "corredor": int(c), "ciclo_h": tc[int(c)]["ciclo_h"],
            "ciclos_por_vehiculo": tc[int(c)]["ciclos_por_vehiculo"],
            "vehiculos_pico": pico,
            "flota": int(techo(pico * (1 + p["technical_reserve"]))),
        })
    return d, cd, pd.DataFrame(filas)


def pasajeros_no_atendidos(por_ruta_dia: pd.DataFrame, por_corredor_dia: pd.DataFrame,
                           real: pd.DataFrame) -> pd.DataFrame:
    """U = max(0, y − R·C) por ruta y día contra la demanda real ``real`` (fecha, ruta_id, y).

    Solo cuenta los días con demanda real observada (``y`` no nulo).
    """
    m = por_ruta_dia.merge(por_corredor_dia[["corredor", "fecha", "ciclos"]], on=["corredor", "fecha"])
    m = m.merge(real[["fecha", "ruta_id", "y"]], on=["fecha", "ruta_id"], how="left")
    m["capacidad_ofrecida"] = m["ciclos"] * m["capacidad_media"]
    m["no_atendidos"] = np.where(m["y"].notna(), np.maximum(0.0, m["y"] - m["capacidad_ofrecida"]), np.nan)
    return m


def costo_operacion(por_corredor_dia: pd.DataFrame, plantilla: pd.DataFrame, no_atendidos_total: float,
                    n_dias: int, p: dict) -> dict:
    """Costo del horizonte: vehículos-día, despachos ejecutados (incluye los subutilizados),
    conductores (mensual prorrateado a ``n_dias``/30) y pasajeros no atendidos."""
    c = p["costs"]
    veh = c["c_veh_dia"] * float(por_corredor_dia["vehiculos"].sum())
    desp = c["c_despacho"] * float(por_corredor_dia["despachos_ejecutados"].sum())
    c_cond_mes = c["smmlv"] * (1 + c["carga_prestacional"])
    cond = c_cond_mes * float(plantilla["plantilla"].sum()) * n_dias / 30.0
    noat = c["c_no_atendido"] * float(no_atendidos_total)
    return {"vehiculos": veh, "despachos": desp, "conductores": cond, "no_atendidos": noat,
            "total": veh + desp + cond + noat}
