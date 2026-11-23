"""Propósito: Valor del pronóstico: cuánto cuesta decidir con cada pronóstico frente a conocer la demanda real.
Responsable: Persona 2.

Se simula la decisión de flota y conductores en los orígenes rodantes de PRUEBA (10 orígenes semanales, cada
uno es una decisión a 30 días) con varios *planes*, cada uno una demanda de planeación ``D``:

    ingenuo_puntual, glm_puntual, glm_p90, red_puntual, red_p90, oraculo

``oraculo`` planea con la demanda real (``D = y``): es la regla de decisión con información perfecta. El
**valor del pronóstico** de un plan es ``costo(plan) − costo(oraculo)`` y se reporta también frente al plan
puntual de la misma familia (``red_p90`` vs ``red_puntual``).

Cada plan se evalúa contra la demanda real observada con ``planificar.costo_con_demanda_real``:
costo de vehículos, despachos (incluye los subutilizados del sentido liviano), conductores y pasajeros no
atendidos (``U = max(0, y − R·C)``). Las métricas se promedian sobre los 10 orígenes (NO se suman: los
orígenes se solapan en el horizonte, así que sumar contaría varias veces el mismo día).

Advertencias de lectura
-----------------------
* El oráculo NO es necesariamente el plan más barato: con ocupación objetivo φ < 1 planea con holgura
  (capacidad ≥ y/φ), así que un plan con algo menos de capacidad puede costar menos sin dejar pasajeros sin
  atender. Lo que sí garantiza el oráculo es **cero pasajeros no atendidos** (se comprueba en las pruebas).
* **Vehículos-día sobrantes / faltantes** (``vehiculos_dia_sobrantes``, ``vehiculos_dia_faltantes``): para cada
  corredor y día, ``max(0, V_plan − V_real)`` y ``max(0, V_real − V_plan)``, donde ``V_real`` son los vehículos que
  habría requerido la demanda real con los mismos parámetros (el oráculo). Se suman sobre corredores y días de cada
  origen y se promedian entre orígenes. Lo mismo para conductores-día. Es la cuantificación de "quedarse corto
  frente a sobrar" que pide el enunciado.
* El ranking entre planes depende de ``costs.c_no_atendido`` y de los demás costos, que son supuestos [S]
  (``configs/m1_operacion.yaml``). ``barrido_costo_no_atendido`` muestra cómo cambia el costo total al variar
  ese supuesto: es el análisis relevante para decidir entre planear con la media o con el P90 (vendedor de
  periódicos, ``docs/math_foundations.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from m1_demanda.operacion import fleet
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.peak_season import vehiculos_adicionales
from m1_demanda.operacion.planificar import planificar
from m1_demanda.ventanas import Panel, Ventanas

COSTOS = ("costo_vehiculos", "costo_despachos", "costo_conductores", "costo_no_atendidos")


def demanda_oraculo(panel: Panel, v: Ventanas) -> np.ndarray:
    """Demanda real de planeación (N, H): ``pasajeros_imp`` de los días objetivo (con imputación donde falta)."""
    H = v.y.shape[1]
    out = np.empty((len(v), H))
    for n in range(len(v)):
        i = panel.idx(v.origen[n])
        out[n] = panel.y_imp[i:i + H, v.ruta_idx[n]]
    if not np.isfinite(out).all():
        raise ValueError("El oráculo encontró días sin demanda (NaN) en el horizonte.")
    return out


def _real_observada(panel: Panel, origen, fechas, ruta_ids: list[str], rutas_idx: dict[str, int]) -> pd.DataFrame:
    i0 = panel.idx(origen)
    filas = []
    for rid in ruta_ids:
        r = rutas_idx[rid]
        y = panel.y_obs[i0:i0 + len(fechas), r].copy()
        y[panel.imputado[i0:i0 + len(fechas), r] | panel.evento[i0:i0 + len(fechas), r]] = np.nan
        filas.append(pd.DataFrame({"fecha": fechas, "ruta_id": rid, "y": y}))
    return pd.concat(filas, ignore_index=True)


def simular(panel: Panel, v: Ventanas, planes: dict[str, np.ndarray], params: dict | None = None,
            mezcla: pd.DataFrame | None = None) -> pd.DataFrame:
    """Simula cada plan en cada origen. Devuelve una fila por (origen, plan).

    ``planes``: ``{nombre: demanda de planeación (N, H)}`` alineada con las ventanas ``v``. Debe haber una ventana
    por (origen, ruta) y las 10 rutas en cada origen.
    """
    params = params or pa.cargar_parametros()
    mezcla = mezcla if mezcla is not None else pa.cargar_mezcla(params)
    rutas = panel.rutas
    ids = list(rutas["ruta_id"])
    idx_de = {r: i for i, r in enumerate(ids)}
    alta = pd.Series(panel.alta, index=panel.fechas)
    H = v.y.shape[1]
    for nombre, d in planes.items():
        if d.shape != v.y.shape:
            raise ValueError(f"El plan {nombre!r} tiene forma {d.shape}; se esperaba {v.y.shape}.")
        if (d < 0).any() or not np.isfinite(d).all():
            raise ValueError(f"El plan {nombre!r} tiene demanda negativa o no finita.")

    filas = []
    for origen in np.unique(v.origen):
        sel = np.where(v.origen == origen)[0]
        rutas_sel = [ids[v.ruta_idx[n]] for n in sel]
        if sorted(rutas_sel) != sorted(ids):
            raise ValueError(f"El origen {origen} no tiene las {len(ids)} rutas.")
        fechas = pd.date_range(pd.Timestamp(origen), periods=H, freq="D")
        real = _real_observada(panel, origen, fechas, ids, idx_de)
        # Recursos que habría requerido la demanda real (mismos parámetros): referencia de "sobrante" y "faltante".
        d_or = demanda_oraculo(panel, v.subconjunto(sel))
        plan_or = pd.concat([pd.DataFrame({"fecha": fechas, "ruta_id": ids[v.ruta_idx[n]], "demanda": d_or[k]})
                             for k, n in enumerate(sel)], ignore_index=True)
        ref = planificar(plan_or, rutas, params, mezcla).por_corredor_dia[
            ["corredor", "fecha", "vehiculos", "conductores_dia"]].rename(
            columns={"vehiculos": "veh_ref", "conductores_dia": "cd_ref"})
        for nombre, d in planes.items():
            plan = pd.concat([pd.DataFrame({"fecha": fechas, "ruta_id": ids[v.ruta_idx[n]], "demanda": d[n]})
                              for n in sel], ignore_index=True)
            res = planificar(plan, rutas, params, mezcla)
            m = fleet.pasajeros_no_atendidos(res.por_ruta_dia, res.por_corredor_dia, real)
            n_dias = res.por_corredor_dia["fecha"].nunique()
            c = fleet.costo_operacion(res.por_corredor_dia, res.plantilla, float(m["no_atendidos"].sum()), n_dias,
                                      params)
            cmp_ = res.por_corredor_dia.merge(ref, on=["corredor", "fecha"])
            obs = m["y"].notna()
            m["alta"] = m["fecha"].map(alta).astype(bool)
            atendidos = np.minimum(m.loc[obs, "y"], m.loc[obs, "capacidad_ofrecida"]).sum()
            pico = vehiculos_adicionales(res.por_corredor_dia, alta)
            adic = pico["adicionales"]
            filas.append({
                "origen": pd.Timestamp(origen).strftime("%Y-%m-%d"), "plan": nombre,
                "flota_total": int(res.flota["flota"].sum()), "plantilla_total": int(res.plantilla["plantilla"].sum()),
                "vehiculos_dia": int(res.por_corredor_dia["vehiculos"].sum()),
                "vehiculos_dia_sobrantes": int((cmp_["vehiculos"] - cmp_["veh_ref"]).clip(lower=0).sum()),
                "vehiculos_dia_faltantes": int((cmp_["veh_ref"] - cmp_["vehiculos"]).clip(lower=0).sum()),
                "conductores_dia_sobrantes": int((cmp_["conductores_dia"] - cmp_["cd_ref"]).clip(lower=0).sum()),
                "conductores_dia_faltantes": int((cmp_["cd_ref"] - cmp_["conductores_dia"]).clip(lower=0).sum()),
                "despachos_ejecutados": int(res.por_corredor_dia["despachos_ejecutados"].sum()),
                "despachos_subutilizados": int(res.por_corredor_dia["despachos_subutilizados"].sum()),
                "vehiculos_adicionales_alta": float(adic.sum(skipna=True)),
                "conductores_adicionales_alta": float(pico["conductores_adicionales"].sum(skipna=True)),
                "demanda_real": float(m.loc[obs, "y"].sum()),
                "pax_atendidos": float(atendidos),
                "pax_no_atendidos": float(m["no_atendidos"].sum()),
                "pax_no_atendidos_alta": float(m.loc[m["alta"], "no_atendidos"].sum()),
                "pax_no_atendidos_base": float(m.loc[~m["alta"], "no_atendidos"].sum()),
                "capacidad_ofrecida": float((m["capacidad_ofrecida"]).sum()),
                "ocupacion_realizada": float(atendidos / m["capacidad_ofrecida"].sum()),
                "costo_vehiculos": c["vehiculos"], "costo_despachos": c["despachos"],
                "costo_conductores": c["conductores"], "costo_no_atendidos": c["no_atendidos"],
                "costo_total": c["total"],
            })
    return pd.DataFrame(filas)


def resumen(sim: pd.DataFrame, referencia: str = "oraculo") -> pd.DataFrame:
    """Promedio por plan sobre los orígenes y diferencia de costo frente a ``referencia`` (valor del pronóstico)."""
    numericas = [c for c in sim.columns if c not in ("origen", "plan")]
    r = sim.groupby("plan", sort=False)[numericas].mean()
    r["n_origenes"] = sim.groupby("plan", sort=False)["origen"].nunique()
    if referencia in r.index:
        r["costo_vs_" + referencia] = r["costo_total"] - r.loc[referencia, "costo_total"]
        r["costo_vs_" + referencia + "_pct"] = 100 * r["costo_vs_" + referencia] / r.loc[referencia, "costo_total"]
    return r.reset_index()


def barrido_costo_no_atendido(sim: pd.DataFrame, params: dict, factores=(0.25, 0.5, 1.0, 2.0, 4.0)) -> pd.DataFrame:
    """Costo total medio por plan al multiplicar ``costs.c_no_atendido`` por cada factor.

    Los demás componentes no dependen de ese costo, así que basta recombinar:
    ``total = total_base − costo_no_atendidos + factor·costo_no_atendidos``.
    """
    base = sim.groupby("plan", sort=False)[["costo_total", "costo_no_atendidos"]].mean()
    filas = []
    for f in factores:
        for plan, r in base.iterrows():
            filas.append({"factor_c_no_atendido": f, "c_no_atendido": params["costs"]["c_no_atendido"] * f,
                          "plan": plan, "costo_total": r["costo_total"] - r["costo_no_atendidos"]
                          + f * r["costo_no_atendidos"]})
    t = pd.DataFrame(filas)
    t["mejor_plan_del_factor"] = t.groupby("factor_c_no_atendido")["costo_total"].transform(
        lambda s: s == s.min())
    return t
