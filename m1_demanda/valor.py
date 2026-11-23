"""Propósito: Genera las tablas y la figura del valor del pronóstico y de la temporada alta (Tarea 13).
Responsable: Persona 2.

Simula la decisión de flota y conductores en los 10 orígenes de PRUEBA con estos planes:

    ingenuo_puntual, ingenuo_p90, glm_puntual, glm_p90, [mlp_puntual, mlp_p90, rnn_puntual, rnn_p90], oraculo

Las redes entran solo si existen ``models/m1_demanda/<familia>_final.pt`` (la mejor semilla por VALIDACIÓN que
deja ``python -m m1_demanda.run --mode final``). Salidas:

* ``reports/tables/m1_valor_pronostico.csv``   promedio por plan, con la diferencia de costo frente al oráculo.
* ``reports/tables/m1_valor_pronostico_origenes.csv``   una fila por (origen, plan).
* ``reports/tables/m1_valor_costo_no_atendido.csv``     costo total al variar ``costs.c_no_atendido``.
* ``reports/tables/m1_temporada_alta.csv``   pasajeros no atendidos y vehículos adicionales en alta vs base.
* ``reports/tables/m1_censura_indicios.csv`` indicios de demanda censurada por la oferta (decisión D9).
* ``reports/figures/m1/m1_valor_pronostico.png``.

Uso: ``python -m m1_demanda.valor``
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config_loader import cargar_yaml
from m1_demanda import figuras
from m1_demanda.baselines import GLMPoisson, IngenuoEstacional
from m1_demanda.data.utils import RAIZ, cargar_config, ruta_abs
from m1_demanda.operacion import forecast_value as fv
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.peak_season import indicios_censura
from m1_demanda.train import MODELOS, cargar_modelo, cfg_con_ventana
from m1_demanda.ventanas import CONFIG_M1, cargar_panel, ventanas_particion


def construir_planes(panel, cfg_m1: dict, cfg_datos: dict) -> tuple[dict, "object"]:
    """Planes de demanda para los orígenes de prueba. Devuelve ``(planes, ventanas_de_referencia)``."""
    v = ventanas_particion(panel, cfg_m1, cfg_datos, "test")
    planes: dict[str, np.ndarray] = {}
    ing = IngenuoEstacional(0.9).ajustar(panel, cfg_m1, cfg_datos)
    glm = GLMPoisson(0.9).ajustar(panel, cfg_m1, cfg_datos)
    for nombre, modelo in (("ingenuo", ing), ("glm", glm)):
        mu, q = modelo.predecir(panel, v)
        planes[f"{nombre}_puntual"], planes[f"{nombre}_p90"] = mu, q
    for fam in ("mlp", "rnn"):
        base = MODELOS / f"{fam}_final"
        if not base.with_suffix(".pt").exists():
            continue
        m = cargar_modelo(base, n_rutas=len(panel.rutas))
        cfg_w = cfg_con_ventana(cfg_m1, int(m.hp["window"]))
        vw = ventanas_particion(panel, cfg_w, cfg_datos, "test")
        # mismas ventanas (origen, ruta) que v: con ventana 56 hay el mismo conjunto en prueba
        if len(vw) != len(v) or not (vw.origen == v.origen).all() or not (vw.ruta_idx == v.ruta_idx).all():
            raise ValueError(f"Las ventanas de {fam} no coinciden con las del resto de planes.")
        mu, q = m.predecir(panel, vw)
        planes[f"{fam}_puntual"], planes[f"{fam}_p90"] = mu, q
    planes["oraculo"] = fv.demanda_oraculo(panel, v)
    return planes, v


def tabla_temporada_alta(sim: pd.DataFrame) -> pd.DataFrame:
    """Pasajeros no atendidos y vehículos adicionales, en alta frente a base, promedio por plan."""
    cols = ["pax_no_atendidos_alta", "pax_no_atendidos_base", "pax_no_atendidos", "vehiculos_adicionales_alta",
            "conductores_adicionales_alta"]
    t = sim.groupby("plan", sort=False)[cols].mean().reset_index()
    t["frac_no_atendidos_en_alta"] = t["pax_no_atendidos_alta"] / t["pax_no_atendidos"].where(t["pax_no_atendidos"] > 0)
    return t


def figura_valor(resumen: pd.DataFrame, barrido: pd.DataFrame):
    """Izquierda: costo por componente y plan. Derecha: costo total frente al costo de no atender (supuesto)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    planes = list(resumen["plan"])
    comp = {"costo_vehiculos": "vehículos", "costo_despachos": "despachos", "costo_conductores": "conductores",
            "costo_no_atendidos": "no atendidos"}
    colores = [figuras.AZUL, figuras.NARANJA, figuras.VERDE_AGUA, figuras.TINTA]
    fondo = np.zeros(len(planes))
    a1.set_axisbelow(True)
    a1.grid(axis="x", visible=False)               # sin líneas verticales sobre las barras
    for (col, nombre), c in zip(comp.items(), colores):
        v = resumen[col].to_numpy() / 1e9
        a1.bar(planes, v, bottom=fondo, color=c, label=nombre)
        fondo += v
    a1.set_ylim(0, fondo.max() * 1.22)             # espacio arriba para la leyenda, fuera de las barras
    a1.set_ylabel("Costo medio por origen (miles de millones de COP)")
    a1.set_title("Costo del plan por componente (prueba, 10 orígenes)")
    a1.tick_params(axis="x", rotation=30)
    a1.legend(ncol=4, loc="upper center", fontsize=8)
    # Derecha: el color identifica al modelo y el trazo al tipo de plan (continuo = media, discontinuo = P90).
    for plan, g in barrido.groupby("plan", sort=False):
        p90 = plan.endswith("_p90")
        a2.plot(g["factor_c_no_atendido"], g["costo_total"] / 1e9, marker="s" if p90 else "o", ms=5,
                ls="--" if p90 else "-", label=plan, color=figuras._color(plan),
                lw=2.4 if plan == "oraculo" else 1.4)
    a2.set_xscale("log", base=2)
    a2.set_xlabel("Factor sobre costs.c_no_atendido (supuesto)")
    a2.set_ylabel("Costo total (miles de millones de COP)")
    a2.set_title("¿Cuándo compensa planear con el P90?")
    a2.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def figura_tornado(tabla: pd.DataFrame, plan: str):
    """Tornado de ±δ: cambio porcentual de flota, plantilla y costo por parámetro (orden por efecto en el costo)."""
    from m1_demanda.operacion.sensitivity import METRICAS, orden_por_efecto

    orden = orden_por_efecto(tabla)[::-1]               # el de mayor efecto arriba
    titulos = {"flota_total": "Flota total", "plantilla_total": "Plantilla de conductores", "costo_total": "Costo total"}
    fig, ejes = plt.subplots(1, 3, figsize=(14.0, 5.0), sharey=True)
    y = np.arange(len(orden))
    for ax, m in zip(ejes, METRICAS):
        bajo = tabla[tabla.variacion < 0].set_index("parametro")[f"d_{m}_pct"].reindex(orden)
        alto = tabla[tabla.variacion > 0].set_index("parametro")[f"d_{m}_pct"].reindex(orden)
        ax.barh(y - 0.19, bajo.to_numpy(), 0.36, color=figuras.NARANJA, label="−10 %")
        ax.barh(y + 0.19, alto.to_numpy(), 0.36, color=figuras.AZUL, label="+10 %")
        ax.axvline(0, color=figuras.TINTA, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(orden, fontsize=8)
        ax.set_xlabel("Cambio relativo (%)")
        ax.set_title(titulos[m])
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)                       # la cuadrícula queda detrás de las barras
    ejes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Sensibilidad de ±10 % — plan {plan}", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Valor del pronóstico y temporada alta (Módulo 1).")
    ap.add_argument("--config", default=CONFIG_M1)
    args = ap.parse_args(argv)
    cfg_m1, cfg_datos, params = cargar_yaml(args.config), cargar_config(), pa.cargar_parametros()
    panel = cargar_panel(cfg_m1, cfg_datos)
    planes, v = construir_planes(panel, cfg_m1, cfg_datos)
    sim = fv.simular(panel, v, planes, params)
    res = fv.resumen(sim)
    barrido = fv.barrido_costo_no_atendido(sim, params)
    tablas = ruta_abs(cfg_datos, "tablas")
    sim.to_csv(tablas / "m1_valor_pronostico_origenes.csv", index=False, encoding="utf-8")
    res.to_csv(tablas / "m1_valor_pronostico.csv", index=False, encoding="utf-8")
    barrido.to_csv(tablas / "m1_valor_costo_no_atendido.csv", index=False, encoding="utf-8")
    tabla_temporada_alta(sim).to_csv(tablas / "m1_temporada_alta.csv", index=False, encoding="utf-8")
    indicios_censura(params=params).to_csv(tablas / "m1_censura_indicios.csv", index=False, encoding="utf-8")
    figuras.guardar(figura_valor(res, barrido), cfg_datos, "m1_valor_pronostico.png")
    # Sensibilidad ±10 %: se mide sobre el plan P90 del mejor modelo disponible (red si existe, si no el GLM).
    from m1_demanda.operacion.sensitivity import sensibilidad
    plan_sens = next((p for p in ("mlp_p90", "rnn_p90", "glm_p90") if p in planes), "glm_p90")
    sens = sensibilidad(panel, v, planes[plan_sens], params)
    sens.insert(0, "plan", plan_sens)
    sens.to_csv(tablas / "m1_sensibilidad.csv", index=False, encoding="utf-8")
    figuras.guardar(figura_tornado(sens, plan_sens), cfg_datos, "sensibilidad_tornado.png")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res[["plan", "flota_total", "plantilla_total", "pax_no_atendidos", "ocupacion_realizada",
                   "costo_total", "costo_vs_oraculo_pct"]].round(2).to_string(index=False))
    print(f"\nEscrito: {(tablas / 'm1_valor_pronostico.csv').relative_to(RAIZ)} y 4 tablas más en reports/tables/")


if __name__ == "__main__":
    main()
