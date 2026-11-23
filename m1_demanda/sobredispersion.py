"""Propósito: Diagnóstico de sobredispersión de los conteos de pasajeros (justifica la decisión D1).
Responsable: Persona 2.

Si los pasajeros fueran Poisson, la varianza sería igual a la media. Se mide cuánto se aleja de eso con
dos indicadores, solo con datos de ENTRENAMIENTO (días observados, no imputados y fuera del Paro Nacional):

1. **Razón varianza/media por celda** ruta × día de la semana × mes (varianza muestral con ddof = 1, celdas
   con al menos ``--min-obs`` días). Es la medida pedida en el plan, pero mezcla años distintos
   (recuperación pospandemia: 2021 estuvo en 0,64 × el nivel de 2023), así que *sobreestima* la dispersión
   propia del conteo.
2. **Dispersión de Pearson del GLM Poisson**: ``Σ (y − μ)² / μ / (n − p)`` con ``μ`` el pronóstico del GLM
   (descuenta calendario, rezagos y nivel local). Un valor de 1 sería Poisson puro. ``n`` cuenta pares
   (ventana, día objetivo): un mismo día aparece en hasta 30 ventanas solapadas, así que ``n`` no es el
   número de días distintos (por eso los grados de libertad ``n − p`` son solo aproximados).

Interpretación: con valores de cientos o miles de pasajeros la razón es muy superior a 1 (el conteo está
fuertemente sobredispersado), lo que justifica (a) usar Poisson solo como cuasi-verosimilitud de la media
(es consistente aunque la varianza no sea igual a la media) y (b) estimar el cuantil alto con una cabeza
separada de pérdida cuantílica en vez de derivarlo de la distribución Poisson, que lo subestimaría.

Uso: ``python -m m1_demanda.sobredispersion``  ->  ``reports/tables/m1_sobredispersion.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.config_loader import cargar_yaml
from m1_demanda.baselines import GLMPoisson
from m1_demanda.data.utils import RAIZ, cargar_config, ruta_abs
from m1_demanda.ventanas import CONFIG_M1, Panel, cargar_panel, ventanas_particion


def razon_por_celda(panel: Panel, cfg_datos: dict, min_obs: int = 5) -> pd.DataFrame:
    """Razón varianza/media por (ruta, día de la semana, mes) en entrenamiento."""
    ini, fin = cfg_datos["particiones"]["train"]
    i0, i1 = panel.idx(ini), panel.idx(fin)
    fechas = panel.fechas[i0:i1 + 1]
    y = panel.y_obs[i0:i1 + 1].copy()
    y[panel.imputado[i0:i1 + 1]] = np.nan
    y[panel.evento[i0:i1 + 1]] = np.nan
    filas = []
    for r, ruta in enumerate(panel.rutas["ruta_id"]):
        df = pd.DataFrame({"y": y[:, r], "dow": fechas.day_of_week, "mes": fechas.month}).dropna()
        g = df.groupby(["dow", "mes"])["y"].agg(["count", "mean", "var"])
        g = g[(g["count"] >= min_obs) & (g["mean"] > 0)]
        g["razon"] = g["var"] / g["mean"]
        filas.append({"ruta_id": ruta, "n_celdas": int(len(g)), "media_pasajeros": float(df["y"].mean()),
                      "razon_p25": float(g["razon"].quantile(0.25)), "razon_mediana": float(g["razon"].median()),
                      "razon_p75": float(g["razon"].quantile(0.75))})
    return pd.DataFrame(filas)


def dispersion_pearson(glm: GLMPoisson, panel: Panel, cfg_m1: dict, cfg_datos: dict) -> pd.DataFrame:
    """Dispersión de Pearson por ruta con los residuos del GLM en entrenamiento (días válidos)."""
    tr = ventanas_particion(panel, cfg_m1, cfg_datos, "train")
    mu = glm._mu(tr, list(panel.nombres_cal))
    filas = []
    for r, ruta in enumerate(panel.rutas["ruta_id"]):
        sel = (tr.ruta_idx == r)[:, None] & (tr.mask > 0)
        y, m = tr.y[sel].astype(float), mu[sel]
        n, p = int(sel.sum()), len(glm.betas[r])
        filas.append({"ruta_id": ruta, "n_pares_ventana_dia": n,
                      "dispersion_pearson_glm": float(np.sum((y - m) ** 2 / m) / (n - p))})
    return pd.DataFrame(filas)


def diagnostico(panel: Panel, cfg_m1: dict, cfg_datos: dict, min_obs: int = 5) -> pd.DataFrame:
    """Tabla por ruta con ambos indicadores y una fila ``TODAS`` (medianas entre rutas)."""
    celdas = razon_por_celda(panel, cfg_datos, min_obs)
    glm = GLMPoisson().ajustar(panel, cfg_m1, cfg_datos)
    pear = dispersion_pearson(glm, panel, cfg_m1, cfg_datos)
    t = celdas.merge(pear, on="ruta_id")
    todas = {"ruta_id": "TODAS", "n_celdas": int(t["n_celdas"].sum()), "n_pares_ventana_dia": int(t["n_pares_ventana_dia"].sum())}
    for c in ("media_pasajeros", "razon_p25", "razon_mediana", "razon_p75", "dispersion_pearson_glm"):
        todas[c] = float(t[c].median())
    return pd.concat([t, pd.DataFrame([todas])], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Diagnóstico de sobredispersión (Módulo 1).")
    ap.add_argument("--config", default=CONFIG_M1)
    ap.add_argument("--min-obs", type=int, default=5, help="mínimo de días por celda")
    args = ap.parse_args(argv)
    cfg_m1, cfg_datos = cargar_yaml(args.config), cargar_config()
    panel = cargar_panel(cfg_m1, cfg_datos)
    tabla = diagnostico(panel, cfg_m1, cfg_datos, args.min_obs)
    ruta = ruta_abs(cfg_datos, "tablas") / "m1_sobredispersion.csv"
    tabla.to_csv(ruta, index=False, encoding="utf-8")
    with pd.option_context("display.width", 200):
        print(tabla.round(2).to_string(index=False))
    print(f"\nEscrito: {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
