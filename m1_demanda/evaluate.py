"""Propósito: Evaluación con origen rodante a 30 días del Módulo 1.
Responsable: Persona 2.

Un *pronosticador* es cualquier objeto con ``predecir(panel, ventanas) -> (mu, q)`` donde
``mu`` y ``q`` tienen forma (N, H): pronóstico puntual y cuantil alto (P90) en pasajeros.

Métricas (sobre los días objetivo con máscara = 1, es decir, observados, no imputados y fuera
del Paro Nacional; nunca sobre valores imputados):

* MAE y RMSE en pasajeros (agregados sobre todos los días-ruta).
* MASE por ruta = MAE de la ruta / escala de la ruta, y el MASE total es el **promedio de los
  MASE por ruta**. Escala = MAE del ingenuo estacional con m = 7 sobre la partición de
  ENTRENAMIENTO (nunca sobre prueba), sin días imputados ni del Paro Nacional.
* Error por horizonte (MAE y MASE para h = 1..H).
* Cobertura del P90 = proporción de días con y <= q (meta: 0,90).
* Pérdida cuantílica (pinball) con τ = 0,9, en pasajeros y escalada por la escala MASE.
* Todo también por subconjunto: ``total``, ``alta`` (día objetivo en temporada alta) y ``base``.

Los 10 orígenes de prueba se solapan: no son observaciones independientes (no usarlos como
muestra independiente en pruebas estadísticas).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from m1_demanda.ventanas import Panel, Ventanas, ventanas_particion

SUBCONJUNTOS = ("total", "alta", "base")


def escala_mase(panel: Panel, cfg_datos: dict, m: int = 7) -> np.ndarray:
    """Escala de MASE por ruta: MAE del ingenuo estacional (rezago m) en ENTRENAMIENTO."""
    ini, fin = cfg_datos["particiones"]["train"]
    i0, i1 = panel.idx(ini), panel.idx(fin)
    y = panel.y_obs.copy()
    y[panel.imputado] = np.nan
    y[panel.evento] = np.nan
    d = np.abs(y[i0 + m:i1 + 1] - y[i0:i1 + 1 - m])
    escala = np.nanmean(d, axis=0)
    if not (np.isfinite(escala).all() and (escala > 0).all()):
        raise ValueError("La escala de MASE debe ser finita y positiva en todas las rutas.")
    return escala


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Pérdida cuantílica L_τ(y, q) = max(τ(y − q), (τ − 1)(y − q))."""
    d = y - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def _pesos(v: Ventanas, subconjunto: str) -> np.ndarray:
    if subconjunto == "total":
        return v.mask
    if subconjunto == "alta":
        return v.mask * v.alta
    if subconjunto == "base":
        return v.mask * (~v.alta)
    raise ValueError(f"Subconjunto {subconjunto!r} inválido; use uno de {SUBCONJUNTOS}.")


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def evaluar(v: Ventanas, mu: np.ndarray, q: np.ndarray, escala: np.ndarray, ids_rutas: list[str],
            tau: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula las métricas. Devuelve ``(tabla, por_horizonte)``.

    ``tabla``: una fila por (subconjunto, ruta) más la fila ``TODAS`` de cada subconjunto, con
    ``n_dias, mae, rmse, mase, cobertura, pinball, pinball_escalada``.
    ``por_horizonte``: ``horizonte`` (1..H), ``n_dias``, ``mae``, ``mase``, ``cobertura``.
    """
    mu, q = np.asarray(mu, dtype=float), np.asarray(q, dtype=float)
    if mu.shape != v.y.shape or q.shape != v.y.shape:
        raise ValueError(f"mu y q deben tener forma {v.y.shape}; se recibió {mu.shape} y {q.shape}.")
    if not (np.isfinite(mu).all() and np.isfinite(q).all()):
        raise ValueError("mu y q contienen valores no finitos.")
    y = v.y.astype(float)
    err = np.abs(y - mu)
    err2 = (y - mu) ** 2
    cubre = (y <= q).astype(float)
    pb = pinball(y, q, tau)

    filas = []
    for sub in SUBCONJUNTOS:
        w = _pesos(v, sub).astype(float)
        por_ruta = []
        for r, ruta in enumerate(ids_rutas):
            sel = v.ruta_idx == r
            n = w[sel].sum()
            mae = _div((err * w)[sel].sum(), n)
            pbm = _div((pb * w)[sel].sum(), n)
            fila = {
                "subconjunto": sub, "ruta_id": ruta, "n_dias": int(n), "mae": mae,
                "rmse": float(np.sqrt(_div((err2 * w)[sel].sum(), n))) if n > 0 else float("nan"),
                "mase": mae / escala[r] if n > 0 else float("nan"),
                "cobertura": _div((cubre * w)[sel].sum(), n),
                "pinball": pbm,
                "pinball_escalada": pbm / escala[r] if n > 0 else float("nan"),
            }
            por_ruta.append(fila)
            filas.append(fila)
        n = w.sum()
        validas = [f for f in por_ruta if f["n_dias"] > 0]
        filas.append({
            "subconjunto": sub, "ruta_id": "TODAS", "n_dias": int(n),
            "mae": _div((err * w).sum(), n),
            "rmse": float(np.sqrt(_div((err2 * w).sum(), n))) if n > 0 else float("nan"),
            "mase": float(np.mean([f["mase"] for f in validas])) if validas else float("nan"),
            "cobertura": _div((cubre * w).sum(), n),
            "pinball": _div((pb * w).sum(), n),
            "pinball_escalada": float(np.mean([f["pinball_escalada"] for f in validas])) if validas else float("nan"),
        })
    tabla = pd.DataFrame(filas)

    H = v.y.shape[1]
    w = v.mask.astype(float)
    esc_v = escala[v.ruta_idx][:, None]
    hor = []
    for k in range(H):
        n = w[:, k].sum()
        mase_rutas = []
        for r in range(len(ids_rutas)):
            sel = v.ruta_idx == r
            nr = w[sel, k].sum()
            if nr > 0:
                mase_rutas.append((err[sel, k] * w[sel, k]).sum() / nr / escala[r])
        hor.append({
            "horizonte": k + 1, "n_dias": int(n),
            "mae": _div((err[:, k] * w[:, k]).sum(), n),
            "mase": float(np.mean(mase_rutas)) if mase_rutas else float("nan"),
            "cobertura": _div((cubre[:, k] * w[:, k]).sum(), n),
        })
    del esc_v
    return tabla, pd.DataFrame(hor)


TRAMOS_HORIZONTE = ((1, 7), (8, 14), (15, 30))      # los tramos que pide el enunciado


def por_tramos(v: Ventanas, mu: np.ndarray, q: np.ndarray, escala: np.ndarray, ids_rutas: list[str],
               tau: float = 0.9, tramos=TRAMOS_HORIZONTE) -> pd.DataFrame:
    """Métricas agregadas por tramo de horizonte (días 1–7, 8–14 y 15–30 del enunciado).

    Para cada tramo se evalúa sobre los días objetivo del tramo (máscara 1) y se devuelve una fila por tramo con
    ``n_dias, mae, rmse, mase, cobertura, pinball_escalada``. El MASE es el promedio de los MASE por ruta, igual que
    en ``evaluar``.
    """
    filas = []
    for a, b in tramos:
        sel = slice(a - 1, b)
        sub = Ventanas(origen=v.origen, ruta_idx=v.ruta_idx, x_hist=v.x_hist, cal_hist=v.cal_hist,
                       cal_fut=v.cal_fut[:, sel], nivel=v.nivel, y=v.y[:, sel], mask=v.mask[:, sel],
                       alta=v.alta[:, sel])
        t, _ = evaluar(sub, mu[:, sel], q[:, sel], escala, ids_rutas, tau)
        g = t[(t.subconjunto == "total") & (t.ruta_id == "TODAS")].iloc[0]
        filas.append({"tramo": f"{a}-{b}", "dia_ini": a, "dia_fin": b, "n_dias": int(g["n_dias"]), "mae": g["mae"],
                      "rmse": g["rmse"], "mase": g["mase"], "cobertura": g["cobertura"],
                      "pinball_escalada": g["pinball_escalada"]})
    return pd.DataFrame(filas)


def evaluar_modelo(modelo, panel: Panel, cfg_m1: dict, cfg_datos: dict, particion: str,
                   paso_dias: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Ventanas, np.ndarray, np.ndarray]:
    """Evalúa un pronosticador en una partición con origen rodante.

    Si el modelo declara su ventana (``modelo.hp["window"]``, como las redes) y difiere de la configuración, las
    ventanas se construyen con la longitud del modelo; los orígenes y los objetivos son los mismos para todos.
    """
    hp = getattr(modelo, "hp", None)
    if hp is not None and int(hp["window"]) != int(cfg_m1["window"]):
        from m1_demanda.train import cfg_con_ventana
        cfg_m1 = cfg_con_ventana(cfg_m1, int(hp["window"]))
    v = ventanas_particion(panel, cfg_m1, cfg_datos, particion, paso_dias)
    if len(v) == 0:
        raise ValueError(f"No hay ventanas en la partición {particion!r}.")
    mu, q = modelo.predecir(panel, v)
    tabla, hor = evaluar(v, mu, q, escala_mase(panel, cfg_datos), list(panel.rutas["ruta_id"]),
                         tau=float(cfg_m1["quantile"]))
    return tabla, hor, v, mu, q
