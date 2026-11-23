"""Propósito: Corrida final con 30 semillas por configuración y tabla de comparación justa (Tarea 12).
Responsable: Persona 2.

Para cada familia de red (MLP y RNN) toma las ``top_k`` mejores configuraciones de la búsqueda
(``models/m1_demanda/mejor_config_<familia>.json``) y las entrena con 30 semillas (``common.multiseed``),
evaluando cada modelo en validación, prueba y post-hueco con el MISMO protocolo que los baselines.

Comparación justa
-----------------
* Mismas particiones, ventanas y métricas para todos los modelos.
* Presupuesto declarado: el ingenuo no tiene hiperparámetros (0 configuraciones), el GLM tiene 1 y cada red
  tuvo ``search.budget`` configuraciones evaluadas en validación con 1 semilla.
* Número de parámetros ENTRENABLES de cada modelo.
* 30 semillas por configuración final (``num_seeds`` de ``configs/global.yaml``); se reporta media, desviación
  estándar, mejor y peor. Los baselines son deterministas: una sola evaluación.
* **Qué modelo se guarda para la app**: por cada familia, la semilla con menor pérdida de VALIDACIÓN, nunca la
  de prueba. La configuración (entre las ``top_k``) con menor MASE medio de validación a lo largo de las
  30 semillas es la "final" de la familia.

Salidas
-------
* ``reports/tables/m1_semillas_<familia>.csv``       una fila por (configuración, semilla, partición).
* ``reports/tables/m1_comparacion.csv``              tabla de comparación justa.
* ``reports/figures/m1/``                            error por horizonte, cobertura por ruta y pronóstico vs. observado.
* ``models/m1_demanda/<familia>_final.{pt,json}``    la mejor semilla de la configuración final.

Uso: ``python -m m1_demanda.run --mode final [--modelo mlp|rnn] [--semillas 30]``
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.multiseed import resumir
from common.seeds import lista_semillas
from m1_demanda.evaluate import evaluar_modelo
from m1_demanda.search import cargar_mejores
from m1_demanda.train import MODELOS, cfg_con_ventana, entrenar, guardar_modelo
from m1_demanda.ventanas import Panel, ventanas_particion

PARTICIONES_EVAL = ("val", "test", "post_hueco")
METRICAS = ("mase", "mae", "rmse", "cobertura", "pinball_escalada", "pinball")


def _fila_total(tabla: pd.DataFrame, subconjunto: str = "total") -> dict:
    t = tabla[(tabla.subconjunto == subconjunto) & (tabla.ruta_id == "TODAS")].iloc[0]
    return {m: float(t[m]) for m in METRICAS}


def correr_configuracion(familia: str, rank: int, hp: dict, cfg_m1: dict, cfg_datos: dict, panel: Panel,
                         semillas: list[int], device="cpu", verbose: bool = False):
    """Entrena ``hp`` con cada semilla y evalúa. Devuelve ``(filas, extras, modelos)``.

    ``filas``: lista de diccionarios (una por semilla, partición y subconjunto).
    ``extras``: ``{"horizonte": [...], "ruta": [...]}`` con el error por horizonte y por ruta de cada semilla.
    ``modelos``: ``{semilla: ResultadoEntrenamiento}`` (se conservan para guardar la mejor por validación).
    """
    w = int(hp["window"])
    cfg_w = cfg_con_ventana(cfg_m1, w) if w != int(cfg_m1["window"]) else cfg_m1
    tr = ventanas_particion(panel, cfg_w, cfg_datos, "train")
    va1 = ventanas_particion(panel, cfg_w, cfg_datos, "val", paso_dias=1)
    filas, modelos = [], {}
    extras: dict[str, list] = {"horizonte": [], "ruta": []}
    for s in semillas:
        res = entrenar(familia, hp, cfg_m1, cfg_datos, panel, semilla=s, device=device,
                       ventanas_train=tr, ventanas_val=va1)
        modelos[s] = res
        for part in PARTICIONES_EVAL:
            tabla, hor, *_ = evaluar_modelo(res.pronosticador, panel, cfg_w, cfg_datos, part)
            ident = {"familia": familia, "config_rank": rank, "semilla": s, "particion": part}
            extras["horizonte"] += [{**ident, **r} for r in hor.to_dict("records")]
            por_ruta = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id != "TODAS")]
            extras["ruta"] += [{**ident, "ruta_id": r["ruta_id"], "mase": r["mase"], "mae": r["mae"],
                                "cobertura": r["cobertura"]} for r in por_ruta.to_dict("records")]
            for sub in ("total", "alta", "base"):
                filas.append({"familia": familia, "config_rank": rank, "semilla": s, "particion": part,
                              "subconjunto": sub, "n_parametros": res.n_parametros, "epocas": len(res.historial),
                              "val_perdida": res.mejor_val, "segundos": res.segundos, **_fila_total(tabla, sub)})
        if verbose:
            v = next(f for f in filas[::-1] if f["particion"] == "val" and f["subconjunto"] == "total")
            print(f"[{familia} #{rank}] semilla {s}: MASE val {v['mase']:.3f}  ({res.segundos:.0f} s)", flush=True)
    return filas, extras, modelos


def correr_familia(familia: str, cfg_m1: dict, cfg_datos: dict, panel: Panel, semillas: list[int],
                   device="cpu", verbose: bool = False, top_k: int | None = None):
    """Corre las ``top_k`` configuraciones con todas las semillas.

    Devuelve ``(tabla, extras, mejor_resultado, info)``; ``extras`` tiene los DataFrames ``horizonte`` y ``ruta``.
    """
    mejores = cargar_mejores(familia)
    configs = mejores[: (top_k or len(mejores))]
    todas, por_config = [], {}
    extras_all: dict[str, list] = {"horizonte": [], "ruta": []}
    t0 = time.perf_counter()
    for c in configs:
        filas, extras, modelos = correr_configuracion(familia, c["rank"], c["hp"], cfg_m1, cfg_datos, panel,
                                                      semillas, device, verbose)
        todas += filas
        for k in extras_all:
            extras_all[k] += extras[k]
        por_config[c["rank"]] = (c, filas, modelos)
    tabla = pd.DataFrame(todas)

    # Configuración final: menor MASE medio de VALIDACIÓN entre semillas (nunca prueba).
    val = tabla[(tabla.particion == "val") & (tabla.subconjunto == "total")]
    rank_final = int(val.groupby("config_rank")["mase"].mean().idxmin())
    c, _, modelos = por_config[rank_final]
    v = val[val.config_rank == rank_final].set_index("semilla")
    semilla_mejor = int(v["val_perdida"].idxmin())            # la de menor pérdida de VALIDACIÓN
    info = {"familia": familia, "config_final_rank": rank_final, "semilla_guardada": semilla_mejor,
            "n_configuraciones_final": len(configs), "segundos_total": time.perf_counter() - t0,
            "hp": c["hp"], "criterio": "menor MASE medio de validación entre 30 semillas; semilla de menor "
                                       "pérdida de validación"}
    return tabla, {k: pd.DataFrame(v) for k, v in extras_all.items()}, modelos[semilla_mejor], info


def tabla_comparacion(tablas_semillas: pd.DataFrame, baselines: pd.DataFrame, presupuestos: dict) -> pd.DataFrame:
    """Tabla de comparación justa: una fila por (modelo/configuración, partición) con media ± desv de las semillas.

    ``tablas_semillas``: concatenación de ``m1_semillas_<familia>.csv``. ``baselines``: filas con ``modelo``,
    ``particion``, métricas y ``n_parametros`` (una sola evaluación). ``presupuestos``: ``{modelo: n_configuraciones}``.
    """
    filas = []
    for (fam, rank, part), g in tablas_semillas[tablas_semillas.subconjunto == "total"].groupby(
            ["familia", "config_rank", "particion"]):
        por = g[["semilla", *METRICAS]].reset_index(drop=True)
        r = resumir(por)
        fila = {"modelo": f"{fam}_cfg{rank}", "tipo": "red", "particion": part, "n_semillas": len(g),
                "n_parametros": int(g["n_parametros"].iloc[0]),
                "configuraciones_evaluadas": presupuestos.get(fam, np.nan),
                "segundos_entrenamiento_medio": float(g["segundos"].mean())}
        for m in METRICAS:
            fila[f"{m}_media"], fila[f"{m}_desv"] = r.loc[m, "media"], r.loc[m, "desv"]
            fila[f"{m}_mejor"], fila[f"{m}_peor"] = r.loc[m, "mejor"], r.loc[m, "peor"]
        filas.append(fila)
    for _, b in baselines.iterrows():
        fila = {"modelo": b["modelo"], "tipo": "baseline", "particion": b["particion"], "n_semillas": 1,
                "n_parametros": int(b["n_parametros"]),
                "configuraciones_evaluadas": presupuestos.get(b["modelo"], np.nan),
                "segundos_entrenamiento_medio": np.nan}
        for m in METRICAS:
            fila[f"{m}_media"], fila[f"{m}_desv"] = float(b[m]), np.nan
            fila[f"{m}_mejor"] = fila[f"{m}_peor"] = float(b[m])
        filas.append(fila)
    return pd.DataFrame(filas)


def guardar_final(familia: str, mejor, info: dict, panel: Panel) -> Path:
    """Guarda la mejor semilla como ``models/m1_demanda/<familia>_final`` junto con la información de selección."""
    base = MODELOS / f"{familia}_final"
    guardar_modelo(mejor.pronosticador, panel.nombres_cal, base)
    ruta = base.with_name(f"{familia}_final_seleccion.json")
    ruta.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return base.with_suffix(".pt")


def elegir_familia_final(tablas_semillas: pd.DataFrame, infos: dict[str, dict] | None = None) -> dict:
    """Elige la red que usará la app: la familia cuya configuración final tiene el menor MASE medio de VALIDACIÓN.

    Solo mira ``particion == "val"`` (nunca prueba). ``infos`` (opcional) trae ``config_final_rank`` por familia; si
    no se da, se toma, por familia, la configuración con menor MASE medio de validación. Devuelve
    ``{"familia", "config_rank", "mase_val_media", "por_familia": {...}, "criterio"}``.
    """
    val = tablas_semillas[(tablas_semillas.particion == "val") & (tablas_semillas.subconjunto == "total")]
    if val.empty:
        raise ValueError("No hay filas de validación para elegir la familia final.")
    por_fam = {}
    for fam, g in val.groupby("familia"):
        medias = g.groupby("config_rank")["mase"].mean()
        rank = int(infos[fam]["config_final_rank"]) if infos and fam in infos else int(medias.idxmin())
        por_fam[fam] = {"config_rank": rank, "mase_val_media": float(medias.loc[rank])}
    mejor = min(por_fam, key=lambda f: por_fam[f]["mase_val_media"])
    return {"familia": mejor, "config_rank": por_fam[mejor]["config_rank"],
            "mase_val_media": por_fam[mejor]["mase_val_media"], "por_familia": por_fam,
            "criterio": "menor MASE medio de VALIDACIÓN entre las 30 semillas de la configuración final de cada familia"}


def escribir_modelo_final(eleccion: dict) -> Path:
    """Escribe ``models/m1_demanda/modelo_final.json`` (lo lee ``predict.py``)."""
    ruta = MODELOS / "modelo_final.json"
    ruta.write_text(json.dumps(eleccion, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return ruta


def semillas_desde_config(cfg_global: dict, n: int | None = None) -> list[int]:
    return lista_semillas(int(cfg_global["seed"]), int(n if n is not None else cfg_global["num_seeds"]))
