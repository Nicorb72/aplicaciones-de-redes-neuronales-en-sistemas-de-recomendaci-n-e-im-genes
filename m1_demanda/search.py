"""Propósito: Búsqueda aleatoria de hiperparámetros con presupuesto explícito (decisión D8).
Responsable: Persona 2.

Reglas (para que la comparación entre modelos sea justa y no haya fuga):

* **Presupuesto declarado**: ``search.budget`` configuraciones por familia (30 por defecto), UNA semilla cada
  una, con parada temprana. Cada configuración evaluada, su tiempo y su métrica quedan en
  ``reports/tables/m1_busqueda_<familia>.csv`` y el resumen (número de configuraciones y tiempo total) en el
  ``.json`` homónimo, vía ``common.search_budget.RegistroPresupuesto``.
* **Solo validación**: cada configuración se evalúa con MASE sobre los orígenes SEMANALES de la partición de
  validación. La partición de prueba no se usa aquí: ``seleccionar_mejores`` lanza ``ValueError`` si recibe
  filas de otra partición.
* **Selección**: menor MASE medio en validación; desempate por la pérdida cuantílica escalada.
* **Reproducible**: el muestreo sale de ``np.random.default_rng(semilla)``; con la misma semilla y espacio se
  obtiene la misma lista de configuraciones.
* Las ``top_k`` mejores se guardan en ``models/m1_demanda/mejor_config_<familia>.json`` para correrlas con
  30 semillas (``multiseed.py``).

Uso: ``python -m m1_demanda.run --mode search --modelo mlp [--presupuesto 3]``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common.search_budget import RegistroPresupuesto
from m1_demanda.evaluate import evaluar_modelo
from m1_demanda.train import MODELOS, cfg_con_ventana, entrenar, hp_desde_config
from m1_demanda.ventanas import Panel, ventanas_particion

CLAVES_MUESTRA_RNN = {"cell": "cell", "hidden": "hidden", "rnn_layers": "layers"}


# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
def espacio_familia(cfg_m1: dict, familia: str) -> dict:
    """Espacio de búsqueda de una familia: parámetros comunes + propios."""
    if familia not in ("mlp", "rnn"):
        raise ValueError(f"La búsqueda aplica a las redes ('mlp' o 'rnn'), no a {familia!r}.")
    s = cfg_m1["search"]
    return {**s["common"], **s[familia]}


def _muestrear_uno(dist: dict, rng: np.random.Generator):
    if len(dist) != 1:
        raise ValueError(f"Distribución inválida {dist!r}: use exactamente una de choice, uniform, loguniform.")
    (tipo, arg), = dist.items()
    if tipo == "choice":
        if not arg:
            raise ValueError("choice no puede estar vacía.")
        v = arg[int(rng.integers(len(arg)))]
        return v.item() if hasattr(v, "item") else v
    if tipo in ("uniform", "loguniform"):
        a, b = float(arg[0]), float(arg[1])
        if not a < b or (tipo == "loguniform" and a <= 0):
            raise ValueError(f"Rango inválido para {tipo}: {arg!r}.")
        if tipo == "uniform":
            return float(rng.uniform(a, b))
        return float(math.exp(rng.uniform(math.log(a), math.log(b))))
    raise ValueError(f"Distribución desconocida: {tipo!r}.")


def muestrear_config(espacio: dict, rng: np.random.Generator) -> dict:
    """Una configuración aleatoria: ``{parametro: valor}`` (el orden de las claves es el del espacio)."""
    return {k: _muestrear_uno(d, rng) for k, d in espacio.items()}


def muestras(cfg_m1: dict, familia: str, n: int, semilla: int) -> list[dict]:
    """Lista reproducible de ``n`` configuraciones."""
    rng = np.random.default_rng(int(semilla))
    esp = espacio_familia(cfg_m1, familia)
    return [muestrear_config(esp, rng) for _ in range(n)]


def a_hiperparametros(cfg_m1: dict, muestra: dict) -> dict:
    """Convierte una muestra plana en el diccionario de hiperparámetros que usa ``train.entrenar``."""
    cambios: dict = {}
    reg: dict = {}
    rnn: dict = {}
    for k, v in muestra.items():
        if k in ("dropout", "weight_decay"):
            reg[k] = v
        elif k in CLAVES_MUESTRA_RNN:
            rnn[CLAVES_MUESTRA_RNN[k]] = v
        else:
            cambios[k] = v
    if reg:
        cambios["regularization"] = reg
    if rnn:
        cambios["rnn"] = rnn
    return hp_desde_config(cfg_m1, **cambios)


# ---------------------------------------------------------------------------
# Selección
# ---------------------------------------------------------------------------
def seleccionar_mejores(tabla: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    """Las ``top_k`` mejores filas por MASE en validación (desempate: pinball escalada).

    Lanza ``ValueError`` si la tabla contiene filas que no sean de validación (la prueba no puede intervenir).
    """
    if "particion" not in tabla.columns or (tabla["particion"] != "val").any():
        raise ValueError("La selección de hiperparámetros solo puede usar la partición de validación.")
    if tabla.empty:
        raise ValueError("No hay configuraciones para seleccionar.")
    ok = tabla.dropna(subset=["mase", "pinball_escalada"])
    return ok.sort_values(["mase", "pinball_escalada"], kind="stable").head(int(top_k)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------
def buscar(familia: str, cfg_m1: dict, cfg_datos: dict, panel: Panel, *, presupuesto: int | None = None,
           semilla: int = 2026, device="cpu", verbose: bool = False) -> tuple[RegistroPresupuesto, pd.DataFrame, list[dict]]:
    """Ejecuta la búsqueda. Devuelve ``(registro, tabla, muestras)``.

    ``tabla`` tiene una fila por configuración con sus parámetros y las métricas de validación.
    """
    n = int(presupuesto if presupuesto is not None else cfg_m1["search"]["budget"])
    lista = muestras(cfg_m1, familia, n, semilla)
    reg = RegistroPresupuesto(familia, presupuesto=n)
    cache: dict[int, tuple] = {}

    for i, muestra in enumerate(lista):
        hp = a_hiperparametros(cfg_m1, muestra)
        w = int(hp["window"])
        cfg_w = cfg_con_ventana(cfg_m1, w) if w != int(cfg_m1["window"]) else cfg_m1
        if w not in cache:       # las ventanas solo dependen de la longitud: se construyen una vez por valor
            cache[w] = (ventanas_particion(panel, cfg_w, cfg_datos, "train"),
                        ventanas_particion(panel, cfg_w, cfg_datos, "val", paso_dias=1))
        tr, va = cache[w]
        with reg.medir(muestra) as r:
            res = entrenar(familia, hp, cfg_m1, cfg_datos, panel, semilla=semilla + i, device=device,
                           ventanas_train=tr, ventanas_val=va)
            tabla_v, _, _, _, _ = evaluar_modelo(res.pronosticador, panel, cfg_w, cfg_datos, "val")
            tot = tabla_v[(tabla_v.subconjunto == "total") & (tabla_v.ruta_id == "TODAS")].iloc[0]
            r.update({"particion": "val", "n_parametros": res.n_parametros, "epocas": len(res.historial),
                      "mejor_epoca": res.mejor_epoca, "val_perdida": res.mejor_val, "mase": float(tot["mase"]),
                      "mae": float(tot["mae"]), "cobertura": float(tot["cobertura"]),
                      "pinball_escalada": float(tot["pinball_escalada"]), "semilla": semilla + i})
        if verbose:
            f = reg.filas[-1]
            print(f"[{familia}] {i + 1:2d}/{n}  MASE val {f['mase']:.3f}  pinball {f['pinball_escalada']:.3f}  "
                  f"cobertura {f['cobertura']:.3f}  {f['n_parametros']:,} par  {f['tiempo_s']:.0f} s", flush=True)
    return reg, reg.tabla(), lista


def guardar_busqueda(familia: str, reg: RegistroPresupuesto, tabla: pd.DataFrame, cfg_m1: dict, semilla: int,
                     carpeta_tablas: Path, top_k: int | None = None) -> tuple[Path, Path]:
    """Escribe la tabla, el resumen de presupuesto y ``mejor_config_<familia>.json`` con las ``top_k`` mejores."""
    k = int(top_k if top_k is not None else cfg_m1["search"]["top_k"])
    mejores = seleccionar_mejores(tabla, k)
    parametros = list(espacio_familia(cfg_m1, familia).keys())
    configuraciones = []
    for rank, fila in enumerate(mejores.to_dict("records"), start=1):
        muestra = {p: fila[p] for p in parametros}
        muestra = {p: (v.item() if hasattr(v, "item") else v) for p, v in muestra.items()}
        configuraciones.append({
            "rank": rank, "indice_busqueda": int(fila["indice"]), "muestra": muestra,
            "hp": a_hiperparametros(cfg_m1, muestra),
            "metricas_validacion": {m: float(fila[m]) for m in ("mase", "mae", "cobertura", "pinball_escalada",
                                                                  "val_perdida")},
            "n_parametros": int(fila["n_parametros"]),
        })
    resumen = {**reg.resumen(), "familia": familia, "semilla_busqueda": int(semilla),
               "criterio": "menor MASE medio en orígenes semanales de VALIDACIÓN; desempate por pinball escalada",
               "top_k": k, "configuraciones": configuraciones}
    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    reg.guardar(carpeta_tablas / f"m1_busqueda_{familia}")
    # el JSON de reg.guardar solo trae el resumen; se amplía con la selección
    (carpeta_tablas / f"m1_busqueda_{familia}.json").write_text(
        json.dumps({k_: v for k_, v in resumen.items() if k_ != "configuraciones"}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_mejor = MODELOS / f"mejor_config_{familia}.json"
    ruta_mejor.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return carpeta_tablas / f"m1_busqueda_{familia}.csv", ruta_mejor


def cargar_mejores(familia: str) -> list[dict]:
    """Lee las configuraciones seleccionadas (ordenadas de mejor a peor) de ``mejor_config_<familia>.json``."""
    ruta = MODELOS / f"mejor_config_{familia}.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}; ejecute la búsqueda primero: "
                                f"`python -m m1_demanda.run --mode search --modelo {familia}`.")
    return json.loads(ruta.read_text(encoding="utf-8"))["configuraciones"]
