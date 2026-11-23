"""Propósito: Medir CPU frente a MPS/CUDA para fijar el dispositivo por defecto con evidencia (Tarea 9).
Responsable: Persona 2.

Para cada dispositivo y familia entrena unas pocas épocas DOS veces con la misma semilla y mide:

* ``ms_por_epoca``: tiempo medio por época (mediana de las corridas).
* ``dif_max_repeticion``: máxima diferencia absoluta entre los pronósticos de las dos corridas con la misma
  semilla en el mismo dispositivo (0 = reproducible bit a bit).
* ``dif_max_vs_cpu``: máxima diferencia relativa de los pronósticos frente a la CPU (el cambio de dispositivo
  no tiene por qué dar resultados idénticos, pero sí parecidos).

Uso: ``python -m m1_demanda.benchmark_dispositivo [--familias mlp rnn] [--epocas 5]``
Salida: ``reports/tables/m1_benchmark_dispositivo.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common.config_loader import cargar_yaml
from common.seeds import elegir_dispositivo
from m1_demanda.data.utils import RAIZ, cargar_config, ruta_abs
from m1_demanda.train import entrenar, hp_desde_config
from m1_demanda.ventanas import CONFIG_M1, cargar_panel, ventanas_particion


def dispositivos_disponibles() -> list[str]:
    d = ["cpu"]
    if torch.backends.mps.is_available():
        d.append("mps")
    if torch.cuda.is_available():
        d.append("cuda")
    return d


def medir(familias: list[str], epocas: int, semilla: int = 2026) -> pd.DataFrame:
    cfg_m1, cfg_datos = cargar_yaml(CONFIG_M1), cargar_config()
    panel = cargar_panel(cfg_m1, cfg_datos)
    tr = ventanas_particion(panel, cfg_m1, cfg_datos, "train")
    va = ventanas_particion(panel, cfg_m1, cfg_datos, "val", paso_dias=1)
    ev = ventanas_particion(panel, cfg_m1, cfg_datos, "val")
    filas = []
    for fam in familias:
        referencia = None
        for dev in dispositivos_disponibles():
            hp = hp_desde_config(cfg_m1, epochs=epocas, patience=epocas + 1)
            corridas, tiempos = [], []
            for _ in range(2):
                r = entrenar(fam, hp, cfg_m1, cfg_datos, panel, semilla, elegir_dispositivo(dev),
                             ventanas_train=tr, ventanas_val=va)
                mu, q = r.pronosticador.predecir(panel, ev)
                corridas.append(np.concatenate([mu.ravel(), q.ravel()]))
                tiempos.append(r.segundos_por_epoca * 1000)
            if dev == "cpu":
                referencia = corridas[0]
            filas.append({
                "familia": fam, "dispositivo": dev, "epocas": epocas, "n_parametros": r.n_parametros,
                "ms_por_epoca": float(np.median(tiempos)),
                "dif_max_repeticion": float(np.max(np.abs(corridas[0] - corridas[1]))),
                "dif_max_vs_cpu": float(np.max(np.abs(corridas[0] - referencia) / np.maximum(np.abs(referencia), 1.0))),
            })
    return pd.DataFrame(filas)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="CPU vs MPS/CUDA: velocidad y reproducibilidad.")
    ap.add_argument("--familias", nargs="+", default=["mlp"])
    ap.add_argument("--epocas", type=int, default=5)
    args = ap.parse_args(argv)
    tabla = medir(args.familias, args.epocas)
    ruta = ruta_abs(cargar_config(), "tablas") / "m1_benchmark_dispositivo.csv"
    tabla.to_csv(ruta, index=False, encoding="utf-8")
    with pd.option_context("display.width", 200, "display.float_format", "{:.6g}".format):
        print(tabla.to_string(index=False))
    print(f"\nEscrito: {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
