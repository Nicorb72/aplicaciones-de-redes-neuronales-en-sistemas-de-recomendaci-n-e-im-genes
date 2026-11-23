"""Propósito: Registro de presupuesto de búsqueda.
Responsable: Persona 5 (versión mínima escrita por la Persona 2 para el Módulo 1; pendiente de revisión).

Cuenta las configuraciones evaluadas y el tiempo de cómputo de una búsqueda de
hiperparámetros, para declarar el presupuesto en la comparación justa entre modelos.

    reg = RegistroPresupuesto("mlp")
    with reg.medir(config) as r:
        r["metrica"] = evaluar(config)
    reg.guardar("reports/tables/m1_busqueda_mlp")      # escribe .csv y .json
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


class RegistroPresupuesto:
    """Registra cada configuración evaluada con su tiempo y sus métricas."""

    def __init__(self, nombre: str, presupuesto: int | None = None):
        self.nombre = nombre
        self.presupuesto = presupuesto
        self.filas: list[dict] = []

    @property
    def n_configuraciones(self) -> int:
        return len(self.filas)

    @property
    def tiempo_total_s(self) -> float:
        return float(sum(f["tiempo_s"] for f in self.filas))

    def agotado(self) -> bool:
        return self.presupuesto is not None and self.n_configuraciones >= self.presupuesto

    def registrar(self, config: dict, tiempo_s: float, **metricas) -> None:
        if self.agotado():
            raise RuntimeError(
                f"Presupuesto agotado: {self.n_configuraciones}/{self.presupuesto} configuraciones."
            )
        self.filas.append({"indice": self.n_configuraciones, **config, **metricas, "tiempo_s": float(tiempo_s)})

    @contextmanager
    def medir(self, config: dict):
        """Mide el tiempo de un bloque y registra la configuración al salir.

        Dentro del bloque se puede escribir en el diccionario entregado
        (``r["mase"] = ...``) y esos valores se guardan como métricas.
        """
        metricas: dict = {}
        t0 = time.perf_counter()
        yield metricas
        self.registrar(config, time.perf_counter() - t0, **metricas)

    def tabla(self) -> pd.DataFrame:
        return pd.DataFrame(self.filas)

    def resumen(self) -> dict:
        return {
            "nombre": self.nombre,
            "presupuesto": self.presupuesto,
            "n_configuraciones": self.n_configuraciones,
            "tiempo_total_s": round(self.tiempo_total_s, 3),
        }

    def guardar(self, base: str | Path) -> tuple[Path, Path]:
        """Escribe ``<base>.csv`` (tabla) y ``<base>.json`` (resumen)."""
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        ruta_csv, ruta_json = base.with_suffix(".csv"), base.with_suffix(".json")
        self.tabla().to_csv(ruta_csv, index=False, encoding="utf-8")
        ruta_json.write_text(
            json.dumps(self.resumen(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return ruta_csv, ruta_json
