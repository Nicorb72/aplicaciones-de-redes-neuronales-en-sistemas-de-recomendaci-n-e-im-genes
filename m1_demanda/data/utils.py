"""Utilidades compartidas por el pipeline de datos del Módulo 1.

Responsable: Persona 1.
"""
from __future__ import annotations

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
CONFIG_DEFECTO = RAIZ / "configs" / "m1_datos.yaml"


def cargar_config(ruta: str | Path | None = None) -> dict:
    """Lee el YAML de datos del módulo 1.

    Los códigos DIVIPOLA deben ir entre comillas en el YAML: PyYAML interpreta
    ``05615`` como un entero octal (2957). Por eso se valida aquí.
    """
    ruta = Path(ruta) if ruta else CONFIG_DEFECTO
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for par in cfg["seleccion_rutas"]["corredores"]:
        for codigo in par:
            if not (isinstance(codigo, str) and len(codigo) == 5 and codigo.isdigit()):
                raise ValueError(
                    f"Código DIVIPOLA inválido en el config: {codigo!r}. "
                    "Escríbalo entre comillas, p. ej. \"05001\"."
                )
    return cfg


def ruta_abs(cfg: dict, clave: str) -> Path:
    """Devuelve (y crea) una carpeta de salida declarada en ``rutas_salida``."""
    p = RAIZ / cfg["rutas_salida"][clave]
    p.mkdir(parents=True, exist_ok=True)
    return p


def id_ruta(origen: str, destino: str) -> str:
    return f"{origen}-{destino}"
