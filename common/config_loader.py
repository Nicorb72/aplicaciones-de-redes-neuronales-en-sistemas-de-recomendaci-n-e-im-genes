"""Propósito: Carga de configuraciones YAML compartidas.
Responsable: Persona 5 (versión mínima escrita por la Persona 2 para el Módulo 1; pendiente de revisión).

``cargar_yaml`` lee un YAML con ``yaml.safe_load`` y, por defecto, rechaza valores vacíos
(``null``) con un mensaje que indica la ruta exacta de cada clave vacía. Así un parámetro
sin definir falla al cargar y no más tarde dentro de un modelo.

Los identificadores con ceros a la izquierda (códigos DIVIPOLA) deben ir entre comillas
en el YAML: PyYAML interpreta ``05615`` como el entero octal 2957.
"""
from __future__ import annotations

from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def claves_vacias(obj, prefijo: str = "") -> list[str]:
    """Devuelve las rutas (``a.b.c``, ``lista[2]``) de los valores ``None`` de una estructura."""
    vacias: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            ruta = f"{prefijo}.{k}" if prefijo else str(k)
            if v is None:
                vacias.append(ruta)
            else:
                vacias += claves_vacias(v, ruta)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            ruta = f"{prefijo}[{i}]"
            if v is None:
                vacias.append(ruta)
            else:
                vacias += claves_vacias(v, ruta)
    return vacias


def cargar_yaml(ruta: str | Path, permitir_vacios: bool = False) -> dict:
    """Lee un YAML. Si ``permitir_vacios`` es False, lanza ``ValueError`` ante valores ``null``."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = RAIZ / ruta
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"El archivo {ruta} está vacío.")
    if not isinstance(cfg, dict):
        raise ValueError(f"El archivo {ruta} debe contener un diccionario en el nivel superior.")
    if not permitir_vacios:
        vacias = claves_vacias(cfg)
        if vacias:
            raise ValueError(f"Valores vacíos en {ruta.name}: {', '.join(vacias)}")
    return cfg
