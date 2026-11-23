"""Propósito: Utilidades compartidas para semillas reproducibles y elección de dispositivo.
Responsable: Persona 5 (versión mínima escrita por la Persona 2 para el Módulo 1; pendiente de revisión).

Uso
---
    from common.seeds import fijar_semillas, elegir_dispositivo, lista_semillas
    fijar_semillas(1984)
    device = elegir_dispositivo("cpu")      # "cpu" | "mps" | "cuda" | "auto"
    semillas = lista_semillas(base=2026, n=30)

Notas
-----
* El dispositivo por defecto es CPU: los modelos del Módulo 1 son pequeños y MPS no
  garantiza resultados idénticos con la misma semilla (se verifica en la Tarea 9).
* ``"auto"`` sigue el orden del profesor: cuda -> mps -> cpu.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

DISPOSITIVOS_VALIDOS = ("cpu", "mps", "cuda", "auto")


def fijar_semillas(semilla: int, determinista: bool = True) -> None:
    """Fija las semillas de ``random``, NumPy y PyTorch.

    Con ``determinista=True`` pide a PyTorch algoritmos deterministas (con aviso, no
    error, si alguna operación no tiene versión determinista).
    """
    if not isinstance(semilla, (int, np.integer)) or isinstance(semilla, bool):
        raise TypeError(f"La semilla debe ser un entero, se recibió {semilla!r}.")
    semilla = int(semilla)
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla % (2**32))
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    if determinista:
        torch.use_deterministic_algorithms(True, warn_only=True)


def lista_semillas(base: int, n: int) -> list[int]:
    """Devuelve ``n`` semillas consecutivas a partir de ``base`` (base, base+1, ...)."""
    if n < 1:
        raise ValueError("n debe ser al menos 1.")
    return [int(base) + i for i in range(n)]


def elegir_dispositivo(preferencia: str = "cpu") -> torch.device:
    """Resuelve la preferencia de dispositivo.

    ``"cpu"``/``"mps"``/``"cuda"`` piden ese dispositivo y fallan con un mensaje claro si
    no está disponible. ``"auto"`` elige cuda, luego mps y luego cpu.
    """
    pref = str(preferencia).lower()
    if pref not in DISPOSITIVOS_VALIDOS:
        raise ValueError(f"Dispositivo {preferencia!r} inválido; use uno de {DISPOSITIVOS_VALIDOS}.")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió cuda, pero no está disponible.")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Se pidió mps, pero no está disponible.")
    return torch.device(pref)
