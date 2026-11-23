"""Pruebas de línea de comandos de m1_demanda.run (Persona 2).

Ejecuta el módulo COMO PROCESO (``python -m m1_demanda.run``). Un ``import`` no detecta errores que solo ocurren
al correrlo como script (p. ej. una función ``main`` mal definida), así que estas pruebas lo cubren.

Ejecutar:  python -m pytest tests/test_m1_cli.py -q
"""
import subprocess
import sys

import pytest

from m1_demanda.data.utils import RAIZ


def _correr(*args, timeout=180):
    return subprocess.run([sys.executable, "-m", "m1_demanda.run", *args], cwd=RAIZ, capture_output=True,
                          text=True, timeout=timeout)


def test_help_corre_y_lista_los_modos():
    r = _correr("--help")
    assert r.returncode == 0, r.stderr
    for modo in ("evaluate", "train", "search", "final", "demo"):
        assert modo in r.stdout
    for opcion in ("--device", "--seed", "--presupuesto", "--semillas", "--config"):
        assert opcion in r.stdout


def test_modo_obligatorio_y_modelo_invalido():
    assert _correr().returncode != 0
    r = _correr("--mode", "evaluate", "--modelo", "arima")
    assert r.returncode != 0 and "invalid choice" in r.stderr


def test_evaluate_ingenuo_escribe_la_tabla_de_metricas():
    r = _correr("--mode", "evaluate", "--modelo", "ingenuo", "--particiones", "val")
    assert r.returncode == 0, r.stderr
    assert "m1_metricas_ingenuo.csv" in r.stdout
    assert (RAIZ / "reports" / "tables" / "m1_metricas_ingenuo.csv").exists()


def test_configuracion_inexistente_da_un_error_que_nombra_el_archivo():
    r = _correr("--mode", "evaluate", "--modelo", "mlp", "--config", "configs/no_existe.yaml")
    assert r.returncode != 0                                        # configuración inexistente
    assert "no_existe" in r.stderr
