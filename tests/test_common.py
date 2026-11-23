"""Pruebas de las utilidades compartidas de common/ (escritas por la Persona 2).

Ejecutar:  python -m pytest tests/test_common.py -q
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from common.config_loader import cargar_yaml, claves_vacias
from common.multiseed import ejecutar_multisemilla, resumir
from common.search_budget import RegistroPresupuesto
from common.seeds import elegir_dispositivo, fijar_semillas, lista_semillas


# --- seeds ------------------------------------------------------------------
def test_misma_semilla_mismos_tensores():
    fijar_semillas(123)
    a = torch.randn(5), np.random.rand(3)
    fijar_semillas(123)
    b = torch.randn(5), np.random.rand(3)
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_semillas_distintas_tensores_distintos():
    fijar_semillas(1)
    a = torch.randn(5)
    fijar_semillas(2)
    assert not torch.equal(a, torch.randn(5))


def test_semilla_no_entera_falla():
    with pytest.raises(TypeError):
        fijar_semillas("42")


def test_lista_semillas():
    assert lista_semillas(10, 3) == [10, 11, 12]
    assert len(lista_semillas(2026, 30)) == 30
    with pytest.raises(ValueError):
        lista_semillas(1, 0)


def test_dispositivo():
    assert elegir_dispositivo("cpu").type == "cpu"
    assert elegir_dispositivo("auto").type in {"cpu", "mps", "cuda"}
    with pytest.raises(ValueError):
        elegir_dispositivo("tpu")


# --- multiseed --------------------------------------------------------------
def test_resumen_coincide_con_calculo_a_mano():
    por = pd.DataFrame({"semilla": [1, 2, 3, 4], "mase": [1.0, 2.0, 3.0, 4.0], "cob": [0.8, 0.9, 0.85, 0.95]})
    r = resumir(por, mayor_es_mejor=["cob"])
    assert r.loc["mase", "media"] == pytest.approx(2.5)
    assert r.loc["mase", "desv"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert (r.loc["mase", "mejor"], r.loc["mase", "peor"]) == (1.0, 4.0)
    assert (r.loc["cob", "mejor"], r.loc["cob", "peor"]) == (0.95, 0.8)


def test_multisemilla_30_semillas_reproducible():
    def fn(s):
        return {"valor": float(torch.randn(1))}

    p1, r1 = ejecutar_multisemilla(fn, lista_semillas(2026, 30))
    p2, _ = ejecutar_multisemilla(fn, lista_semillas(2026, 30))
    assert len(p1) == 30 and p1["semilla"].tolist() == lista_semillas(2026, 30)
    pd.testing.assert_frame_equal(p1, p2)
    assert set(r1.columns) == {"media", "desv", "mejor", "peor"}


def test_multisemilla_valida_entradas():
    with pytest.raises(ValueError):
        ejecutar_multisemilla(lambda s: {"a": 1.0}, [])
    with pytest.raises(TypeError):
        ejecutar_multisemilla(lambda s: 1.0, [1])


# --- search_budget ----------------------------------------------------------
def test_registro_cuenta_configuraciones_y_guarda(tmp_path):
    reg = RegistroPresupuesto("mlp", presupuesto=2)
    for i in range(2):
        with reg.medir({"lr": 0.1 * (i + 1)}) as r:
            r["mase"] = 1.0 + i
    assert reg.n_configuraciones == 2 and reg.agotado()
    assert reg.tiempo_total_s >= 0
    with pytest.raises(RuntimeError):
        reg.registrar({"lr": 1}, 0.1)
    csv, js = reg.guardar(tmp_path / "busqueda")
    assert len(pd.read_csv(csv)) == 2
    assert json.loads(js.read_text(encoding="utf-8"))["n_configuraciones"] == 2


# --- config_loader ----------------------------------------------------------
def test_yaml_con_vacios_da_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\nb:\n  c:\n  d: [1, null]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"b\.c.*b\.d\[1\]"):
        cargar_yaml(p)
    assert cargar_yaml(p, permitir_vacios=True)["a"] == 1


def test_claves_vacias_anidadas():
    assert claves_vacias({"x": {"y": None}, "z": [None]}) == ["x.y", "z[0]"]


def test_global_yaml_valido():
    cfg = cargar_yaml("configs/global.yaml")
    assert cfg["num_seeds"] == 30 and isinstance(cfg["seed"], int)
