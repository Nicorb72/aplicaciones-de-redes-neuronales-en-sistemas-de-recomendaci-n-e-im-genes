"""Pruebas de search.py (Persona 2): muestreo, presupuesto y selección solo con validación.

Ejecutar:  python -m pytest tests/test_m1_search.py -q
"""
import json

import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda import search as sr
from m1_demanda import ventanas as vt
from m1_demanda.data.utils import cargar_config


@pytest.fixture(scope="module")
def cfg_datos():
    return cargar_config()


@pytest.fixture(scope="module")
def cfg_m1():
    return cargar_yaml("configs/m1_demanda.yaml")


@pytest.fixture(scope="module")
def panel(cfg_m1, cfg_datos):
    return vt.cargar_panel(cfg_m1, cfg_datos)


# --- muestreo -------------------------------------------------------------------------------
def test_muestreo_reproducible_con_la_misma_semilla(cfg_m1):
    a = sr.muestras(cfg_m1, "mlp", 10, semilla=5)
    b = sr.muestras(cfg_m1, "mlp", 10, semilla=5)
    c = sr.muestras(cfg_m1, "mlp", 10, semilla=6)
    assert a == b and a != c


def test_las_muestras_respetan_el_espacio(cfg_m1):
    for fam in ("mlp", "rnn"):
        esp = sr.espacio_familia(cfg_m1, fam)
        for m in sr.muestras(cfg_m1, fam, 60, semilla=1):
            assert list(m) == list(esp)
            for k, d in esp.items():
                (tipo, arg), = d.items()
                if tipo == "choice":
                    assert m[k] in arg
                else:
                    assert arg[0] <= m[k] <= arg[1]


def test_distribuciones_cubren_el_rango(cfg_m1):
    ms = sr.muestras(cfg_m1, "mlp", 400, semilla=0)
    lr = np.array([m["lr"] for m in ms])
    assert lr.min() < 1e-3 < lr.max() and lr.min() >= 3e-4 and lr.max() <= 1e-2
    assert {m["window"] for m in ms} == {28, 56} and {m["layers"] for m in ms} == {1, 2, 3}
    # loguniform: la mediana geométrica queda cerca de √(a·b), no de (a+b)/2
    assert np.exp(np.median(np.log(lr))) == pytest.approx(np.sqrt(3e-4 * 1e-2), rel=0.25)


def test_distribucion_invalida():
    rng = np.random.default_rng(0)
    for mala in ({"choice": []}, {"uniform": [2, 1]}, {"loguniform": [0, 1]}, {"gauss": [0, 1]},
                 {"choice": [1], "uniform": [0, 1]}):
        with pytest.raises(ValueError):
            sr.muestrear_config({"x": mala}, rng)


def test_familia_invalida(cfg_m1):
    with pytest.raises(ValueError, match="redes"):
        sr.espacio_familia(cfg_m1, "glm")


def test_conversion_a_hiperparametros(cfg_m1):
    hp = sr.a_hiperparametros(cfg_m1, {"window": 56, "units": 32, "dropout": 0.2, "weight_decay": 1e-5, "lr": 0.002,
                                       "batch_size": 64, "route_embedding_dim": 4, "layers": 1, "use_cal_hist": False})
    assert hp["window"] == 56 and hp["units"] == 32 and hp["regularization"] == {"dropout": 0.2, "weight_decay": 1e-5}
    assert hp["use_cal_hist"] is False and hp["lr"] == 0.002
    hp_r = sr.a_hiperparametros(cfg_m1, {"cell": "lstm", "hidden": 64, "rnn_layers": 2})
    assert hp_r["rnn"] == {"cell": "lstm", "hidden": 64, "layers": 2}
    assert cfg_m1["rnn"] == {"cell": "gru", "hidden": 32, "layers": 1}       # la configuración no se modifica


# --- selección solo con validación -------------------------------------------------------------------
def _tabla(particion="val"):
    return pd.DataFrame({"indice": [0, 1, 2, 3], "particion": particion, "mase": [0.9, 0.8, 0.8, np.nan],
                         "pinball_escalada": [0.2, 0.25, 0.22, 0.1]})


def test_seleccion_ordena_por_mase_con_desempate_por_pinball():
    m = sr.seleccionar_mejores(_tabla(), top_k=3)
    assert m["indice"].tolist() == [2, 1, 0]            # empate 0,8 -> gana el de menor pinball (2); NaN excluido


def test_el_selector_rechaza_datos_de_prueba():
    for part in ("test", "post_hueco", "train"):
        with pytest.raises(ValueError, match="validación"):
            sr.seleccionar_mejores(_tabla(part))
    mixta = pd.concat([_tabla("val"), _tabla("test")], ignore_index=True)
    with pytest.raises(ValueError, match="validación"):
        sr.seleccionar_mejores(mixta)
    with pytest.raises(ValueError, match="validación"):
        sr.seleccionar_mejores(_tabla().drop(columns="particion"))


# --- búsqueda real con presupuesto pequeño -------------------------------------------------------------
@pytest.fixture(scope="module")
def corrida(cfg_m1, cfg_datos, panel, tmp_path_factory):
    cfg = {**cfg_m1, "epochs": 3, "patience": 3}
    reg, tabla, lista = sr.buscar("mlp", cfg, cfg_datos, panel, presupuesto=3, semilla=11)
    return cfg, reg, tabla, lista


def test_se_respeta_el_presupuesto_y_se_registra_el_tiempo(corrida):
    cfg, reg, tabla, lista = corrida
    assert reg.n_configuraciones == 3 == len(tabla) == len(lista) and reg.agotado()
    assert (tabla["tiempo_s"] > 0).all() and reg.tiempo_total_s == pytest.approx(tabla["tiempo_s"].sum())
    assert (tabla["particion"] == "val").all()
    with pytest.raises(RuntimeError, match="Presupuesto agotado"):
        reg.registrar({"x": 1}, 0.1)


def test_la_busqueda_es_reproducible(corrida, cfg_datos, panel):
    cfg, _, tabla, _ = corrida
    _, tabla2, _ = sr.buscar("mlp", cfg, cfg_datos, panel, presupuesto=3, semilla=11)
    pd.testing.assert_series_equal(tabla["mase"], tabla2["mase"])
    pd.testing.assert_series_equal(tabla["pinball_escalada"], tabla2["pinball_escalada"])


def test_metricas_de_la_busqueda_son_las_de_validacion(corrida, cfg_datos, panel):
    """La tabla trae MASE de validación finito; las columnas de parámetros coinciden con el espacio."""
    cfg, _, tabla, _ = corrida
    assert np.isfinite(tabla[["mase", "pinball_escalada", "cobertura"]]).all().all()
    assert set(sr.espacio_familia(cfg, "mlp")) <= set(tabla.columns)
    assert (tabla["n_parametros"] > 0).all() and (tabla["epocas"] >= 1).all()


def test_guardado_de_resultados(corrida, cfg_datos, tmp_path, monkeypatch):
    cfg, reg, tabla, _ = corrida
    monkeypatch.setattr(sr, "MODELOS", tmp_path / "modelos")
    csv, mejor = sr.guardar_busqueda("mlp", reg, tabla, cfg, semilla=11, carpeta_tablas=tmp_path / "tablas")
    assert len(pd.read_csv(csv)) == 3
    resumen = json.loads((tmp_path / "tablas" / "m1_busqueda_mlp.json").read_text(encoding="utf-8"))
    assert resumen["n_configuraciones"] == 3 and resumen["presupuesto"] == 3
    d = json.loads(mejor.read_text(encoding="utf-8"))
    assert len(d["configuraciones"]) == cfg["search"]["top_k"] == 2
    c1, c2 = d["configuraciones"]
    assert c1["rank"] == 1 and c1["metricas_validacion"]["mase"] <= c2["metricas_validacion"]["mase"]
    assert c1["hp"]["window"] in (28, 56) and "regularization" in c1["hp"]
    assert tabla["mase"].min() == pytest.approx(c1["metricas_validacion"]["mase"])
    monkeypatch.setattr(sr, "MODELOS", tmp_path / "modelos")
    assert sr.cargar_mejores("mlp")[0]["rank"] == 1


def test_cargar_mejores_sin_busqueda(monkeypatch, tmp_path):
    monkeypatch.setattr(sr, "MODELOS", tmp_path)
    with pytest.raises(FileNotFoundError, match="búsqueda"):
        sr.cargar_mejores("rnn")
