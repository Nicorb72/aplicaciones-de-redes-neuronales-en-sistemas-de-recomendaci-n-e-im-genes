"""Pruebas de la MLP (nn_models.py) y del entrenamiento (train.py) (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_mlp.py -q
"""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.data.utils import cargar_config
from m1_demanda.evaluate import evaluar_modelo
from m1_demanda.losses import PerdidaCombinada, media_y_cuantil
from m1_demanda.nn_models import MLPDemanda, contar_parametros
from m1_demanda.train import (DatasetVentanas, cargar_modelo, entrenar, guardar_modelo, hp_desde_config,
                              validar)


@pytest.fixture(scope="module")
def cfg_datos():
    return cargar_config()


@pytest.fixture(scope="module")
def cfg_m1():
    return cargar_yaml("configs/m1_demanda.yaml")


@pytest.fixture(scope="module")
def panel(cfg_m1, cfg_datos):
    return vt.cargar_panel(cfg_m1, cfg_datos)


@pytest.fixture(scope="module")
def entrenado(panel, cfg_m1, cfg_datos):
    hp = hp_desde_config(cfg_m1, epochs=6, patience=10)
    return entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=7, device="cpu")


# --- arquitectura -----------------------------------------------------------------------
def test_dimensiones_de_salida_y_q_mayor_o_igual_que_mu():
    m = MLPDemanda(n_rutas=10, ventana=28, horizonte=30, n_cal=27, dim_embedding=3, capas=2, unidades=16)
    b = 5
    f, g = m(torch.randn(b, 28), torch.randn(b, 28, 27), torch.randn(b, 30, 27), torch.randint(0, 10, (b,)))
    assert f.shape == (b, 30) and g.shape == (b, 30)
    mu, q = media_y_cuantil(f, g, torch.rand(b) * 500 + 10)
    assert (mu > 0).all() and (q >= mu).all()


def test_conteo_de_parametros_a_mano():
    # entrada = L + L·F + H·F + dim = 28 + 756 + 810 + 3 = 1597; capas 1597→16, 16→16; cabezas 2·(16·30 + 30)
    m = MLPDemanda(n_rutas=10, ventana=28, horizonte=30, n_cal=27, dim_embedding=3, capas=2, unidades=16)
    esperado = 10 * 3 + (1597 * 16 + 16) + (16 * 16 + 16) + 2 * (16 * 30 + 30)
    assert contar_parametros(m) == esperado == sum(p.numel() for p in m.parameters())


def test_contar_parametros_solo_entrenables():
    m = MLPDemanda(10, 28, 30, 27, 3, 1, 8)
    for p in m.embedding.parameters():
        p.requires_grad = False
    assert contar_parametros(m) == sum(p.numel() for p in m.parameters()) - 10 * 3


def test_arranque_neutro():
    """Al inicio f = 0 (μ = ℓ) y el P90 es un múltiplo moderado de la media."""
    m = MLPDemanda(10, 28, 30, 27, 3, 2, 16).eval()
    f, g = m(torch.randn(4, 28), torch.randn(4, 28, 27), torch.randn(4, 30, 27), torch.zeros(4, dtype=torch.long))
    assert torch.allclose(f, torch.zeros_like(f))
    mu, q = media_y_cuantil(f, g, torch.ones(4))
    assert 1.0 < float((q / mu).detach().mean()) < 1.6


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        MLPDemanda(10, 28, 30, 27, capas=0)


# --- sobreajuste de un lote pequeño ------------------------------------------------------------
def test_sobreajusta_un_lote_pequeno(panel, cfg_m1, cfg_datos):
    torch.manual_seed(0)
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val").subconjunto(slice(0, 24))
    ds = DatasetVentanas(v)
    b = {k: t for k, t in ds.t.items()}
    m = MLPDemanda(len(panel.rutas), 28, 30, len(panel.nombres_cal), 3, 2, 64, dropout=0.0)
    perdida = PerdidaCombinada(0.9, 1.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def paso():
        f, g = m(b["x_hist"], b["cal_hist"], b["cal_fut"], b["ruta_idx"])
        return perdida(f, g, b["nivel"], b["y"], b["mask"])

    # La Poisson sin el término constante log y! no llega a 0: su mínimo es mean(r·(1 − log r)), r = y/ℓ,
    # alcanzado con f = log r en cada día. Un sobreajuste correcto debe cerrar casi toda la brecha hasta él.
    r = (b["y"] / b["nivel"][:, None]).clamp_min(1e-9)
    piso = float(((r * (1 - torch.log(r))) * b["mask"]).sum() / b["mask"].sum())
    inicial = [float(x) for x in paso()]
    for _ in range(1500):
        opt.zero_grad()
        paso()[0].backward()
        opt.step()
    final = [float(x) for x in paso()]
    assert final[1] >= piso - 1e-4                                   # nunca por debajo del mínimo teórico
    assert (inicial[1] - final[1]) / (inicial[1] - piso) > 0.9       # cierra > 90 % de la brecha
    assert final[2] < 0.2 * inicial[2]                                 # y el pinball cae con fuerza


# --- entrenamiento ---------------------------------------------------------------------------------
def test_entrenamiento_reduce_la_perdida_y_reporta_parametros(entrenado):
    h = entrenado.historial
    assert len(h) >= 1 and entrenado.n_parametros == contar_parametros(entrenado.pronosticador.modelo)
    assert h[-1]["train"] < h[0]["train"]
    assert 1 <= entrenado.mejor_epoca <= len(h)
    assert entrenado.pronosticador.meta["dispositivo"] == "cpu"


def test_misma_semilla_mismo_resultado(panel, cfg_m1, cfg_datos, entrenado):
    hp = hp_desde_config(cfg_m1, epochs=6, patience=10)
    otro = entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=7, device="cpu")
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val")
    a = entrenado.pronosticador.predecir(panel, v)
    b = otro.pronosticador.predecir(panel, v)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_semillas_distintas_dan_modelos_distintos(panel, cfg_m1, cfg_datos, entrenado):
    hp = hp_desde_config(cfg_m1, epochs=6, patience=10)
    otro = entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=8, device="cpu")
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val")
    assert not np.array_equal(entrenado.pronosticador.predecir(panel, v)[0], otro.pronosticador.predecir(panel, v)[0])


def test_la_parada_temprana_recupera_la_mejor_epoca(panel, cfg_m1, cfg_datos):
    hp = hp_desde_config(cfg_m1, epochs=40, patience=3, lr=0.05)      # lr alto: la validación se degrada pronto
    r = entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=3, device="cpu")
    va = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val", paso_dias=1)
    from m1_demanda.train import _hp_dataloader
    dl = _hp_dataloader(DatasetVentanas(va), 512, False, 0)
    val_final = validar(dl, r.pronosticador.modelo, PerdidaCombinada(0.9, 1.0), torch.device("cpu"))["total"]
    assert val_final == pytest.approx(r.mejor_val, rel=1e-5)
    assert len(r.historial) <= 40


def test_ventana_de_56_dias_reconstruye_las_ventanas(panel, cfg_m1, cfg_datos):
    hp = hp_desde_config(cfg_m1, window=56, epochs=2, patience=5)
    r = entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=1, device="cpu")
    assert r.pronosticador.hp["window"] == 56
    v56 = vt.ventanas_particion(panel, {**cfg_m1, "window": 56}, cfg_datos, "val")
    mu, q = r.pronosticador.predecir(panel, v56)
    assert mu.shape == (len(v56), 30)
    with pytest.raises(ValueError, match="56"):          # una ventana de 28 días no sirve para este modelo
        r.pronosticador.predecir(panel, vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val"))


# --- evaluación, persistencia y no uso de prueba -----------------------------------------------------------
def test_evaluacion_con_el_protocolo(entrenado, panel, cfg_m1, cfg_datos):
    tabla, hor, v, mu, q = evaluar_modelo(entrenado.pronosticador, panel, cfg_m1, cfg_datos, "val")
    assert np.isfinite(mu).all() and (q >= mu).all() and len(hor) == 30
    assert tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].mase.iloc[0] > 0


def test_guardar_y_recargar_da_el_mismo_pronostico(entrenado, panel, cfg_m1, cfg_datos, tmp_path):
    guardar_modelo(entrenado.pronosticador, panel.nombres_cal, tmp_path / "mlp")
    de_nuevo = cargar_modelo(tmp_path / "mlp", n_rutas=len(panel.rutas))
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val")
    a, b = entrenado.pronosticador.predecir(panel, v), de_nuevo.predecir(panel, v)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-6)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-6)
    assert de_nuevo.n_parametros == entrenado.n_parametros


def test_el_entrenamiento_no_toca_la_particion_de_prueba(panel, cfg_m1, cfg_datos):
    """Si se corrompe toda la prueba, el modelo entrenado no cambia (solo ve entrenamiento y validación)."""
    hp = hp_desde_config(cfg_m1, epochs=3, patience=5)
    base = entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=5, device="cpu")
    alterado = copy.copy(panel)
    alterado.y_imp, alterado.y_obs = panel.y_imp.copy(), panel.y_obs.copy()
    i0, i1 = alterado.idx("2023-10-01"), alterado.idx("2023-12-31")
    alterado.y_imp[i0:i1 + 1] = 1e7
    alterado.y_obs[i0:i1 + 1] = 1e7
    otro = entrenar("mlp", hp, cfg_m1, cfg_datos, alterado, semilla=5, device="cpu")
    for pa, pb in zip(base.pronosticador.modelo.parameters(), otro.pronosticador.modelo.parameters()):
        torch.testing.assert_close(pa, pb)


def test_evaluar_modelo_usa_la_ventana_del_modelo(panel, cfg_m1, cfg_datos):
    """Una red con ventana de 56 días se evalúa sin que el llamador cambie la configuración (ventana por defecto: 28).

    Defecto que había: ``evaluar_modelo`` construía siempre ventanas de 28 días y la red final (56) fallaba por CLI.
    """
    hp = hp_desde_config(cfg_m1, window=56, epochs=2, patience=3, units=16)
    r = entrenar("mlp", hp, cfg_m1, cfg_datos, panel, semilla=2, device="cpu")
    assert int(cfg_m1["window"]) == 28
    tabla, hor, v, mu, q = evaluar_modelo(r.pronosticador, panel, cfg_m1, cfg_datos, "val")
    assert v.x_hist.shape[1] == 56 and mu.shape == (len(v), 30) and len(hor) == 30
    # los orígenes y las rutas son los mismos que con ventana de 28 días (comparación justa)
    v28 = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val")
    assert set(v.origen) == set(v28.origen) and len(v) == len(v28)


def test_hp_desde_config_aplica_cambios(cfg_m1):
    hp = hp_desde_config(cfg_m1, units=8, regularization={"dropout": 0.3})
    assert hp["units"] == 8 and hp["regularization"]["dropout"] == 0.3
    assert hp["regularization"]["weight_decay"] == cfg_m1["regularization"]["weight_decay"]
    assert cfg_m1["units"] == 64                                     # no modifica la configuración original
