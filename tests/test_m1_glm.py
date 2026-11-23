"""Pruebas del baseline GLM Poisson (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_glm.py -q
"""
import copy

import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.baselines import CAL_REDUNDANTES, SEMANAS_REZAGO, GLMPoisson
from m1_demanda.data.utils import cargar_config
from m1_demanda.evaluate import evaluar_modelo


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
def glm(panel, cfg_m1, cfg_datos):
    return GLMPoisson(0.9).ajustar(panel, cfg_m1, cfg_datos)


def test_matriz_de_diseno_tiene_rango_completo_y_sin_columnas_prohibidas(glm, panel, cfg_m1, cfg_datos):
    # El rango completo se exige en ENTRENAMIENTO, donde se ajusta el modelo. En validación (jul-sep) hay
    # variables de calendario constantes en cero (Semana Santa, receso de octubre, fin de año, elecciones)
    # y su rango es menor sin que eso sea un defecto.
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "train")
    X = glm.disenio(v, list(panel.nombres_cal))
    Xr = X[v.ruta_idx == 0].reshape(-1, X.shape[-1])
    assert np.linalg.matrix_rank(Xr) == X.shape[-1]               # sin colinealidad exacta
    assert len(glm.columnas) == X.shape[-1]
    for c in glm.columnas:
        assert c not in cfg_m1["forbidden_inputs"] and not c.startswith("despachos")
        assert c not in CAL_REDUNDANTES
    assert glm.columnas[0] == "intercepto" and sum(c.startswith("lag_") for c in glm.columnas) == SEMANAS_REZAGO


def test_rezagos_son_el_mismo_dia_de_la_semana(glm, panel, cfg_m1):
    """El rezago de 1 semana del día objetivo k debe ser el día o+k-7 (si k<7) del historial."""
    o = pd.Timestamp("2023-11-05")
    v = vt.construir_ventanas(panel, cfg_m1, [o], rutas=[0])
    X = glm.disenio(v, list(panel.nombres_cal))
    j = glm.columnas.index("lag_1sem")
    i = panel.idx(o)
    nivel = v.nivel[0]
    for k in range(7):
        esperado = np.log((panel.y_imp[i - 7 + k, 0] + 1) / (nivel + 1))
        assert X[0, k, j] == pytest.approx(esperado, abs=1e-5)


def test_ajuste_recupera_coeficientes_en_datos_poisson_sinteticos():
    rng = np.random.default_rng(0)
    beta = np.array([0.1, 0.5, -0.3, 0.2])
    X = np.column_stack([np.ones(20000), rng.normal(size=(20000, 3))])
    nivel = rng.uniform(100, 300, size=20000)
    y = rng.poisson(nivel * np.exp(X @ beta)).astype(float)
    est, ok = GLMPoisson._ajustar_ruta(X, y, np.log(nivel))
    assert ok
    np.testing.assert_allclose(est, beta, atol=0.01)


def test_todas_las_rutas_convergen(glm):
    assert all(glm.convergio.values()) and len(glm.betas) == 10
    assert glm.n_parametros == 10 * len(glm.columnas) and glm.n_configuraciones == 1


def test_predicciones_positivas_acotadas_y_p90_sobre_la_media(glm, panel, cfg_m1, cfg_datos):
    tabla, hor, v, mu, q = evaluar_modelo(glm, panel, cfg_m1, cfg_datos, "test")
    assert np.isfinite(mu).all() and (mu > 0).all() and (q >= mu).all()
    ratio = mu / v.nivel[:, None]
    assert ratio.min() >= np.exp(-glm.tope) - 1e-9 and ratio.max() <= np.exp(glm.tope) + 1e-9


def test_glm_supera_al_ingenuo_en_validacion_y_el_p90_cubre_cerca_de_090(glm, panel, cfg_m1, cfg_datos):
    from m1_demanda.baselines import IngenuoEstacional
    ing = IngenuoEstacional(0.9).ajustar(panel, cfg_m1, cfg_datos)
    t_g = evaluar_modelo(glm, panel, cfg_m1, cfg_datos, "val")[0]
    t_i = evaluar_modelo(ing, panel, cfg_m1, cfg_datos, "val")[0]
    g = t_g[(t_g.subconjunto == "total") & (t_g.ruta_id == "TODAS")].iloc[0]
    i = t_i[(t_i.subconjunto == "total") & (t_i.ruta_id == "TODAS")].iloc[0]
    assert g.mase < i.mase
    assert g.cobertura == pytest.approx(0.9, abs=0.03)


def test_no_usa_datos_desde_el_origen(glm, panel, cfg_m1):
    """Corromper todo dato desde el origen no cambia el pronóstico del GLM."""
    o = pd.Timestamp("2023-11-05")
    v = vt.construir_ventanas(panel, cfg_m1, [o])
    mu, _ = glm.predecir(panel, v)
    i = panel.idx(o)
    alterado = copy.copy(panel)
    alterado.y_imp = panel.y_imp.copy()
    alterado.y_obs = panel.y_obs.copy()
    alterado.y_imp[i:] = 1e7
    alterado.y_obs[i:] = 1e7
    v2 = vt.construir_ventanas(alterado, cfg_m1, [o])
    mu2, _ = glm.predecir(alterado, v2)
    np.testing.assert_allclose(mu, mu2)


def test_exige_ajuste_previo(panel, cfg_m1):
    v = vt.construir_ventanas(panel, cfg_m1, [pd.Timestamp("2023-11-05")])
    with pytest.raises(RuntimeError, match="ajustar"):
        GLMPoisson().predecir(panel, v)
