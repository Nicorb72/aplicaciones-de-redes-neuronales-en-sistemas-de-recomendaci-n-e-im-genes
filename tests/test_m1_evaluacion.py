"""Pruebas de evaluate.py y del ingenuo estacional (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_evaluacion.py -q
"""
import copy

import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.baselines import IngenuoEstacional, rezago_ingenuo
from m1_demanda.data.utils import cargar_config
from m1_demanda.evaluate import escala_mase, evaluar, evaluar_modelo, pinball
from m1_demanda.ventanas import Ventanas


@pytest.fixture(scope="module")
def cfg_datos():
    return cargar_config()


@pytest.fixture(scope="module")
def cfg_m1():
    return cargar_yaml("configs/m1_demanda.yaml")


@pytest.fixture(scope="module")
def panel(cfg_m1, cfg_datos):
    return vt.cargar_panel(cfg_m1, cfg_datos)


def _ventanas_sinteticas(y, mask, ruta_idx=None, alta=None):
    N, H = y.shape
    return Ventanas(
        origen=np.array(["2023-10-01"] * N, dtype="datetime64[D]"),
        ruta_idx=np.zeros(N, dtype=int) if ruta_idx is None else ruta_idx,
        x_hist=np.zeros((N, 2), np.float32), cal_hist=np.zeros((N, 2, 1), np.float32),
        cal_fut=np.zeros((N, H, 1), np.float32), nivel=np.ones(N, np.float32),
        y=np.asarray(y, np.float32), mask=np.asarray(mask, np.float32),
        alta=np.zeros((N, H), bool) if alta is None else alta,
    )


# --- métricas contra ejemplos a mano ------------------------------------------
def test_metricas_ejemplo_a_mano():
    y = np.array([[10.0, 20.0, 30.0]])
    mu = np.array([[12.0, 18.0, 30.0]])
    q = np.array([[15.0, 19.0, 40.0]])
    v = _ventanas_sinteticas(y, np.ones((1, 3)))
    tabla, hor = evaluar(v, mu, q, escala=np.array([2.0]), ids_rutas=["r"], tau=0.9)
    t = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].iloc[0]
    assert t.mae == pytest.approx(4 / 3)                       # (2 + 2 + 0) / 3
    assert t.rmse == pytest.approx(np.sqrt(8 / 3))            # sqrt((4 + 4 + 0) / 3)
    assert t.mase == pytest.approx((4 / 3) / 2.0)
    assert t.cobertura == pytest.approx(2 / 3)                 # 10<=15 si, 20<=19 no, 30<=40 si
    # pinball(0,9): d = -5 -> -0,1*-5 = 0,5 ; d = 1 -> 0,9 ; d = -10 -> 1,0 ; media = 2,4 / 3
    assert t.pinball == pytest.approx((0.5 + 0.9 + 1.0) / 3)
    assert list(hor.horizonte) == [1, 2, 3]
    assert hor.mae.tolist() == [2.0, 2.0, 0.0]


def test_mascara_excluye_dias():
    y = np.array([[10.0, 20.0, 1000.0]])
    mu = np.array([[10.0, 20.0, 0.0]])
    v = _ventanas_sinteticas(y, [[1, 1, 0]])
    tabla, _ = evaluar(v, mu, mu, np.array([1.0]), ["r"])
    t = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].iloc[0]
    assert t.n_dias == 2 and t.mae == 0.0


def test_subconjuntos_alta_y_base():
    y = np.array([[10.0, 10.0]])
    mu = np.array([[10.0, 14.0]])
    alta = np.array([[False, True]])
    v = _ventanas_sinteticas(y, np.ones((1, 2)), alta=alta)
    tabla, _ = evaluar(v, mu, mu, np.array([1.0]), ["r"])
    g = tabla[tabla.ruta_id == "TODAS"].set_index("subconjunto")
    assert g.loc["base", "mae"] == 0.0 and g.loc["alta", "mae"] == 4.0 and g.loc["total", "mae"] == 2.0


def test_mase_total_es_promedio_de_mase_por_ruta():
    y = np.array([[10.0], [10.0]])
    mu = np.array([[12.0], [20.0]])
    v = _ventanas_sinteticas(y, np.ones((2, 1)), ruta_idx=np.array([0, 1]))
    tabla, _ = evaluar(v, mu, mu, np.array([2.0, 10.0]), ["a", "b"])
    t = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].iloc[0]
    assert t.mase == pytest.approx((2 / 2 + 10 / 10) / 2)      # MASE por ruta: 1 y 1


def test_forma_incorrecta_y_nan_dan_error():
    v = _ventanas_sinteticas(np.ones((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="forma"):
        evaluar(v, np.ones((1, 2)), np.ones((1, 2)), np.array([1.0]), ["r"])
    with pytest.raises(ValueError, match="finitos"):
        evaluar(v, np.full((1, 3), np.nan), np.ones((1, 3)), np.array([1.0]), ["r"])


def test_cobertura_de_un_cuantil_conocido():
    """Si q es el cuantil 0,9 verdadero de la distribución, la cobertura debe ser ≈ 0,90."""
    rng = np.random.default_rng(0)
    y = rng.poisson(200, size=(2000, 30)).astype(float)
    q = np.full_like(y, np.quantile(y, 0.9))
    v = _ventanas_sinteticas(y, np.ones_like(y))
    tabla, _ = evaluar(v, y, q, np.array([1.0]), ["r"])
    cob = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].cobertura.iloc[0]
    assert cob == pytest.approx(0.9, abs=0.01)


def test_pinball_signo_correcto():
    # Con τ = 0,9, subestimar (q < y) cuesta 9 veces más que sobrestimar la misma cantidad.
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9)[0] == pytest.approx(1.8)
    assert pinball(np.array([10.0]), np.array([12.0]), 0.9)[0] == pytest.approx(0.2)


def test_minimizar_pinball_sobre_constante_da_el_cuantil():
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 100.0, size=5000)
    cands = np.linspace(y.min(), y.max(), 2000)
    mejor = cands[np.argmin([pinball(y, np.full_like(y, c), 0.9).mean() for c in cands])]
    assert mejor == pytest.approx(np.quantile(y, 0.9), rel=0.02)


# --- escala MASE -----------------------------------------------------------------
def test_escala_mase_solo_usa_entrenamiento(panel, cfg_datos):
    esc = escala_mase(panel, cfg_datos)
    assert esc.shape == (10,) and (esc > 0).all()
    # Cambiar datos de prueba (octubre-diciembre 2023) no altera la escala.
    alterado = copy.copy(panel)
    alterado.y_obs = panel.y_obs.copy()
    alterado.y_obs[panel.idx("2023-10-01"):panel.idx("2023-12-31") + 1] = 1e9
    np.testing.assert_array_equal(esc, escala_mase(alterado, cfg_datos))


# --- ingenuo estacional -----------------------------------------------------------
def test_rezago_ingenuo():
    # k = 0 es el origen (h = 1). Rezago = 7 para h = 1..7, 14 para h = 8..14, etc.
    assert [rezago_ingenuo(k) for k in (0, 6, 7, 13, 14, 29)] == [7, 7, 14, 14, 21, 35]
    assert all(rezago_ingenuo(k) > k for k in range(30))       # nunca usa un dato desde el origen


def test_ingenuo_repite_la_ultima_semana_y_no_usa_datos_desde_el_origen(panel, cfg_m1):
    origen = pd.Timestamp("2023-11-05")
    v = vt.construir_ventanas(panel, cfg_m1, [origen])
    mu = IngenuoEstacional.puntual(panel, v)
    i = panel.idx(origen)
    for n in range(len(v)):
        r = v.ruta_idx[n]
        ultima_semana = panel.y_imp[i - 7:i, r]
        np.testing.assert_array_equal(mu[n], np.resize(ultima_semana, 30))
    # Corromper todo dato desde el origen no cambia el pronóstico.
    alterado = copy.copy(panel)
    alterado.y_imp = panel.y_imp.copy()
    alterado.y_imp[i:] = 1e7
    np.testing.assert_array_equal(mu, IngenuoEstacional.puntual(alterado, v))


def test_ingenuo_exige_calibracion(panel, cfg_m1):
    v = vt.construir_ventanas(panel, cfg_m1, [pd.Timestamp("2023-11-05")])
    with pytest.raises(RuntimeError, match="ajustar"):
        IngenuoEstacional().predecir(panel, v)


def test_p90_se_calibra_con_validacion_y_cubre_alrededor_de_90(panel, cfg_m1, cfg_datos):
    m = IngenuoEstacional(0.9).ajustar(panel, cfg_m1, cfg_datos)
    assert m.razon_p90.shape == (30,) and (m.razon_p90 >= 1).all()
    tabla, hor, v, mu, q = evaluar_modelo(m, panel, cfg_m1, cfg_datos, "val")
    assert (q >= mu).all()
    cob = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].cobertura.iloc[0]
    assert cob == pytest.approx(0.9, abs=0.03)
    assert len(hor) == 30 and hor.horizonte.tolist() == list(range(1, 31))
