"""Pruebas de forecast_value.py y peak_season.py (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_valor.py -q
"""
import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.baselines import GLMPoisson, IngenuoEstacional
from m1_demanda.data.utils import cargar_config
from m1_demanda.operacion import forecast_value as fv
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.peak_season import indicios_censura, vehiculos_adicionales
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


@pytest.fixture(scope="module")
def params():
    return pa.cargar_parametros()


@pytest.fixture(scope="module")
def prueba(panel, cfg_m1, cfg_datos):
    return vt.ventanas_particion(panel, cfg_m1, cfg_datos, "test")


@pytest.fixture(scope="module")
def sim(panel, cfg_m1, cfg_datos, prueba, params):
    ing = IngenuoEstacional(0.9).ajustar(panel, cfg_m1, cfg_datos)
    glm = GLMPoisson(0.9).ajustar(panel, cfg_m1, cfg_datos)
    mu_i, q_i = ing.predecir(panel, prueba)
    mu_g, q_g = glm.predecir(panel, prueba)
    planes = {"ingenuo_puntual": mu_i, "glm_puntual": mu_g, "glm_p90": q_g, "oraculo": fv.demanda_oraculo(panel, prueba)}
    return fv.simular(panel, prueba, planes, params)


# --- oráculo -------------------------------------------------------------------------
def test_demanda_oraculo_es_la_real(panel, prueba):
    d = fv.demanda_oraculo(panel, prueba)
    assert d.shape == prueba.y.shape
    np.testing.assert_allclose(d[prueba.mask > 0], prueba.y[prueba.mask > 0])


def test_el_oraculo_no_deja_pasajeros_sin_atender(sim):
    o = sim[sim.plan == "oraculo"]
    assert (o["pax_no_atendidos"] == 0).all() and (o["costo_no_atendidos"] == 0).all()
    assert (o["pax_atendidos"] <= o["demanda_real"] + 1e-6).all()


def test_cada_plan_se_simula_en_los_10_origenes(sim):
    assert sim.groupby("plan")["origen"].nunique().eq(10).all() and len(sim) == 40


# --- monotonía: planear con más demanda nunca deja más pasajeros sin atender ----------------
def test_p90_deja_menos_pasajeros_sin_atender_y_cuesta_mas_en_recursos(sim):
    g = sim[sim.plan == "glm_puntual"].set_index("origen")
    q = sim[sim.plan == "glm_p90"].set_index("origen")
    assert (q["pax_no_atendidos"] <= g["pax_no_atendidos"] + 1e-6).all()
    assert (q["flota_total"] >= g["flota_total"]).all()
    assert (q["vehiculos_dia"] >= g["vehiculos_dia"]).all()
    assert (q["costo_vehiculos"] + q["costo_despachos"] >= g["costo_vehiculos"] + g["costo_despachos"] - 1e-6).all()


def test_ocupacion_realizada_es_una_fraccion(sim):
    assert ((sim["ocupacion_realizada"] > 0) & (sim["ocupacion_realizada"] <= 1 + 1e-9)).all()


def test_descomposicion_del_costo(sim):
    suma = sim[list(fv.COSTOS)].sum(axis=1)
    np.testing.assert_allclose(suma, sim["costo_total"])


def test_no_atendidos_alta_mas_base_es_el_total(sim):
    np.testing.assert_allclose(sim["pax_no_atendidos_alta"] + sim["pax_no_atendidos_base"], sim["pax_no_atendidos"])


# --- resumen y barrido ----------------------------------------------------------------------------
def test_resumen_promedia_y_resta_el_oraculo(sim):
    r = fv.resumen(sim).set_index("plan")
    assert r.loc["oraculo", "costo_vs_oraculo"] == 0 and (r["n_origenes"] == 10).all()
    for plan in r.index:
        assert r.loc[plan, "costo_total"] == pytest.approx(sim[sim.plan == plan]["costo_total"].mean())
        assert r.loc[plan, "costo_vs_oraculo"] == pytest.approx(r.loc[plan, "costo_total"] - r.loc["oraculo", "costo_total"])


def test_barrido_reconstruye_el_costo_base_y_es_monotono(sim, params):
    b = fv.barrido_costo_no_atendido(sim, params, factores=(0.5, 1.0, 2.0))
    base = sim.groupby("plan")["costo_total"].mean()
    uno = b[b.factor_c_no_atendido == 1.0].set_index("plan")["costo_total"]
    pd.testing.assert_series_equal(uno.sort_index(), base.sort_index(), check_names=False)
    for plan, g in b.groupby("plan"):
        assert g.sort_values("factor_c_no_atendido")["costo_total"].is_monotonic_increasing
    # con un costo de no atender muy alto, el plan con menos pasajeros sin atender gana: el oráculo
    alto = fv.barrido_costo_no_atendido(sim, params, factores=(1000.0,))
    assert alto.loc[alto.mejor_plan_del_factor, "plan"].iloc[0] == "oraculo"


# --- validaciones de entrada -------------------------------------------------------------------------
def test_plan_con_forma_incorrecta_o_negativo(panel, prueba, params):
    ok = fv.demanda_oraculo(panel, prueba)
    with pytest.raises(ValueError, match="forma"):
        fv.simular(panel, prueba, {"x": ok[:, :10]}, params)
    mal = ok.copy()
    mal[0, 0] = -1
    with pytest.raises(ValueError, match="negativa"):
        fv.simular(panel, prueba, {"x": mal}, params)


def test_origen_sin_todas_las_rutas(panel, prueba, params):
    parcial = prueba.subconjunto(slice(0, 95))                      # quita rutas del último origen
    with pytest.raises(ValueError, match="rutas"):
        fv.simular(panel, parcial, {"oraculo": fv.demanda_oraculo(panel, parcial)}, params)


# --- temporada alta -----------------------------------------------------------------------------------
def test_vehiculos_adicionales_a_mano():
    f = pd.date_range("2023-12-01", periods=4)
    pc = pd.DataFrame({"fecha": list(f) * 2, "corredor": [1] * 4 + [2] * 4,
                       "vehiculos": [10, 12, 20, 18, 5, 5, 4, 4]})
    alta = pd.Series([False, False, True, True], index=f)
    r = vehiculos_adicionales(pc, alta).set_index("corredor")
    assert (r.loc[1, "pico_alta"], r.loc[1, "pico_base"], r.loc[1, "adicionales"]) == (20, 12, 8)
    assert r.loc[2, "adicionales"] == 0                              # pico de alta (4) menor que el de base (5): no negativo


def test_vehiculos_adicionales_sin_dias_de_un_tipo():
    f = pd.date_range("2023-12-01", periods=3)
    pc = pd.DataFrame({"fecha": f, "corredor": 1, "vehiculos": [3, 4, 5]})
    r = vehiculos_adicionales(pc, pd.Series([False, False, False], index=f))
    assert np.isnan(r.loc[0, "adicionales"])


def test_adicionales_no_negativos_en_la_simulacion(sim):
    assert (sim["vehiculos_adicionales_alta"] >= 0).all()
    assert (sim["conductores_adicionales_alta"] >= 0).all()


def test_conductores_adicionales_y_dias_a_mano():
    f = pd.date_range("2023-12-01", periods=4)
    pc = pd.DataFrame({"fecha": f, "corredor": 1, "vehiculos": [10, 12, 20, 18],
                       "conductores_dia": [15, 18, 33, 30]})
    r = vehiculos_adicionales(pc, pd.Series([False, False, True, True], index=f)).iloc[0]
    assert (r.cd_pico_alta, r.cd_pico_base, r.conductores_adicionales) == (33, 18, 15)
    assert (r.dias_alta, r.dias_base) == (2, 2) and r.adicionales == 8


def test_indicios_de_censura(params):
    t = indicios_censura(0.9, params)
    assert len(t) == 10 and t["ruta_id"].is_unique
    assert ((t["frac_alta"] >= 0) & (t["frac_alta"] <= 1)).all() and ((t["frac_base"] >= 0) & (t["frac_base"] <= 1)).all()
    assert (t["dias_alta"] > 0).all() and (t["dias_base"] > 0).all()
    mas_estricto = indicios_censura(1.2, params)
    assert (mas_estricto["frac_alta"] <= t["frac_alta"] + 1e-12).all()   # umbral más alto -> menos días "cerca"
