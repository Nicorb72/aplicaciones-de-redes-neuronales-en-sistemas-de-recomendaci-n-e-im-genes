"""Pruebas de sensitivity.py (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_sensibilidad.py -q
"""
import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.data.utils import cargar_config
from m1_demanda.operacion import forecast_value as fv
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.sensitivity import DEMANDA, PARAMETROS, orden_por_efecto, sensibilidad


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
def v(panel, cfg_m1, cfg_datos):
    return vt.ventanas_particion(panel, cfg_m1, cfg_datos, "test")


@pytest.fixture(scope="module")
def plan(panel, v):
    return fv.demanda_oraculo(panel, v) * 1.1          # plan con holgura, determinista y sin entrenar nada


@pytest.fixture(scope="module")
def tabla(panel, v, plan, params):
    return sensibilidad(panel, v, plan, params)


def test_estructura_de_la_tabla(tabla):
    assert tabla.iloc[0]["parametro"] == "base"
    assert len(tabla) == 1 + 2 * (len(PARAMETROS) + 1)
    assert set(tabla["variacion"]) == {-0.1, 0.0, 0.1}
    assert set(tabla["parametro"]) == {"base", DEMANDA, *PARAMETROS}


def test_delta_cero_no_cambia_nada(panel, v, plan, params):
    t = sensibilidad(panel, v, plan, params, delta=0.0, parametros=("capacity_by_class", "absenteeism", DEMANDA))
    for m in ("flota_total", "plantilla_total", "costo_total"):
        np.testing.assert_allclose(t[f"d_{m}_pct"], 0.0, atol=1e-9)


def test_el_caso_base_coincide_con_la_simulacion_directa(tabla, panel, v, plan, params):
    s = fv.simular(panel, v, {"p": plan}, params)
    assert tabla.iloc[0]["costo_total"] == pytest.approx(s["costo_total"].mean())
    assert tabla.iloc[0]["flota_total"] == pytest.approx(s["flota_total"].mean())


def _d(tabla, parametro, signo, metrica):
    f = tabla[(tabla.parametro == parametro) & (np.sign(tabla.variacion) == signo)]
    return float(f[f"d_{metrica}_pct"].iloc[0])


@pytest.mark.parametrize("parametro", ["capacity_by_class", "target_occupancy"])
def test_mas_capacidad_u_ocupacion_no_da_mas_recursos(tabla, parametro):
    for m in ("flota_total", "plantilla_total"):
        assert _d(tabla, parametro, +1, m) <= 0 <= _d(tabla, parametro, -1, m)


def test_mas_demanda_no_da_menos_recursos(tabla):
    for m in ("flota_total", "plantilla_total", "costo_total"):
        assert _d(tabla, DEMANDA, +1, m) >= 0 >= _d(tabla, DEMANDA, -1, m)


def test_mas_horas_de_operacion_no_da_mas_flota(tabla):
    assert _d(tabla, "operating_hours", +1, "flota_total") <= 0 <= _d(tabla, "operating_hours", -1, "flota_total")


def test_mas_ausentismo_y_menos_jornada_no_dan_menos_conductores(tabla):
    assert _d(tabla, "absenteeism", +1, "plantilla_total") >= 0
    assert _d(tabla, "workday.jornada_semanal_h", -1, "plantilla_total") >= 0
    assert _d(tabla, "workday.max_conduccion_dia_h", -1, "plantilla_total") >= 0


def test_los_costos_unitarios_no_cambian_flota_ni_plantilla_y_mueven_el_costo(tabla):
    for p in ("costs.c_veh_dia", "costs.c_despacho", "costs.smmlv"):
        assert _d(tabla, p, +1, "flota_total") == 0 and _d(tabla, p, +1, "plantilla_total") == 0
        assert _d(tabla, p, +1, "costo_total") > 0 > _d(tabla, p, -1, "costo_total")


def test_el_costo_lineal_responde_con_proporcion_conocida(tabla, panel, v, plan, params):
    """El costo de despachos es proporcional a c_despacho: +10 % en el parámetro = +10 % en ese componente."""
    s = fv.simular(panel, v, {"p": plan}, params)
    parte = s["costo_despachos"].mean() / s["costo_total"].mean() * 10.0
    assert _d(tabla, "costs.c_despacho", +1, "costo_total") == pytest.approx(parte, rel=1e-6)


def test_orden_por_efecto(tabla):
    o = orden_por_efecto(tabla)
    assert set(o) == {DEMANDA, *PARAMETROS} and o[-1] in {"costs.c_no_atendido", "technical_reserve", "absenteeism"}
    efectos = [tabla[(tabla.parametro == p)]["d_costo_total_pct"].abs().max() for p in o]
    assert efectos == sorted(efectos, reverse=True)


def test_delta_invalido(panel, v, plan, params):
    with pytest.raises(ValueError, match="delta"):
        sensibilidad(panel, v, plan, params, delta=1.5)


def test_parametro_desconocido(panel, v, plan, params):
    with pytest.raises(ValueError, match="desconocido"):
        sensibilidad(panel, v, plan, params, parametros=("no_existe",))
