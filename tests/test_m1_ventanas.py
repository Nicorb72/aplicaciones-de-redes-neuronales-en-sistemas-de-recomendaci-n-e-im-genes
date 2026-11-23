"""Pruebas de rolling_origin.py y ventanas.py (Persona 2): ventanas sin fuga de información.

Ejecutar:  python -m pytest tests/test_m1_ventanas.py -q
No requieren internet: usan los datos versionados en data/m1_demanda.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.data.utils import RAIZ, cargar_config
from m1_demanda.rolling_origin import origenes_particion

PROC = RAIZ / "data" / "m1_demanda" / "processed"


@pytest.fixture(scope="module")
def cfg_datos():
    return cargar_config()


@pytest.fixture(scope="module")
def cfg_m1():
    return cargar_yaml("configs/m1_demanda.yaml")


@pytest.fixture(scope="module")
def panel(cfg_m1, cfg_datos):
    return vt.cargar_panel(cfg_m1, cfg_datos)


# --- rolling_origin ---------------------------------------------------------
def test_origenes_prueba_coinciden_con_particiones_json(cfg_datos):
    esperado = json.loads((PROC / "particiones.json").read_text(encoding="utf-8"))["origenes_rodantes_test"]
    assert origenes_particion(cfg_datos, "test", paso_dias=7) == esperado
    assert len(esperado) == 10


def test_origenes_dentro_de_la_particion(cfg_datos):
    p = cfg_datos["particiones"]
    for nombre in ("train", "val", "test", "post_hueco"):
        ini, fin = pd.Timestamp(p[nombre][0]), pd.Timestamp(p[nombre][1])
        for o in origenes_particion(cfg_datos, nombre, paso_dias=7):
            o = pd.Timestamp(o)
            assert ini <= o and o + pd.Timedelta(days=29) <= fin
    assert origenes_particion(cfg_datos, "test", 7)[-1] == "2023-12-02"


def test_particion_invalida(cfg_datos):
    with pytest.raises(ValueError):
        origenes_particion(cfg_datos, "otra")


# --- columnas prohibidas ----------------------------------------------------
@pytest.mark.parametrize("col", ["evento_no_previsible", "despachos", "despachos_efectivos",
                                 "pasajeros_transito", "imputado", "particion", "segmento", "nombre_festivo"])
def test_columna_prohibida_da_error(cfg_m1, col):
    with pytest.raises(ValueError, match="prohibidas"):
        vt.validar_columnas_entrada(["festivo", col], cfg_m1)


def test_calendario_configurado_no_tiene_prohibidas(cfg_m1):
    vt.validar_columnas_entrada(cfg_m1["calendar_features"], cfg_m1)


def test_cargar_panel_rechaza_columna_prohibida(cfg_m1, cfg_datos):
    mala = copy.deepcopy(cfg_m1)
    mala["calendar_features"] = mala["calendar_features"] + ["evento_no_previsible"]
    with pytest.raises(ValueError, match="prohibidas"):
        vt.cargar_panel(mala, cfg_datos)


# --- forma y cobertura ------------------------------------------------------
def test_hay_ventanas_para_los_10_origenes_de_prueba(panel, cfg_m1, cfg_datos):
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "test")
    assert len(v) == 100
    assert len(np.unique(v.origen)) == 10 and len(np.unique(v.ruta_idx)) == 10
    assert v.x_hist.shape == (100, 28) and v.cal_fut.shape[:2] == (100, 30)
    assert v.cal_hist.shape == (100, 28, len(cfg_m1["calendar_features"]))


def test_objetivos_dentro_de_la_particion(panel, cfg_m1, cfg_datos):
    p = cfg_datos["particiones"]
    for nombre in ("train", "val", "test"):
        v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, nombre)
        ini, fin = np.datetime64(p[nombre][0]), np.datetime64(p[nombre][1])
        assert v.origen.min() >= ini
        assert v.origen.max() + np.timedelta64(29, "D") <= fin


# --- fuga de información ----------------------------------------------------
def test_ventana_no_cambia_si_se_altera_el_futuro(panel, cfg_m1):
    """Si se corrompe todo dato desde el origen en adelante, la entrada NO debe cambiar."""
    origen = pd.Timestamp("2023-11-05")
    base = vt.construir_ventanas(panel, cfg_m1, [origen])
    i = panel.idx(origen)
    alterado = copy.copy(panel)
    alterado.y_imp = panel.y_imp.copy()
    alterado.y_obs = panel.y_obs.copy()
    alterado.y_imp[i:] = 1e7
    alterado.y_obs[i:] = 1e7
    otro = vt.construir_ventanas(alterado, cfg_m1, [origen])
    assert len(base) == len(otro) == 10
    np.testing.assert_array_equal(base.x_hist, otro.x_hist)
    np.testing.assert_array_equal(base.nivel, otro.nivel)
    np.testing.assert_array_equal(base.cal_hist, otro.cal_hist)
    assert not np.array_equal(base.y, otro.y)  # el objetivo sí cambia: confirma que la alteración fue efectiva


def test_x_hist_coincide_con_calculo_independiente(panel, cfg_m1):
    """Recalcula la entrada leyendo el CSV directamente, solo con fechas < origen."""
    origen = pd.Timestamp("2023-10-15")
    v = vt.construir_ventanas(panel, cfg_m1, [origen])
    s = pd.read_csv(PROC / "series_diarias.csv.gz", parse_dates=["fecha"], dtype={"ruta_id": str})
    for k in range(len(v)):
        ruta = panel.rutas["ruta_id"].iloc[v.ruta_idx[k]]
        h = s[(s.ruta_id == ruta) & (s.fecha < origen) & (s.fecha >= origen - pd.Timedelta(days=28))]
        h = h.sort_values("fecha").pasajeros_imp.to_numpy()
        nivel = h.mean()
        assert len(h) == 28
        assert v.nivel[k] == pytest.approx(nivel, rel=1e-5)
        np.testing.assert_allclose(v.x_hist[k], np.log((h + 1) / (nivel + 1)), rtol=1e-4, atol=1e-5)


# --- huecos, segmentos y máscaras ------------------------------------------
def test_no_hay_ventanas_con_historial_en_hueco_largo(panel, cfg_m1, cfg_datos):
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "train")
    ruta = list(panel.rutas["ruta_id"]).index("50001-11001")
    o = pd.to_datetime(v.origen[v.ruta_idx == ruta])
    hueco_ini, hueco_fin = pd.Timestamp("2022-03-14"), pd.Timestamp("2022-03-29")
    # el historial [o-28, o-1] no puede tocar el hueco: no hay origen entre hueco_ini+1 y hueco_fin+28
    prohibidos = (o > hueco_ini) & (o <= hueco_fin + pd.Timedelta(days=28))
    assert not prohibidos.any()
    # en cambio sí hay ventanas cuyos OBJETIVOS caen en el hueco (se enmascaran)
    i = np.where((v.ruta_idx == ruta) & (pd.to_datetime(v.origen) == pd.Timestamp("2022-03-10")))[0]
    assert len(i) == 1 and v.mask[i[0], 4:20].sum() == 0


def test_ninguna_ventana_tiene_nan_ni_cruza_segmentos(panel, cfg_m1, cfg_datos):
    for nombre in ("train", "val", "test", "post_hueco"):
        v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, nombre)
        assert np.isfinite(v.x_hist).all() and np.isfinite(v.nivel).all() and (v.nivel > 0).all()
        for k in range(0, len(v), 17):
            i = panel.idx(v.origen[k])
            seg = panel.segmento[i - 28:i + 30, v.ruta_idx[k]]
            assert (seg == seg[0]).all() and seg[0] > 0


def test_paro_nacional_no_entra_como_entrada_ni_pesa_en_la_perdida(panel, cfg_m1, cfg_datos):
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "train")
    paro_ini, paro_fin = pd.Timestamp("2021-04-28"), pd.Timestamp("2021-06-15")
    o = pd.to_datetime(v.origen)
    # historial tocando el paro => ventana descartada
    assert not ((o > paro_ini) & (o <= paro_fin + pd.Timedelta(days=28))).any()
    # ventanas anteriores cuyos objetivos caen dentro del paro: peso 0 en esos días
    k = np.where(o == pd.Timestamp("2021-04-20"))[0]
    assert len(k) > 0
    dia_en_paro = (paro_ini - pd.Timestamp("2021-04-20")).days  # índice del primer día del paro
    assert (v.mask[k, dia_en_paro:] == 0).all()


def test_mascara_excluye_imputados_y_faltantes(panel, cfg_m1, cfg_datos):
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "train")
    for k in range(len(v)):
        i = panel.idx(v.origen[k])
        r = v.ruta_idx[k]
        malo = panel.imputado[i:i + 30, r] | ~np.isfinite(panel.y_obs[i:i + 30, r])
        assert (v.mask[k][malo] == 0).all()


def test_modo_produccion_sin_objetivo(panel, cfg_m1):
    prod = vt.origen_produccion(panel)
    assert prod == pd.Timestamp("2026-03-31")
    v = vt.construir_ventanas(panel, cfg_m1, [prod], con_objetivo=False)
    assert len(v) == 10 and v.mask.sum() == 0 and v.y.sum() == 0
    assert v.cal_fut.shape == (10, 30, len(cfg_m1["calendar_features"]))
