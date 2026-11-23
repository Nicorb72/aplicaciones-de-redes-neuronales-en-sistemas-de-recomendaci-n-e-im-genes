"""Pruebas del contrato de m1_demanda/predict.py (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_contrato.py -q
No requieren internet. Estos casos se proponen a la Persona 5 para su tests/test_m1_predict.py.
"""
import copy
import json
import time

import numpy as np
import pandas as pd
import pytest

from m1_demanda import predict
from m1_demanda.data.utils import RAIZ

RUTA = "50001-11001"
COLUMNAS = ["fecha", "ruta_id", "horizonte", "pronostico", "p90", "base_plan",
            "despachos", "vehiculos", "conductores", "real"]


@pytest.fixture(scope="module")
def df():
    return predict.forecast(RUTA, 0.9)


# --- list_routes -----------------------------------------------------------------
def test_list_routes():
    r = predict.list_routes()
    assert list(r.columns) == ["ruta_id", "nombre", "corredor", "sentido", "ruta_regreso"]
    assert len(r) == 10 and r["ruta_id"].is_unique
    assert set(r["ruta_regreso"]) == set(r["ruta_id"])
    assert RUTA in set(r["ruta_id"])


# --- forma y tipos -----------------------------------------------------------------
def test_columnas_y_30_filas(df):
    assert list(df.columns) == COLUMNAS
    assert len(df) == 30 and df["horizonte"].tolist() == list(range(1, 31))
    assert (df["ruta_id"] == RUTA).all()
    assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
    assert df["fecha"].is_monotonic_increasing and df["fecha"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    for c in ("despachos", "vehiculos", "conductores"):
        assert pd.api.types.is_integer_dtype(df[c]) and (df[c] >= 0).all()
    assert df[["pronostico", "p90", "base_plan"]].notna().all().all()


def test_p90_no_menor_que_la_media_y_base_de_plan(df):
    assert (df["p90"] >= df["pronostico"]).all()
    np.testing.assert_allclose(df["base_plan"], df["p90"])            # quantile = 0,9
    media = predict.forecast(RUTA, None)
    np.testing.assert_allclose(media["base_plan"], media["pronostico"])  # quantile = None
    assert (media["despachos"] <= df["despachos"]).all()                 # planear con P90 nunca da menos


def test_origen_por_defecto_es_el_dia_siguiente_al_ultimo_dato(df):
    assert df.attrs["origen"] == "2026-03-31" and df.attrs["ultimo_dato"] == "2026-03-30"
    assert df["fecha"].iloc[0] == pd.Timestamp("2026-03-31") and df["real"].isna().all()


def test_attrs(df):
    for k in ("modelo", "origen", "ultimo_dato", "flota_corredor", "plantilla_conductores", "advertencias"):
        assert k in df.attrs
    # El modelo declarado es el que marca models/m1_demanda/modelo_final.json (la red de menor MASE de validación).
    marca = json.loads((RAIZ / "models/m1_demanda/modelo_final.json").read_text(encoding="utf-8"))["familia"]
    assert marca in ("mlp", "rnn")
    assert df.attrs["modelo"] == marca
    assert df.attrs["flota_corredor"] >= df["vehiculos"].max()
    assert df.attrs["plantilla_conductores"] >= df["conductores"].max()
    # Un pronóstico normal de producción con la red NO trae advertencias espurias.
    assert isinstance(df.attrs["advertencias"], list) and df.attrs["advertencias"] == []


def test_los_dos_sentidos_comparten_vehiculos_y_conductores():
    ida, reg = predict.forecast("50001-11001", 0.9), predict.forecast("11001-50001", 0.9)
    pd.testing.assert_series_equal(ida["vehiculos"], reg["vehiculos"])
    pd.testing.assert_series_equal(ida["conductores"], reg["conductores"])
    assert ida.attrs["flota_corredor"] == reg.attrs["flota_corredor"]


# --- origen de prueba (reproduce un pronóstico con valores reales) ----------------------
def test_origen_de_prueba_rellena_real_y_advierte_que_no_es_fuera_de_muestra():
    d = predict.forecast(RUTA, 0.9, origen="2023-12-02")          # prueba: fuera de muestra
    assert d["fecha"].iloc[0] == pd.Timestamp("2023-12-02") and d["real"].notna().sum() >= 28
    assert not any("fuera de muestra" in w for w in d.attrs["advertencias"])
    dentro = predict.forecast(RUTA, 0.9, origen="2023-03-01")     # entrenamiento: dentro de muestra
    assert any("fuera de muestra" in w for w in dentro.attrs["advertencias"])


def test_la_red_no_usa_datos_desde_el_origen(monkeypatch):
    """Corromper todo dato desde el origen no cambia el pronóstico de la red (misma garantía que el ingenuo)."""
    base = predict.forecast(RUTA, None, origen="2023-12-02")
    cfg_m1, cfg_datos, panel, params, mezcla = predict._recursos()
    i = panel.idx("2023-12-02")
    alterado = copy.copy(panel)
    alterado.y_imp, alterado.y_obs = panel.y_imp.copy(), panel.y_obs.copy()
    alterado.y_imp[i:] = 1e7
    alterado.y_obs[i:] = 1e7
    monkeypatch.setattr(predict, "_recursos", lambda: (cfg_m1, cfg_datos, alterado, params, mezcla))
    otro = predict.forecast(RUTA, None, origen="2023-12-02")
    np.testing.assert_allclose(base["pronostico"], otro["pronostico"])
    np.testing.assert_allclose(base["p90"], otro["p90"])


def test_respaldo_al_ingenuo_sin_red(monkeypatch):
    """Sin red entrenada se usa el ingenuo (última semana) y se advierte."""
    monkeypatch.setattr(predict, "_red", lambda: (None, "No hay una red entrenada (prueba)."))
    d = predict.forecast(RUTA, 0.9, origen="2023-12-02")
    assert d.attrs["modelo"] == "ingenuo_estacional"
    assert any("No hay una red" in w for w in d.attrs["advertencias"])
    s = pd.read_csv(RAIZ / "data/m1_demanda/processed/series_diarias.csv.gz", parse_dates=["fecha"],
                    dtype={"ruta_id": str})
    ult = s[(s.ruta_id == RUTA) & (s.fecha < "2023-12-02") & (s.fecha >= "2023-11-25")].sort_values("fecha")
    np.testing.assert_allclose(d["pronostico"].iloc[:7], ult["pasajeros_imp"].to_numpy())   # repite la última semana


def test_respaldo_por_ruta_cuando_el_historial_de_la_red_tiene_huecos():
    """Duitama→Sogamoso tiene un hueco largo en enero de 2026: con la ventana de la red el historial es inválido
    y esa ruta usa el ingenuo (la última semana sí es válida); la advertencia lo dice."""
    ruta = "15238-15759"
    d = predict.forecast(ruta, 0.9, origen="2026-02-10")           # historial de 56 días toca 8–23 de enero
    assert "ingenuo_estacional" in d.attrs["modelo"]
    assert any(ruta in w and "ingenuo" in w for w in d.attrs["advertencias"])
    assert d["pronostico"].notna().all() and (d["p90"] >= d["pronostico"]).all()


# --- errores de entrada -------------------------------------------------------------
@pytest.mark.parametrize("q", [0.5, 0.95, 1, "0.9", True])
def test_quantile_invalido(q):
    with pytest.raises(ValueError, match="quantile"):
        predict.forecast(RUTA, q)


def test_ruta_desconocida():
    for r in ("99999-99999", 50001, None):
        with pytest.raises(ValueError, match="route_id"):
            predict.forecast(r)


def test_op_params_desconocido_y_tipo_invalido():
    with pytest.raises(ValueError, match="desconocido"):
        predict.forecast(RUTA, 0.9, {"no_existe": 1})
    with pytest.raises(ValueError, match="op_params"):
        predict.forecast(RUTA, 0.9, [("absenteeism", 0.1)])


def test_op_params_cambia_la_operacion_en_la_direccion_esperada(df):
    mas_ocupacion = predict.forecast(RUTA, 0.9, {"target_occupancy": 0.9})
    assert (mas_ocupacion["despachos"] <= df["despachos"]).all()
    mas_ausentismo = predict.forecast(RUTA, 0.9, {"absenteeism": 0.2})
    assert mas_ausentismo.attrs["plantilla_conductores"] >= df.attrs["plantilla_conductores"]
    # el pronóstico no depende de los parámetros de operación
    pd.testing.assert_series_equal(mas_ocupacion["pronostico"], df["pronostico"])


def test_op_params_invalido_se_rechaza():
    with pytest.raises(ValueError):
        predict.forecast(RUTA, 0.9, {"target_occupancy": 5})


def test_origen_invalido():
    with pytest.raises(ValueError, match="origen"):
        predict.forecast(RUTA, 0.9, origen="no-es-fecha")
    with pytest.raises(ValueError, match="posterior"):
        predict.forecast(RUTA, 0.9, origen="2026-06-01")
    with pytest.raises(ValueError):
        predict.forecast(RUTA, 0.9, origen="2024-06-01")      # sin datos en 2024 (hueco de reporte)


def test_no_modifica_estado_global(df):
    predict.forecast(RUTA, 0.9, {"absenteeism": 0.3})
    otra = predict.forecast(RUTA, 0.9)
    pd.testing.assert_frame_equal(df, otra)
    assert otra.attrs["parametros_operacion"]["absenteeism"] == 0.08


# --- sin internet y rapidez ------------------------------------------------------------
def test_funciona_sin_internet(monkeypatch):
    import socket

    def sin_red(*a, **k):
        raise AssertionError("predict.py intentó usar la red")

    monkeypatch.setattr(socket, "socket", sin_red)
    monkeypatch.setattr(socket, "create_connection", sin_red)
    assert len(predict.forecast(RUTA, 0.9)) == 30


def test_llamada_rapida():
    predict.forecast(RUTA, 0.9)           # calienta la caché
    t = time.perf_counter()
    predict.forecast("66001-63001", 0.9)
    assert time.perf_counter() - t < 2.0


# --- fixture para la Persona 5 ---------------------------------------------------------------
def test_fixture_coincide_con_el_contrato(df):
    fx = pd.read_csv(RAIZ / "tests/fixtures/m1_forecast_ejemplo.csv", parse_dates=["fecha"], dtype={"ruta_id": str})
    assert list(fx.columns) == COLUMNAS and len(fx) == 30
    pd.testing.assert_frame_equal(fx.drop(columns="real"), df.drop(columns="real"), check_dtype=False, rtol=1e-9)
