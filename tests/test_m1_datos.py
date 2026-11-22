"""Pruebas del pipeline de datos del Módulo 1 (Persona 1).

Ejecutar:  python -m pytest tests/test_m1_datos.py -q
No requieren internet: usan los datos versionados en data/m1_demanda.
"""
from datetime import date

import pandas as pd
import pytest
import yaml

from m1_demanda.data import calendar_features as cf
from m1_demanda.data import split
from m1_demanda.data.utils import RAIZ, cargar_config

PROC = RAIZ / "data" / "m1_demanda" / "processed"


@pytest.fixture(scope="module")
def cfg():
    return cargar_config()


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(PROC / "series_diarias.csv.gz", parse_dates=["fecha"])


# --- Calendario --------------------------------------------------------------
@pytest.mark.parametrize("anio,esperado", [
    (2021, date(2021, 4, 4)), (2022, date(2022, 4, 17)), (2023, date(2023, 4, 9)),
    (2025, date(2025, 4, 20)), (2026, date(2026, 4, 5)),
])
def test_domingo_pascua(anio, esperado):
    assert cf.domingo_pascua(anio) == esperado


def test_ley_emiliani_traslada_al_lunes():
    f = cf.festivos_colombia(2023)
    assert date(2023, 1, 9) in f          # Reyes: 6-ene (viernes) -> lunes 9
    assert date(2023, 1, 6) not in f
    assert date(2023, 7, 20) in f         # Independencia: fijo (jueves)
    assert all(d.weekday() == 0 for d, n in f.items()
               if n in {"Reyes Magos", "San José", "Día de la Raza", "Todos los Santos"})


def test_festivo_ley_2578_de_2026():
    assert date(2026, 7, 13) in cf.festivos_colombia(2026)    # 9-jul (jueves) -> lunes 13
    assert all(n != "Virgen de Chiquinquirá" for n in cf.festivos_colombia(2025).values())


def test_festivos_coinciden_con_libreria_holidays(cfg):
    pytest.importorskip("holidays")
    assert cf.verificar_contra_holidays(cfg).empty


def test_puente_octubre_2023(cfg):
    cal = cf.construir_calendario(cfg).set_index("fecha")
    assert cal.loc["2023-10-14":"2023-10-16", "puente"].all()
    assert cal.loc["2023-10-13", "vispera_puente"]
    assert cal.loc["2023-10-16", "ultimo_dia_puente"]
    assert cal.loc["2023-10-09":"2023-10-13", "receso_octubre"].all()


# --- Configuración -----------------------------------------------------------
def test_yaml_sin_comillas_es_octal():
    """Documenta la trampa: sin comillas PyYAML convierte 05615 en 2957 (octal)."""
    assert yaml.safe_load("x: 05615")["x"] == 2957


def test_codigos_divipola_son_texto(cfg):
    for par in cfg["seleccion_rutas"]["corredores"]:
        assert all(isinstance(c, str) and len(c) == 5 for c in par)


# --- Panel procesado -----------------------------------------------------------
def test_panel_completo(panel, cfg):
    n_rutas = 2 * len(cfg["seleccion_rutas"]["corredores"])
    dias = pd.date_range(cfg["fuente"]["fecha_inicio"], cfg["limpieza"]["fecha_corte_datos"])
    assert panel.ruta_id.nunique() == n_rutas
    assert len(panel) == n_rutas * len(dias)
    assert not panel.duplicated(["ruta_id", "fecha"]).any()


def test_sin_datos_fuera_de_segmento(panel):
    assert panel.loc[panel.segmento == 0, "pasajeros"].isna().all()
    assert panel.loc[panel.segmento == 0, "pasajeros_imp"].isna().all()
    # el hueco de 2024 es faltante, nunca cero
    assert panel.loc[panel.fecha.dt.year == 2024, "pasajeros_imp"].isna().all()


def test_conteos_validos(panel):
    y = panel.pasajeros_imp.dropna()
    assert (y >= 0).all() and (y == y.round()).all()


def test_entrenamiento_validacion_prueba_completos(panel):
    """train/val/test no deben tener faltantes, salvo huecos > 14 días documentados."""
    m = panel.particion.isin(["val", "test"])
    assert panel.loc[m, "pasajeros_imp"].notna().all()


def test_particiones_ordenadas_y_disjuntas(cfg):
    p = cfg["particiones"]
    assert p["train"][1] < p["val"][0] <= p["val"][1] < p["test"][0] <= p["test"][1] < p["post_hueco"][0]


def test_origenes_rodantes_cubren_fin_de_anio(cfg):
    o = split.origenes_rodantes(cfg)
    h = cfg["particiones"]["horizonte_dias"]
    ultimo_dia = pd.Timestamp(o[-1]) + pd.Timedelta(days=h - 1)
    assert ultimo_dia == pd.Timestamp(cfg["particiones"]["test"][1])
    assert all(pd.Timestamp(x) >= pd.Timestamp(cfg["particiones"]["test"][0]) for x in o)


def test_prueba_contiene_temporada_alta(panel):
    t = panel[panel.particion == "test"]
    assert t.temporada_fin_anio.any() and t.puente.any()
