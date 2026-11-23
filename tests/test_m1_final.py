"""Pruebas de final.py (Persona 2): comparación justa y elección de la red de la app solo con validación.

Ejecutar:  python -m pytest tests/test_m1_final.py -q
"""
import json

import numpy as np
import pandas as pd
import pytest

from m1_demanda import final as fi


def _tabla(valores: dict) -> pd.DataFrame:
    """valores: {(familia, rank): (mase_val, mase_test)} con 3 semillas cada una."""
    filas = []
    for (fam, rank), (mv, mt) in valores.items():
        for s in range(3):
            for part, m in (("val", mv), ("test", mt)):
                filas.append({"familia": fam, "config_rank": rank, "semilla": s, "particion": part,
                              "subconjunto": "total", "n_parametros": 1000 * rank, "epocas": 10, "val_perdida": 1.0 + s,
                              "segundos": 5.0, "mase": m + 0.01 * s, "mae": 100.0, "rmse": 150.0, "cobertura": 0.9,
                              "pinball_escalada": 0.2, "pinball": 50.0})
    return pd.DataFrame(filas)


def test_la_eleccion_usa_solo_validacion_aunque_prueba_diga_otra_cosa():
    # mlp gana en validación y pierde en prueba: debe elegirse mlp
    t = _tabla({("mlp", 1): (0.85, 1.20), ("mlp", 2): (0.90, 0.80), ("rnn", 1): (0.88, 0.70), ("rnn", 2): (0.95, 0.70)})
    e = fi.elegir_familia_final(t)
    assert e["familia"] == "mlp" and e["config_rank"] == 1
    assert e["mase_val_media"] == pytest.approx(0.86)                     # media de 0,85, 0,86, 0,87
    assert set(e["por_familia"]) == {"mlp", "rnn"} and e["por_familia"]["rnn"]["config_rank"] == 1


def test_la_eleccion_respeta_la_configuracion_final_indicada():
    t = _tabla({("mlp", 1): (0.85, 1.0), ("mlp", 2): (0.90, 1.0), ("rnn", 1): (0.88, 1.0)})
    e = fi.elegir_familia_final(t, infos={"mlp": {"config_final_rank": 2}, "rnn": {"config_final_rank": 1}})
    assert e["por_familia"]["mlp"]["config_rank"] == 2 and e["familia"] == "rnn"


def test_sin_validacion_da_error():
    t = _tabla({("mlp", 1): (0.85, 1.0)})
    with pytest.raises(ValueError, match="validación"):
        fi.elegir_familia_final(t[t.particion == "test"])


def test_escribir_modelo_final(tmp_path, monkeypatch):
    monkeypatch.setattr(fi, "MODELOS", tmp_path)
    ruta = fi.escribir_modelo_final({"familia": "rnn", "config_rank": 1, "mase_val_media": 0.8})
    assert json.loads(ruta.read_text(encoding="utf-8"))["familia"] == "rnn"


def test_tabla_de_comparacion_30_filas_por_familia_y_resumen_correcto():
    filas = []
    for s in range(30):
        for part in ("val", "test", "post_hueco"):
            filas.append({"familia": "mlp", "config_rank": 1, "semilla": s, "particion": part, "subconjunto": "total",
                          "n_parametros": 152292, "epocas": 50, "val_perdida": 1.0, "segundos": 8.0,
                          "mase": 0.9 + 0.001 * s, "mae": 250.0, "rmse": 400.0, "cobertura": 0.9,
                          "pinball_escalada": 0.2, "pinball": 60.0})
    semillas = pd.DataFrame(filas)
    base = pd.DataFrame([{"modelo": "ingenuo", "particion": p, "n_parametros": 0, "mase": 1.3, "mae": 380.0,
                          "rmse": 620.0, "cobertura": 0.8, "pinball_escalada": 0.4, "pinball": 126.0}
                         for p in ("val", "test", "post_hueco")])
    c = fi.tabla_comparacion(semillas, base, {"mlp": 30, "ingenuo": 0})
    red = c[(c.modelo == "mlp_cfg1") & (c.particion == "test")].iloc[0]
    assert red.n_semillas == 30 and red.n_parametros == 152292 and red.configuraciones_evaluadas == 30
    assert red.mase_media == pytest.approx(0.9 + 0.001 * 14.5)
    assert red.mase_desv == pytest.approx(np.std(0.9 + 0.001 * np.arange(30), ddof=1))
    assert (red.mase_mejor, red.mase_peor) == (pytest.approx(0.9), pytest.approx(0.929))
    b = c[(c.modelo == "ingenuo") & (c.particion == "test")].iloc[0]
    assert b.n_semillas == 1 and b.configuraciones_evaluadas == 0 and np.isnan(b.mase_desv)
    assert len(c) == 6


def test_correr_configuracion_guarda_extras_por_horizonte_y_ruta():
    """Con 2 semillas y pocas épocas: la tabla de semillas y los extras tienen la estructura esperada."""
    from common.config_loader import cargar_yaml
    from m1_demanda import ventanas as vt
    from m1_demanda.data.utils import cargar_config
    from m1_demanda.train import hp_desde_config

    cfg_m1, cfg_datos = cargar_yaml("configs/m1_demanda.yaml"), cargar_config()
    panel = vt.cargar_panel(cfg_m1, cfg_datos)
    hp = hp_desde_config(cfg_m1, epochs=2, patience=3, units=16)
    filas, extras, modelos = fi.correr_configuracion("mlp", 1, hp, cfg_m1, cfg_datos, panel, [5, 6])
    assert set(modelos) == {5, 6}
    assert len(filas) == 2 * 3 * 3                                          # semillas × particiones × subconjuntos
    assert len(extras["horizonte"]) == 2 * 3 * 30 and len(extras["ruta"]) == 2 * 3 * 10
    assert {"alta", "base", "total"} == {f["subconjunto"] for f in filas}
