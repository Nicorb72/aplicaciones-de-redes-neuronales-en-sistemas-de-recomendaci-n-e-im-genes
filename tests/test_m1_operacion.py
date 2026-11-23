"""Pruebas de la operación del Módulo 1: parámetros, flota y conductores (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_operacion.py -q
"""
import copy

import numpy as np
import pandas as pd
import pytest

from common.config_loader import cargar_yaml
from m1_demanda.data.utils import RAIZ
from m1_demanda.operacion import fleet
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.planificar import costo_con_demanda_real, planificar

IDA, REG = "50001-11001", "11001-50001"           # corredor 1
FECHA = pd.Timestamp("2026-08-01")                # jornada vigente: 42 h


@pytest.fixture(scope="module")
def p():
    return pa.cargar_parametros()


@pytest.fixture(scope="module")
def rutas():
    return pd.read_csv(RAIZ / "data/m1_demanda/processed/rutas.csv", dtype={"ruta_id": str})


@pytest.fixture()
def mezcla_bus():
    """Mezcla de juguete: todas las rutas del corredor 1 con 100 % BUS (capacidad 40)."""
    return pd.DataFrame({"ruta_id": [IDA, REG], "clase_vehiculo": ["BUS", "BUS"],
                         "participacion_despachos": [1.0, 1.0]})


def plan(ida, reg, fecha=FECHA):
    return pd.DataFrame({"fecha": [fecha, fecha], "ruta_id": [IDA, REG], "demanda": [ida, reg]})


# --- parámetros --------------------------------------------------------------
def test_yaml_de_operacion_carga_y_valida(p):
    assert p["workday"]["max_conduccion_dia_h"] == 9
    assert p["costs"]["smmlv"] == 1750905
    assert set(pa.CLASES) <= set(p["capacity_by_class"])


def test_parametro_F_vacio_se_rechaza_con_mensaje_claro(p):
    mal = copy.deepcopy(p)
    mal["workday"]["max_conduccion_dia_h"] = None
    with pytest.raises(ValueError, match="PENDIENTE: fuente"):
        pa.validar_parametros(mal)


def test_parametro_numerico_invalido(p):
    mal = copy.deepcopy(p)
    mal["target_occupancy"] = 1.5
    with pytest.raises(ValueError, match="target_occupancy"):
        pa.validar_parametros(mal)


def test_actualizar_parametros_y_clave_desconocida(p):
    nuevo = pa.actualizar_parametros(p, {"absenteeism": 0.2, "cycle_time.1.viaje_h": 4.0})
    assert nuevo["absenteeism"] == 0.2 and nuevo["cycle_time"][1]["viaje_h"] == 4.0
    assert p["absenteeism"] == 0.08                       # el original no cambia
    with pytest.raises(ValueError, match="desconocido"):
        pa.actualizar_parametros(p, {"no_existe": 1})
    with pytest.raises(ValueError, match="desconocido"):
        pa.actualizar_parametros(p, {"workday.no_existe": 1})


def test_ciclo_que_no_cabe_en_el_dia(p):
    with pytest.raises(ValueError, match="no cabe"):
        pa.actualizar_parametros(p, {"operating_hours": 4})


def test_jornada_semanal_ley_2101(p):
    f = lambda d: pa.jornada_semanal_h(p, d)  # noqa: E731
    assert [f("2023-07-14"), f("2023-07-15"), f("2024-07-15"), f("2025-07-15"), f("2026-07-14"),
            f("2026-07-15")] == [48, 47, 46, 44, 44, 42]


def test_escalar_parametro(p):
    mas = pa.escalar_parametro(p, "capacity_by_class", 1.1)
    assert mas["capacity_by_class"]["BUS"] == pytest.approx(p["capacity_by_class"]["BUS"] * 1.1)
    j = pa.escalar_parametro(p, "workday.jornada_semanal_h", 0.9)
    assert j["workday"]["jornada_semanal_h"][-1]["horas"] == pytest.approx(42 * 0.9)
    c = pa.escalar_parametro(p, "cycle_time.1", 1.1)
    assert c["cycle_time"][1]["viaje_h"] == pytest.approx(3.5 * 1.1)
    with pytest.raises(ValueError):
        pa.escalar_parametro(p, "nada", 1.1)


def test_capacidad_media_ponderada(p):
    m = pd.DataFrame({"ruta_id": ["x", "x"], "clase_vehiculo": ["BUS", "MICROBUS"],
                      "participacion_despachos": [0.25, 0.75]})
    assert pa.capacidad_media(p, m, "x") == pytest.approx(0.25 * 40 + 0.75 * 15)
    with pytest.raises(ValueError):
        pa.capacidad_media(p, m, "otra")


def test_capacidad_media_con_mezcla_real(p):
    mez = pa.cargar_mezcla(p)
    for r in mez.ruta_id.unique():
        c = pa.capacidad_media(p, mez, r)
        assert min(p["capacity_by_class"].values()) <= c <= max(p["capacity_by_class"].values())


# --- ejemplo a mano ----------------------------------------------------------------
def test_ejemplo_a_mano_corredor_1(p, rutas, mezcla_bus):
    # φ·C = 0,75·40 = 30. N_ida = ⌈100/30⌉ = 4; N_reg = ⌈50/30⌉ = 2; R = max = 4 (no la suma 6).
    # T = 2·3,5 + 2·0,5 = 8 h; ciclos/vehículo = ⌊18/8⌋ = 2; V = ⌈4/2⌉ = 2; F = ⌈2·1,1⌉ = 3.
    # Despachos ejecutados 2R = 8; subutilizados (4−4)+(4−2) = 2.
    # H = 4·2·3,5 = 28 h; CD = ⌈28/9⌉ = 4; por horas: 28/(42·1/7) = 4,667; K = ⌈4,667/(1−0,08)⌉ = ⌈5,07⌉ = 6.
    res = planificar(plan(100, 50), rutas, p, mezcla_bus)
    cd = res.por_corredor_dia.iloc[0]
    assert (cd.despachos_ida, cd.despachos_regreso, cd.ciclos) == (4, 2, 4)
    assert cd.despachos_ejecutados == 8 and cd.despachos_subutilizados == 2
    assert cd.ciclos_por_vehiculo == 2 and cd.vehiculos == 2
    assert res.flota.iloc[0].flota == 3
    assert cd.horas_conduccion == pytest.approx(28.0) and cd.conductores_dia == 4
    pl = res.plantilla.iloc[0]
    assert pl.jornada_semanal_h == 42 and pl.por_horas_semanales == pytest.approx(28 / 6)
    assert pl.plantilla == 6


def test_el_descanso_semanal_aumenta_la_plantilla_a_mano(p, rutas, mezcla_bus):
    """A mano, un día con φ·C = 30: demanda 1200 -> N = 40 y R = 40 ciclos; H = 40·2·3,5 = 280 h;
    CD = ⌈280/9⌉ = 32 conductores-día. Con descanso de 1 día por semana (trabajan 6 de 7), el pico con descanso
    es 32·7/6 = 37,33; sin descanso (7 de 7) es 32. Se comprueba esa fórmula y que quitar el descanso nunca
    aumenta la plantilla."""
    res = planificar(plan(1200, 1200), rutas, p, mezcla_bus)
    pl = res.plantilla.iloc[0]
    assert pl.conductores_dia_pico == 32
    assert pl.pico_con_descanso == pytest.approx(32 * 7 / 6)
    sin_descanso = pa.actualizar_parametros(p, {"workday.dias_trabajo_por_semana": 7})
    pl7 = planificar(plan(1200, 1200), rutas, sin_descanso, mezcla_bus).plantilla.iloc[0]
    assert pl7.pico_con_descanso == pytest.approx(32.0)
    assert pl.plantilla >= pl7.plantilla


def test_el_pico_con_descanso_domina_cuando_hay_picos_aislados(p, rutas, mezcla_bus):
    """30 días: 29 días con demanda baja y 1 día pico. Las horas totales son pocas, pero el día pico exige
    conductores: ahí domina el término del pico con descanso."""
    fechas = pd.date_range("2026-08-01", periods=30)
    dem = [60.0] * 29 + [3000.0]
    pln = pd.concat([pd.DataFrame({"fecha": fechas, "ruta_id": r, "demanda": dem}) for r in (IDA, REG)])
    pl = planificar(pln, rutas, p, mezcla_bus).plantilla.iloc[0]
    assert pl.pico_con_descanso > pl.por_horas_semanales
    esperado = int(np.ceil(pl.pico_con_descanso / (1 - p["absenteeism"]) - 1e-9))
    assert pl.plantilla == esperado


def test_dias_de_trabajo_invalidos(p):
    for malo in (0, 8):
        with pytest.raises(ValueError, match="dias_trabajo_por_semana"):
            pa.actualizar_parametros(p, {"workday.dias_trabajo_por_semana": malo})


def test_se_usa_el_maximo_y_no_la_suma_de_los_sentidos(p, rutas, mezcla_bus):
    res = planificar(plan(100, 100), rutas, p, mezcla_bus)
    assert res.por_corredor_dia.iloc[0].ciclos == 4          # y no 8
    res2 = planificar(plan(100, 1), rutas, p, mezcla_bus)
    assert res2.por_corredor_dia.iloc[0].ciclos == 4          # el sentido liviano no cambia los ciclos


def test_demanda_cero_da_cero(p, rutas, mezcla_bus):
    res = planificar(plan(0, 0), rutas, p, mezcla_bus)
    assert res.por_corredor_dia.iloc[0].ciclos == 0 and res.flota.iloc[0].flota == 0
    assert res.plantilla.iloc[0].plantilla == 0


def test_falta_un_sentido_da_error(p, rutas, mezcla_bus):
    solo_ida = plan(100, 50).iloc[[0]]
    with pytest.raises(ValueError, match="regreso"):
        planificar(solo_ida, rutas, p, mezcla_bus)


def test_demanda_negativa_o_ruta_desconocida(p, rutas, mezcla_bus):
    with pytest.raises(ValueError, match="no negativa"):
        planificar(plan(-1, 5), rutas, p, mezcla_bus)
    mal = plan(10, 10)
    mal.loc[0, "ruta_id"] = "99999-99999"
    with pytest.raises(ValueError, match="rutas.csv"):
        planificar(mal, rutas, p, mezcla_bus)


# --- monotonía ----------------------------------------------------------------------
def test_mas_capacidad_no_da_mas_vehiculos(p, rutas, mezcla_bus):
    antes = planificar(plan(500, 300), rutas, p, mezcla_bus)
    p2 = pa.escalar_parametro(p, "capacity_by_class", 1.1)
    despues = planificar(plan(500, 300), rutas, p2, mezcla_bus)
    assert despues.flota.iloc[0].flota <= antes.flota.iloc[0].flota
    assert despues.por_corredor_dia.iloc[0].despachos_ejecutados <= antes.por_corredor_dia.iloc[0].despachos_ejecutados


def test_mas_ausentismo_no_da_menos_conductores(p, rutas, mezcla_bus):
    antes = planificar(plan(500, 300), rutas, p, mezcla_bus).plantilla.iloc[0].plantilla
    despues = planificar(plan(500, 300), rutas, pa.actualizar_parametros(p, {"absenteeism": 0.2}),
                         mezcla_bus).plantilla.iloc[0].plantilla
    assert despues >= antes


def test_mas_demanda_no_da_menos_recursos(p, rutas, mezcla_bus):
    a = planificar(plan(500, 300), rutas, p, mezcla_bus)
    b = planificar(plan(550, 330), rutas, p, mezcla_bus)
    assert b.flota.iloc[0].flota >= a.flota.iloc[0].flota
    assert b.plantilla.iloc[0].plantilla >= a.plantilla.iloc[0].plantilla


def test_delta_cero_no_cambia_nada(p, rutas, mezcla_bus):
    p0 = pa.escalar_parametro(p, "absenteeism", 1.0)
    assert p0 == p
    assert planificar(plan(500, 300), rutas, p0, mezcla_bus).flota.equals(
        planificar(plan(500, 300), rutas, p, mezcla_bus).flota)


# --- no atendidos y costo ----------------------------------------------------------
def test_no_atendidos_y_costo_a_mano(p, rutas, mezcla_bus):
    res = planificar(plan(100, 50), rutas, p, mezcla_bus)
    # R = 4 ciclos, C = 40 -> capacidad ofrecida 160 por sentido. Demanda real 200 y 150:
    real = pd.DataFrame({"fecha": [FECHA, FECHA], "ruta_id": [IDA, REG], "y": [200.0, 150.0]})
    m = fleet.pasajeros_no_atendidos(res.por_ruta_dia, res.por_corredor_dia, real).set_index("ruta_id")
    assert m.loc[IDA, "no_atendidos"] == pytest.approx(40.0) and m.loc[REG, "no_atendidos"] == 0.0
    c = costo_con_demanda_real(res, real)
    esperado = (p["costs"]["c_veh_dia"] * 2 + p["costs"]["c_despacho"] * 8
                + p["costs"]["smmlv"] * (1 + p["costs"]["carga_prestacional"]) * 6 * 1 / 30
                + p["costs"]["c_no_atendido"] * 40)
    assert c["total"] == pytest.approx(esperado)
    assert c["pasajeros_no_atendidos"] == 40.0


def test_no_atendidos_ignora_dias_sin_demanda_real(p, rutas, mezcla_bus):
    res = planificar(plan(100, 50), rutas, p, mezcla_bus)
    real = pd.DataFrame({"fecha": [FECHA, FECHA], "ruta_id": [IDA, REG], "y": [float("nan"), 150.0]})
    m = fleet.pasajeros_no_atendidos(res.por_ruta_dia, res.por_corredor_dia, real)
    assert m["no_atendidos"].isna().sum() == 1


def test_config_de_operacion_tiene_fuentes_citadas():
    """Cada valor [F] debe tener su cita en el YAML (no se aceptan números normativos sin fuente)."""
    texto = (RAIZ / "configs/m1_operacion.yaml").read_text(encoding="utf-8")
    for cita in ("Ley 2101 de 2021", "Convenio 153", "Decreto 1469", "Circular 87", "PENDIENTE"):
        assert cita in texto
    assert cargar_yaml("configs/m1_operacion.yaml")["workday"]["max_conduccion_dia_h"] == 9
