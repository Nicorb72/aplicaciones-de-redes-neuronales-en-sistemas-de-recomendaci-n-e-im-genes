"""Propósito: Interfaz de predicción de demanda del Módulo 1 (contrato con la app de la Persona 5).
Responsable: Persona 2.

Contrato
--------
``list_routes() -> pd.DataFrame``
    Columnas: ``ruta_id, nombre, corredor, sentido, ruta_regreso``.

``forecast(route_id, quantile=0.9, op_params=None, *, origen=None) -> pd.DataFrame``
    * ``quantile``: ``None`` -> la flota y los conductores se planean con la MEDIA; ``0.9`` -> con el
      cuantil P90; cualquier otro valor -> ``ValueError``.
    * ``op_params``: diccionario con claves punteadas de ``configs/m1_operacion.yaml`` que sobrescriben
      el valor por defecto (p. ej. ``{"target_occupancy": 0.8, "workday.max_conduccion_dia_h": 8}``).
      Una clave desconocida da ``ValueError``.
    * ``origen``: primer día pronosticado (``"YYYY-MM-DD"``). ``None`` -> el día siguiente al último dato
      disponible (hoy 2026-03-31). Un origen dentro del periodo con datos reproduce un pronóstico
      pasado y rellena ``real``.

    Columnas del resultado (30 filas, una por día de horizonte):

    ==============  ====================================================================================
    fecha           día pronosticado
    ruta_id         ruta pedida
    horizonte       1..30
    pronostico      pasajeros esperados (media)
    p90             cuantil 0,9 de pasajeros
    base_plan       demanda con la que se dimensiona (media o P90 según ``quantile``)
    despachos       despachos requeridos por la ruta ese día (N)
    vehiculos       vehículos necesarios ese día en el CORREDOR de la ruta (V)
    conductores     conductores-día necesarios ese día en el CORREDOR (CD)
    real            pasajeros observados (limpios) o NaN
    ==============  ====================================================================================

    ``df.attrs``: ``modelo, origen, ultimo_dato, flota_corredor, plantilla_conductores, advertencias,
    quantile, parametros_operacion``. (pandas marca ``attrs`` como experimental y se pierde en algunas
    operaciones; si la app lo necesita estable se puede cambiar por una tupla ``(df, resumen)``.)

Garantías: no reentrena, no usa internet, carga solo artefactos de ``models/m1_demanda/`` y datos
versionados; los recursos se cachean (``lru_cache``) para que cada llamada tarde menos de 2 s.

Modelo (v1): usa la red marcada en ``models/m1_demanda/modelo_final.json`` (la familia con menor MASE de
VALIDACIÓN, ``mlp`` o ``rnn``). Si no hay red, o si el historial de una ruta tiene datos faltantes, cruza un
hueco de reporte o toca un evento no previsible, esa ruta usa el ingenuo estacional (última semana) y se
agrega una advertencia. ``attrs["modelo"]`` dice qué se usó (``"mlp"``, ``"rnn"``, ``"ingenuo_estacional"`` o
una combinación ``"mlp+ingenuo_estacional"``). Si el origen cae en entrenamiento o validación se advierte que
el pronóstico no es fuera de muestra.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from common.config_loader import cargar_yaml
from m1_demanda.baselines import IngenuoEstacional
from m1_demanda.data.utils import RAIZ, cargar_config
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.planificar import planificar
from m1_demanda.train import cargar_modelo, cfg_con_ventana
from m1_demanda.ventanas import CONFIG_M1, cargar_panel, construir_ventanas, origen_produccion

MODELOS = RAIZ / "models" / "m1_demanda"
CUANTIL_PLAN = 0.9
COLUMNAS = ["fecha", "ruta_id", "horizonte", "pronostico", "p90", "base_plan",
            "despachos", "vehiculos", "conductores", "real"]


# ---------------------------------------------------------------------------
# Recursos en caché
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _recursos():
    cfg_m1 = cargar_yaml(CONFIG_M1)
    cfg_datos = cargar_config()
    panel = cargar_panel(cfg_m1, cfg_datos)
    params = pa.cargar_parametros()
    mezcla = pa.cargar_mezcla(params)
    return cfg_m1, cfg_datos, panel, params, mezcla


@lru_cache(maxsize=1)
def _ingenuo():
    """Ingenuo con el P90 guardado en ``models/m1_demanda/ingenuo.json`` (si no existe, se calibra)."""
    cfg_m1, cfg_datos, panel, _, _ = _recursos()
    ruta = MODELOS / "ingenuo.json"
    if ruta.exists():
        return IngenuoEstacional.cargar(ruta), None
    modelo = IngenuoEstacional(cuantil=float(cfg_m1["quantile"])).ajustar(panel, cfg_m1, cfg_datos)
    return modelo, ("No se encontró models/m1_demanda/ingenuo.json; el P90 del ingenuo se calibró con "
                    "validación al vuelo (ejecute `python -m m1_demanda.run --mode train --modelo ingenuo`).")


@lru_cache(maxsize=1)
def _red():
    """Red final (``models/m1_demanda/<familia>_final``) según ``modelo_final.json``; ``None`` si no hay.

    ``modelo_final.json`` lo escribe ``python -m m1_demanda.run --mode final`` (familia con menor MASE medio de
    VALIDACIÓN). Devuelve ``(PronosticadorRed | None, aviso | None)``.
    """
    _, _, panel, _, _ = _recursos()
    marca = MODELOS / "modelo_final.json"
    if not marca.exists():
        return None, ("No hay una red entrenada en models/m1_demanda/ (falta modelo_final.json): se usa el "
                      "ingenuo estacional.")
    familia = json.loads(marca.read_text(encoding="utf-8"))["familia"]
    base = MODELOS / f"{familia}_final"
    if not base.with_suffix(".pt").exists():
        return None, f"modelo_final.json apunta a {familia}_final, que no existe: se usa el ingenuo estacional."
    return cargar_modelo(base, n_rutas=len(panel.rutas), device="cpu"), None


def _pronosticar_ruta(panel, cfg_m1: dict, idx: int, origen: pd.Timestamp, horizonte: int):
    """Pronóstico de una ruta: red si hay historial válido; si no, ingenuo. Devuelve ``(mu, q, modelo, aviso)``."""
    red, _ = _red()
    aviso = None
    if red is not None:
        cfg = cfg_con_ventana(cfg_m1, int(red.hp["window"]))
        v = construir_ventanas(panel, cfg, [origen], rutas=[idx], con_objetivo=False)
        if len(v) == 1:
            mu, q = red.predecir(panel, v)
            return mu[0], q[0], red.familia, None
        aviso = (f"{panel.rutas.loc[idx, 'ruta_id']}: el historial de {red.hp['window']} días antes de "
                 f"{origen.date()} tiene datos faltantes, cruza un hueco de reporte o toca un evento no previsible; "
                 "se usa el ingenuo estacional (última semana).")
    ingenuo, _ = _ingenuo()
    mu, q = ingenuo.predecir_ruta(panel, idx, origen, horizonte)
    return mu, q, "ingenuo_estacional", aviso


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def list_routes() -> pd.DataFrame:
    """Rutas disponibles: ``ruta_id, nombre, corredor, sentido, ruta_regreso``."""
    rutas = _recursos()[2].rutas
    return rutas[["ruta_id", "nombre", "corredor", "sentido", "ruta_regreso"]].copy().reset_index(drop=True)


def _validar_cuantil(quantile) -> float | None:
    if quantile is None:
        return None
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float, np.floating)):
        raise ValueError(f"quantile debe ser None o {CUANTIL_PLAN}, se recibió {quantile!r}.")
    if abs(float(quantile) - CUANTIL_PLAN) > 1e-9:
        raise ValueError(f"quantile debe ser None (media) o {CUANTIL_PLAN} (P90); se recibió {quantile!r}.")
    return CUANTIL_PLAN


def _resolver_origen(origen, panel) -> pd.Timestamp:
    if origen is None:
        return origen_produccion(panel)
    try:
        o = pd.Timestamp(origen)
    except (ValueError, TypeError):
        raise ValueError(f"origen {origen!r} no es una fecha válida (use 'YYYY-MM-DD').") from None
    if pd.isna(o):
        raise ValueError("origen no puede ser nulo.")
    o = o.normalize()
    if o > origen_produccion(panel):
        raise ValueError(f"origen {o.date()} es posterior al día siguiente al último dato "
                         f"({panel.ultimo_dato.date()}): no hay historial para pronosticar.")
    return o


def forecast(route_id: str, quantile: float | None = CUANTIL_PLAN, op_params: dict | None = None, *,
             origen: str | None = None) -> pd.DataFrame:
    """Pronóstico a 30 días de una ruta y operación (despachos, vehículos y conductores) de su corredor."""
    cfg_m1, cfg_datos, panel, params_base, mezcla = _recursos()
    horizonte = int(cfg_m1["horizon"])

    rutas = panel.rutas
    if not isinstance(route_id, str) or route_id not in set(rutas["ruta_id"]):
        raise ValueError(f"route_id {route_id!r} desconocido. Rutas válidas: {', '.join(rutas['ruta_id'])}")
    q = _validar_cuantil(quantile)
    if op_params is not None and not isinstance(op_params, dict):
        raise ValueError("op_params debe ser un diccionario o None.")
    params = pa.actualizar_parametros(params_base, op_params)
    o = _resolver_origen(origen, panel)

    fila = rutas[rutas["ruta_id"] == route_id].iloc[0]
    regreso = fila["ruta_regreso"]
    advertencias: list[str] = []
    _, aviso_red = _red()
    if aviso_red:
        advertencias.append(aviso_red)
    _, aviso_ing = _ingenuo()
    if aviso_ing and (_red()[0] is None):
        advertencias.append(aviso_ing)
    if o <= pd.Timestamp(cfg_datos["particiones"]["val"][1]):
        advertencias.append(f"El origen {o.date()} cae dentro del periodo de entrenamiento o validación: el "
                            "pronóstico NO es fuera de muestra y su error será optimista.")

    fechas = pd.date_range(o, periods=horizonte, freq="D")
    resultados, modelos_usados = {}, set()
    for rid in (route_id, regreso):
        idx = int(rutas.index[rutas["ruta_id"] == rid][0])
        try:
            mu, qq, nombre, aviso = _pronosticar_ruta(panel, cfg_m1, idx, o, horizonte)
        except ValueError as e:
            raise ValueError(f"No se puede pronosticar {rid} desde {o.date()}: {e}") from None
        if aviso:
            advertencias.append(aviso)
        modelos_usados.add(nombre)
        resultados[rid] = (idx, mu, qq)
    modelo_attr = modelos_usados.pop() if len(modelos_usados) == 1 else "+".join(sorted(modelos_usados))

    base = {rid: (qq if q is not None else mu) for rid, (_, mu, qq) in resultados.items()}
    plan = pd.concat([pd.DataFrame({"fecha": fechas, "ruta_id": rid, "demanda": base[rid]})
                      for rid in resultados], ignore_index=True)
    res = planificar(plan, rutas, params, mezcla)

    idx, mu, qq = resultados[route_id]
    pr = res.por_ruta_dia[res.por_ruta_dia["ruta_id"] == route_id].set_index("fecha")
    pc = res.por_corredor_dia[res.por_corredor_dia["corredor"] == int(fila["corredor"])].set_index("fecha")
    i0 = panel.idx(o)
    real = panel.y_obs[i0:i0 + horizonte, idx].copy()
    real[panel.imputado[i0:i0 + horizonte, idx]] = np.nan   # solo se muestra lo realmente observado

    df = pd.DataFrame({
        "fecha": fechas, "ruta_id": route_id, "horizonte": np.arange(1, horizonte + 1),
        "pronostico": mu, "p90": qq, "base_plan": base[route_id],
        "despachos": pr.loc[fechas, "despachos"].to_numpy(),
        "vehiculos": pc.loc[fechas, "vehiculos"].to_numpy(),
        "conductores": pc.loc[fechas, "conductores_dia"].to_numpy(),
        "real": real,
    })[COLUMNAS]
    corredor = int(fila["corredor"])
    df.attrs.update({
        "modelo": modelo_attr,
        "origen": o.strftime("%Y-%m-%d"),
        "ultimo_dato": panel.ultimo_dato.strftime("%Y-%m-%d"),
        "flota_corredor": int(res.flota.set_index("corredor").loc[corredor, "flota"]),
        "plantilla_conductores": int(res.plantilla.set_index("corredor").loc[corredor, "plantilla"]),
        "advertencias": advertencias,
        "quantile": q,
        "parametros_operacion": params,
    })
    return df
