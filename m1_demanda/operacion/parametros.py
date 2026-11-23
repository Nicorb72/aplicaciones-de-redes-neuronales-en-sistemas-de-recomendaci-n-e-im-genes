"""Propósito: Carga, validación y modificación de los parámetros de operación del Módulo 1.
Responsable: Persona 2.

Lee ``configs/m1_operacion.yaml``. Reglas:

* Ningún parámetro puede quedar vacío. Si un valor [F] (fuente primaria) está en ``null`` con la
  nota "PENDIENTE: fuente", se rechaza con un mensaje que indica cuál falta.
* ``actualizar_parametros`` aplica cambios con claves punteadas (``"workday.max_conduccion_dia_h"``,
  ``"cycle_time.1.viaje_h"``). Una clave desconocida da ``ValueError`` (contrato de ``predict.py``).
* ``escalar_parametro`` multiplica un parámetro (o grupo) por un factor; lo usa la sensibilidad ±10 %.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from common.config_loader import cargar_yaml, claves_vacias
from m1_demanda.data.utils import RAIZ

CONFIG_OPERACION = "configs/m1_operacion.yaml"
CLASES = ("BUS", "BUSETA", "MICROBUS", "CAMIONETA", "CAMPERO", "AUTOMOVIL")
# Parámetros [F] que no pueden faltar (el resto son [S] y también deben estar definidos).
OBLIGATORIOS_F = ("workday.jornada_semanal_h", "workday.max_conduccion_dia_h", "costs.smmlv")
CLAVES_NUMERICAS = (
    "target_occupancy", "operating_hours", "technical_reserve", "absenteeism",
    "workday.max_conduccion_dia_h", "workday.max_conduccion_continua_h", "workday.dias_trabajo_por_semana",
    "costs.smmlv", "costs.carga_prestacional", "costs.c_veh_dia", "costs.c_despacho",
    "costs.c_no_atendido", "sensitivity.delta",
)


# ---------------------------------------------------------------------------
# Carga y validación
# ---------------------------------------------------------------------------
def _obtener(p: dict, clave: str):
    """Devuelve el valor de una clave punteada o lanza KeyError."""
    nodo = p
    for seg in clave.split("."):
        if not isinstance(nodo, dict):
            raise KeyError(clave)
        llave = next((k for k in nodo if str(k) == seg), None)
        if llave is None:
            raise KeyError(clave)
        nodo = nodo[llave]
    return nodo


def validar_parametros(p: dict) -> dict:
    """Valida el diccionario de parámetros y lo devuelve. Lanza ``ValueError`` si algo no cumple."""
    for clave in OBLIGATORIOS_F:
        try:
            valor = _obtener(p, clave)
        except KeyError:
            raise ValueError(f"Falta el parámetro [F] {clave!r} en m1_operacion.yaml.") from None
        if valor is None:
            raise ValueError(f"PENDIENTE: fuente. El parámetro [F] {clave!r} está vacío; documente la fuente "
                             f"primaria y complete el valor en configs/m1_operacion.yaml.")
    vacias = claves_vacias(p)
    if vacias:
        raise ValueError("Parámetros vacíos en m1_operacion.yaml: " + ", ".join(vacias))
    for clave in CLAVES_NUMERICAS:
        try:
            v = _obtener(p, clave)
        except KeyError:
            raise ValueError(f"Falta el parámetro {clave!r} en m1_operacion.yaml.") from None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"El parámetro {clave!r} debe ser numérico, se recibió {v!r}.")
    if not 0 < p["target_occupancy"] <= 1:
        raise ValueError("target_occupancy debe estar en (0, 1].")
    if not 0 <= p["absenteeism"] < 1:
        raise ValueError("absenteeism debe estar en [0, 1).")
    if p["technical_reserve"] < 0:
        raise ValueError("technical_reserve no puede ser negativa.")
    if p["operating_hours"] <= 0 or p["operating_hours"] > 24:
        raise ValueError("operating_hours debe estar en (0, 24].")
    if p["workday"]["max_conduccion_dia_h"] <= 0:
        raise ValueError("workday.max_conduccion_dia_h debe ser positivo.")
    if not 1 <= p["workday"]["dias_trabajo_por_semana"] <= 7:
        raise ValueError("workday.dias_trabajo_por_semana debe estar entre 1 y 7.")
    faltan = [c for c in CLASES if c not in p["capacity_by_class"]]
    if faltan:
        raise ValueError(f"capacity_by_class no define las clases: {', '.join(faltan)}")
    if any(v <= 0 for v in p["capacity_by_class"].values()):
        raise ValueError("Las capacidades por clase deben ser positivas.")
    jor = p["workday"]["jornada_semanal_h"]
    if not isinstance(jor, list) or not jor:
        raise ValueError("workday.jornada_semanal_h debe ser una lista de {desde, horas}.")
    desdes = [str(j["desde"]) for j in jor]
    if desdes != sorted(desdes) or any(j["horas"] <= 0 for j in jor):
        raise ValueError("jornada_semanal_h debe estar ordenada por fecha y tener horas positivas.")
    for c, ct in p["cycle_time"].items():
        if ct["viaje_h"] <= 0 or ct["alistamiento_h"] < 0:
            raise ValueError(f"cycle_time del corredor {c}: viaje_h > 0 y alistamiento_h >= 0.")
        ciclo = 2 * ct["viaje_h"] + 2 * ct["alistamiento_h"]
        if ciclo > p["operating_hours"]:
            raise ValueError(f"El ciclo del corredor {c} ({ciclo} h) no cabe en operating_hours ({p['operating_hours']} h).")
    return p


def cargar_parametros(ruta: str | Path = CONFIG_OPERACION) -> dict:
    """Lee y valida ``configs/m1_operacion.yaml``."""
    p = cargar_yaml(ruta, permitir_vacios=True)
    return validar_parametros(p)


def cargar_mezcla(p: dict) -> pd.DataFrame:
    """Participación por clase y ruta (solo entrenamiento), de ``mezcla_clase_vehiculo.csv``."""
    df = pd.read_csv(RAIZ / p["class_mix"], dtype={"ruta_id": str, "origen": str, "destino": str})
    return df[["ruta_id", "clase_vehiculo", "participacion_despachos"]]


# ---------------------------------------------------------------------------
# Modificación
# ---------------------------------------------------------------------------
def actualizar_parametros(p: dict, cambios: dict | None) -> dict:
    """Devuelve una copia de ``p`` con ``cambios`` (claves punteadas). Clave desconocida -> ValueError."""
    nuevo = copy.deepcopy(p)
    for clave, valor in (cambios or {}).items():
        segs = str(clave).split(".")
        nodo = nuevo
        for seg in segs[:-1]:
            llave = next((k for k in nodo if str(k) == seg), None) if isinstance(nodo, dict) else None
            if llave is None:
                raise ValueError(f"Parámetro de operación desconocido: {clave!r}")
            nodo = nodo[llave]
        if not isinstance(nodo, dict):
            raise ValueError(f"Parámetro de operación desconocido: {clave!r}")
        llave = next((k for k in nodo if str(k) == segs[-1]), None)
        if llave is None:
            raise ValueError(f"Parámetro de operación desconocido: {clave!r}")
        nodo[llave] = valor
    return validar_parametros(nuevo)


def escalar_parametro(p: dict, clave: str, factor: float) -> dict:
    """Copia de ``p`` con el parámetro ``clave`` multiplicado por ``factor``.

    Si ``clave`` apunta a un número, lo escala; a un diccionario de números (``capacity_by_class``,
    ``cycle_time.1``), escala todos; a una lista de ``{desde, horas}`` (jornada), escala las horas.
    """
    nuevo = copy.deepcopy(p)
    try:
        nodo = _obtener(nuevo, clave)
    except KeyError:
        raise ValueError(f"Parámetro de operación desconocido: {clave!r}") from None

    def esc(v):
        if isinstance(v, dict):
            return {k: esc(x) for k, x in v.items()}
        if isinstance(v, list):
            return [esc(x) if isinstance(x, dict) and "horas" in x else x for x in v]
        return v * factor

    if isinstance(nodo, list):
        nuevo_valor = [{**x, "horas": x["horas"] * factor} for x in nodo]
    elif isinstance(nodo, dict):
        nuevo_valor = {k: (x * factor if not isinstance(x, (dict, list)) else esc(x)) for k, x in nodo.items()}
    else:
        nuevo_valor = nodo * factor
    segs = clave.split(".")
    padre = _obtener(nuevo, ".".join(segs[:-1])) if len(segs) > 1 else nuevo
    llave = next(k for k in padre if str(k) == segs[-1])
    padre[llave] = nuevo_valor
    return validar_parametros(nuevo)


# ---------------------------------------------------------------------------
# Consultas
# ---------------------------------------------------------------------------
def jornada_semanal_h(p: dict, fecha) -> float:
    """Jornada semanal máxima (horas) vigente en ``fecha`` según Ley 2101 de 2021."""
    f = pd.Timestamp(fecha).strftime("%Y-%m-%d")
    vigente = None
    for j in p["workday"]["jornada_semanal_h"]:
        if str(j["desde"]) <= f:
            vigente = float(j["horas"])
    if vigente is None:
        raise ValueError(f"No hay jornada definida para {f}.")
    return vigente


def tiempos_ciclo(p: dict, corredor: int) -> dict:
    """Tiempos del corredor: ``viaje_h``, ``alistamiento_h``, ``ciclo_h`` y ``ciclos_por_vehiculo``."""
    try:
        ct = _obtener(p["cycle_time"], str(corredor))
    except KeyError:
        raise ValueError(f"No hay cycle_time para el corredor {corredor}.") from None
    ciclo = 2 * ct["viaje_h"] + 2 * ct["alistamiento_h"]
    vueltas = int(p["operating_hours"] // ciclo)
    if vueltas < 1:
        raise ValueError(f"El ciclo del corredor {corredor} ({ciclo} h) no cabe en operating_hours.")
    return {"viaje_h": float(ct["viaje_h"]), "alistamiento_h": float(ct["alistamiento_h"]),
            "ciclo_h": float(ciclo), "ciclos_por_vehiculo": vueltas}


def capacidad_media(p: dict, mezcla: pd.DataFrame, ruta_id: str) -> float:
    """C_r = Σ_k w_{r,k}·cap_k con w = participación de despachos por clase (solo entrenamiento)."""
    m = mezcla[mezcla["ruta_id"] == ruta_id]
    if m.empty:
        raise ValueError(f"La ruta {ruta_id} no está en mezcla_clase_vehiculo.csv.")
    cap = p["capacity_by_class"]
    desconocidas = sorted(set(m["clase_vehiculo"]) - set(cap))
    if desconocidas:
        raise ValueError(f"Clases sin capacidad definida: {desconocidas}")
    w = m["participacion_despachos"].to_numpy(dtype=float)
    c = m["clase_vehiculo"].map(cap).to_numpy(dtype=float)
    return float((w * c).sum() / w.sum())
