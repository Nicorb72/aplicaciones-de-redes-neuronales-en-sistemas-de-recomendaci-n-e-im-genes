"""Propósito: Análisis de temporada alta: vehículos adicionales y señales de demanda censurada por la oferta.
Responsable: Persona 2.

Temporada alta = día objetivo con ``temporada_alta`` verdadero en el calendario (receso de octubre, puentes,
Semana Santa, vacaciones y fin de año; ``m1_demanda/data/calendar_features.py``).

1. **Vehículos adicionales a contratar** (``vehiculos_adicionales``): por corredor, la diferencia entre el pico de
   vehículos de los días de temporada alta y el pico de los días base dentro del mismo horizonte (mínimo 0).
   Es lo que habría que contratar *encima* de la flota que ya basta para un día normal.

2. **Indicios de demanda censurada** (``indicios_censura``, decisión D9). Si no hay sillas, el pasajero no
   aparece en los datos, así que el pronóstico de picos queda sesgado hacia abajo. No se modela la censura
   (la oferta diaria no es observable); se miden *indicios*: la fracción de días-ruta en que los pasajeros por
   despacho efectivo se acercan a la capacidad media de la ruta, en temporada alta frente a base. LIMITACIÓN:
   los pasajeros de origen no miden la ocupación real (los que suben en tránsito no están) y la capacidad por
   clase es un supuesto [S] de ``configs/m1_operacion.yaml``; son indicios, no pruebas.
"""
from __future__ import annotations

import pandas as pd

from m1_demanda.data.utils import RAIZ, cargar_config
from m1_demanda.operacion import parametros as pa


def vehiculos_adicionales(por_corredor_dia: pd.DataFrame, alta_por_fecha: pd.Series) -> pd.DataFrame:
    """Pico de vehículos en días de temporada alta frente a días base, por corredor.

    ``por_corredor_dia``: salida de ``fleet.dimensionar_flota`` (columnas ``fecha, corredor, vehiculos``).
    ``alta_por_fecha``: Serie booleana indexada por fecha (``True`` = temporada alta).
    Devuelve ``corredor, pico_alta, pico_base, adicionales``. Si el horizonte no tiene días de alguno de los
    dos tipos, el pico faltante queda en NaN y ``adicionales`` también.
    """
    cols = ["fecha", "corredor", "vehiculos"] + (["conductores_dia"] if "conductores_dia" in por_corredor_dia else [])
    d = por_corredor_dia[cols].copy()
    d["alta"] = d["fecha"].map(alta_por_fecha).astype(bool)

    def adicional(g: pd.DataFrame, col: str) -> tuple[float, float, float]:
        pa_ = g.loc[g["alta"], col].max() if g["alta"].any() else float("nan")
        pb = g.loc[~g["alta"], col].max() if (~g["alta"]).any() else float("nan")
        return pa_, pb, (max(0.0, pa_ - pb) if pd.notna(pa_) and pd.notna(pb) else float("nan"))

    filas = []
    for c, g in d.groupby("corredor"):
        pa_, pb, adic = adicional(g, "vehiculos")
        fila = {"corredor": int(c), "pico_alta": pa_, "pico_base": pb, "adicionales": adic,
                "dias_alta": int(g["alta"].sum()), "dias_base": int((~g["alta"]).sum())}
        if "conductores_dia" in g:
            ca, cb, cadic = adicional(g, "conductores_dia")
            fila.update({"cd_pico_alta": ca, "cd_pico_base": cb, "conductores_adicionales": cadic})
        filas.append(fila)
    return pd.DataFrame(filas)


def indicios_censura(umbral: float = 0.9, params: dict | None = None) -> pd.DataFrame:
    """Fracción de días-ruta de ENTRENAMIENTO con ocupación de origen cercana a la capacidad, por ruta.

    ``ocupacion = pax_por_despacho_efectivo / capacidad_media_ruta`` (pasajeros de origen por despacho que
    salió con gente). Devuelve por ruta: ``frac_alta`` (en temporada alta), ``frac_base`` y ``razon``.
    """
    p = params or pa.cargar_parametros()
    cfg = cargar_config()
    proc = RAIZ / cfg["rutas_salida"]["processed"]
    s = pd.read_csv(proc / "series_diarias.csv.gz", parse_dates=["fecha"], dtype={"ruta_id": str},
                    usecols=["fecha", "ruta_id", "pasajeros", "despachos_efectivos", "particion", "temporada_alta",
                             "imputado"])
    s = s[(s["particion"] == "train") & s["pasajeros"].notna() & ~s["imputado"] & (s["despachos_efectivos"] > 0)]
    mezcla = pa.cargar_mezcla(p)
    cap = {r: pa.capacidad_media(p, mezcla, r) for r in s["ruta_id"].unique()}
    s = s.assign(ocupacion=(s["pasajeros"] / s["despachos_efectivos"]) / s["ruta_id"].map(cap))
    s["cerca"] = s["ocupacion"] >= umbral
    filas = []
    for r, g in s.groupby("ruta_id"):
        fa = float(g.loc[g["temporada_alta"], "cerca"].mean())
        fb = float(g.loc[~g["temporada_alta"], "cerca"].mean())
        filas.append({"ruta_id": r, "capacidad_media": cap[r], "umbral_ocupacion": umbral,
                      "dias_alta": int(g["temporada_alta"].sum()), "dias_base": int((~g["temporada_alta"]).sum()),
                      "frac_alta": fa, "frac_base": fb, "razon": fa / fb if fb > 0 else float("nan")})
    return pd.DataFrame(filas)
