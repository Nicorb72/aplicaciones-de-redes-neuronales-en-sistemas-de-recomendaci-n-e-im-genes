"""Construcción de las series de pasajeros por ruta y por día (Módulo 1).

Responsable: Persona 1.

Variable objetivo
    y[r, t] = suma de ``pasajeros`` de todos los registros con tipo_despacho = ORIGEN,
    municipio_origen_ruta = o(r) y municipio_destino_ruta = d(r), en la fecha t,
    sumando todas las terminales, horas, clases de vehículo y niveles de servicio.

Este script ejecuta todo el pipeline de datos a partir de los archivos crudos
versionados (no requiere internet):

    python -m m1_demanda.data.build_series [--config configs/m1_datos.yaml]

Salidas (data/m1_demanda/processed/)
    series_diarias.csv.gz    panel largo ruta x día con objetivo, banderas de calidad,
                             partición y variables de calendario.
    pasajeros_ancho.csv      matriz fecha x ruta con ``pasajeros_imp`` (cómodo para modelar).
    rutas.csv                catálogo de rutas con nombres DIVIPOLA, corredor y sentido.
    mezcla_clase_vehiculo.csv participación por clase de vehículo y pasajeros por despacho
                             (solo periodo de entrenamiento) -> insumo de flota.
    validez_sistema.csv      días válidos/ inválidos del sistema y segmentos.
    particiones.json         fechas de train/val/test/post_hueco y orígenes rodantes.
Además: reports/tables/m1_calidad_datos.csv (bitácora de limpieza por ruta).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from m1_demanda.data import calendar_features, clean, split
from m1_demanda.data.utils import cargar_config, id_ruta, ruta_abs

COLS_CAL = ["dia_semana", "mes", "festivo", "nombre_festivo", "puente", "vispera_puente",
            "ultimo_dia_puente", "dia_despues_puente", "semana_santa", "receso_octubre",
            "vacaciones_mitad_anio", "vacaciones_fin_anio", "temporada_fin_anio",
            "temporada_alta", "eleccion", "dias_hasta_festivo", "dias_desde_festivo",
            "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


def catalogo_rutas(cfg: dict, divipola: pd.DataFrame) -> pd.DataFrame:
    dv = divipola.set_index("cod_mpio")
    filas = []
    for i, (a, b) in enumerate(cfg["seleccion_rutas"]["corredores"], start=1):
        for sentido, (o, d) in (("ida", (a, b)), ("regreso", (b, a))):
            filas.append({
                "ruta_id": id_ruta(o, d), "origen": o, "destino": d,
                "nombre_origen": dv.nom_mpio.get(o, "?").title(),
                "depto_origen": dv.dpto.get(o, "?").title(),
                "nombre_destino": dv.nom_mpio.get(d, "?").title(),
                "depto_destino": dv.dpto.get(d, "?").title(),
                "corredor": i, "sentido": sentido,
                "ruta_regreso": id_ruta(d, o),
            })
    r = pd.DataFrame(filas)
    r["nombre"] = r.nombre_origen.str.replace(", D.C.", "", regex=False) + " → " + \
        r.nombre_destino.str.replace(", D.C.", "", regex=False)
    return r


def construir(cfg: dict) -> dict:
    proc = ruta_abs(cfg, "processed")
    tablas = ruta_abs(cfg, "tablas")

    cal = calendar_features.construir_calendario(cfg)
    cal.to_csv(ruta_abs(cfg, "external") / "calendario_colombia.csv", index=False)

    rutas_raw, sistema, divipola = clean.cargar_crudos(cfg)
    valid = clean.validez_sistema(cfg, sistema)
    valid.to_csv(proc / "validez_sistema.csv", index=False)

    ruta_dia = clean.agregar_ruta_dia(cfg, rutas_raw)
    panel, log = clean.limpiar(cfg, ruta_dia, valid, cal)

    cat = catalogo_rutas(cfg, divipola)
    panel = panel.merge(cat[["ruta_id", "nombre", "corredor", "sentido"]], on="ruta_id")
    panel["particion"] = split.asignar_particion(panel.fecha, cfg)
    panel = panel.merge(cal[["fecha"] + COLS_CAL], on="fecha", how="left")
    panel = panel.sort_values(["ruta_id", "fecha"]).reset_index(drop=True)

    panel.to_csv(proc / "series_diarias.csv.gz", index=False, date_format="%Y-%m-%d")
    ancho = panel.pivot(index="fecha", columns="ruta_id", values="pasajeros_imp")
    ancho.to_csv(proc / "pasajeros_ancho.csv", date_format="%Y-%m-%d")
    cat.to_csv(proc / "rutas.csv", index=False)
    mez = clean.mezcla_clase_vehiculo(cfg, rutas_raw, hasta=cfg["particiones"]["train"][1])
    mez.to_csv(proc / "mezcla_clase_vehiculo.csv", index=False)
    info = split.guardar_particiones(cfg, proc / "particiones.json")

    log = log.merge(cat[["ruta_id", "nombre"]], on="ruta_id")
    log.to_csv(tablas / "m1_calidad_datos.csv", index=False)

    resumen = {
        "mediana_registros_sistema_dia": valid.attrs["mediana_registros_dia"],
        "dias_validos_sistema": int(valid.dia_valido_sistema.sum()),
        "dias_invalidos_sistema": int((~valid.dia_valido_sistema).sum()),
        "segmentos": {int(s): [str(g.fecha.min().date()), str(g.fecha.max().date())]
                      for s, g in valid[valid.segmento > 0].groupby("segmento")},
        "n_rutas": int(panel.ruta_id.nunique()),
        "filas_panel": int(len(panel)),
        "origenes_rodantes_test": info["origenes_rodantes_test"],
    }
    (proc / "resumen_pipeline.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"panel": panel, "log": log, "resumen": resumen}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Construye las series ruta x día del Módulo 1")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = cargar_config(args.config)
    out = construir(cfg)
    print(json.dumps(out["resumen"], indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220)
    print(out["log"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
