"""Descarga reproducible de los datos del Módulo 1.

Responsable: Persona 1.

Fuente principal
    Superintendencia de Transporte. "Operación de pasajeros y despacho de vehículos
    en la modalidad de transporte de pasajeros por carretera". datos.gov.co,
    identificador ``eh75-8ah6``. https://www.datos.gov.co/d/eh75-8ah6

Codificación de municipios
    DANE. "DIVIPOLA - Códigos municipios" (corte 30-dic-2024). datos.gov.co,
    identificador ``gdxc-w37w``. https://www.datos.gov.co/d/gdxc-w37w

Por qué se agrega en el servidor
    El conjunto completo tiene ~49 millones de registros (uno por terminal, hora,
    clase de vehículo, nivel de servicio y ruta). Descargarlo entero no es viable ni
    necesario: la variable objetivo es *pasajeros por ruta y por día*. Por eso se usa
    la API Socrata (SoQL) para agregar en el servidor y solo se descargan:

    1. ``metadata_eh75-8ah6.json``: diccionario de variables publicado por la fuente.
    1b. ``resumen_anual.csv``: registros por año en todo el conjunto (tamaño y fechas imposibles).
    2. ``sistema_diario.csv``: totales nacionales por día y tipo de despacho
       (sirve para detectar días con reporte incompleto de todo el sistema).
    3. ``ranking_rutas.csv``: pasajeros totales por ruta (origen-destino).
    4. ``rutas_diario_clase.csv.gz``: para las rutas candidatas, agregados por
       ruta x día x clase de vehículo x tipo de despacho.
    5. ``divipola.csv``: códigos y nombres de municipios (DANE).

Uso
    python -m m1_demanda.data.download              # descarga todo lo que falte
    python -m m1_demanda.data.download --forzar     # vuelve a descargar todo

Los archivos quedan en ``data/m1_demanda/raw`` y se versionan en el repositorio,
de modo que el resto del pipeline NO depende de internet.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from m1_demanda.data.utils import cargar_config, ruta_abs

AGENTE = {"User-Agent": "trabajo3-rn-unal/1.0 (proyecto academico)"}


def _soql(cfg: dict, dataset: str, consulta: str) -> pd.DataFrame:
    """Ejecuta una consulta SoQL paginada y devuelve un DataFrame (todo como texto)."""
    url = f"{cfg['fuente']['base_url']}/resource/{dataset}.csv"
    tam = cfg["fuente"]["page_size"]
    partes, offset = [], 0
    while True:
        q = f"{consulta} LIMIT {tam} OFFSET {offset}"
        for intento in range(4):
            try:
                r = requests.get(url, params={"$query": q}, headers=AGENTE,
                                 timeout=cfg["fuente"]["timeout_s"])
                if r.status_code == 400:  # error de sintaxis: no tiene sentido reintentar
                    raise ValueError(f"SoQL inválida: {r.text[:400]}")
                r.raise_for_status()
                break
            except requests.RequestException as e:  # reintento con espera exponencial
                if intento == 3:
                    raise
                print(f"   reintento {intento + 1} ({type(e).__name__})")
                time.sleep(5 * 2 ** intento)
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        partes.append(df)
        if len(df) < tam:
            break
        offset += tam
    return pd.concat(partes, ignore_index=True)


def descargar_metadata(cfg: dict, destino) -> None:
    url = f"{cfg['fuente']['base_url']}/api/views/{cfg['fuente']['dataset_id']}.json"
    meta = requests.get(url, headers=AGENTE, timeout=60).json()
    meta["_descargado_utc"] = datetime.now(timezone.utc).isoformat()
    destino.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def descargar_sistema_diario(cfg: dict) -> pd.DataFrame:
    f0, f1 = cfg["fuente"]["fecha_inicio"], cfg["fuente"]["fecha_fin"]
    q = (
        "SELECT fecha_despacho AS fecha, tipo_despacho, count(*) AS n_registros, "
        "sum(despachos) AS despachos, sum(pasajeros) AS pasajeros "
        f"WHERE fecha_despacho >= '{f0}' AND fecha_despacho <= '{f1}' "
        "GROUP BY fecha, tipo_despacho ORDER BY fecha, tipo_despacho"
    )
    return _soql(cfg, cfg["fuente"]["dataset_id"], q)


def descargar_resumen_anual(cfg: dict) -> pd.DataFrame:
    """Registros, despachos y pasajeros por año sobre TODO el conjunto (sin filtrar
    fechas). Sirve para reportar el tamaño real del conjunto y las fechas imposibles."""
    q = (
        "SELECT date_extract_y(fecha_despacho) AS anio, tipo_despacho, count(*) AS n_registros, "
        "sum(despachos) AS total_despachos, sum(pasajeros) AS total_pasajeros, "
        "sum(case(pasajeros = 0, 1, true, 0)) AS registros_cero_pax "
        "GROUP BY anio, tipo_despacho ORDER BY anio, tipo_despacho"
    )
    return _soql(cfg, cfg["fuente"]["dataset_id"], q).rename(
        columns={"total_despachos": "despachos", "total_pasajeros": "pasajeros"})


def descargar_ranking(cfg: dict) -> pd.DataFrame:
    f0, f1 = cfg["fuente"]["fecha_inicio"], cfg["fuente"]["fecha_fin"]
    q = (
        "SELECT municipio_origen_ruta AS origen, municipio_destino_ruta AS destino, "
        "sum(pasajeros) AS pasajeros, sum(despachos) AS despachos, count(*) AS n_registros, "
        "count(distinct fecha_despacho) AS dias_con_registro, "
        "count(distinct terminal) AS n_terminales "
        f"WHERE tipo_despacho = '{cfg['limpieza']['tipo_despacho_objetivo']}' "
        f"AND fecha_despacho >= '{f0}' AND fecha_despacho <= '{f1}' "
        "GROUP BY origen, destino ORDER BY pasajeros DESC"
    )
    # Solo se necesitan las primeras filas, pero se guardan 2000 para documentar
    # la cola larga de rutas.
    url = f"{cfg['fuente']['base_url']}/resource/{cfg['fuente']['dataset_id']}.csv"
    r = requests.get(url, params={"$query": q + " LIMIT 2000"}, headers=AGENTE,
                     timeout=cfg["fuente"]["timeout_s"])
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), dtype=str)


def rutas_candidatas(cfg: dict, ranking: pd.DataFrame) -> list[tuple[str, str]]:
    """Top-N rutas por pasajeros + sentido de regreso + corredores del config."""
    top = ranking.head(cfg["seleccion_rutas"]["n_candidatas"])
    pares = set(zip(top.origen, top.destino))
    pares |= {(d, o) for o, d in pares}
    for o, d in cfg["seleccion_rutas"]["corredores"]:
        pares |= {(o, d), (d, o)}
    return sorted(pares)


def _expr_sobrecupo(cfg: dict) -> str:
    """Expresión SoQL que cuenta registros con más pasajeros que la cota plausible."""
    casos = ", ".join(
        f"clase_vehiculo = '{c}' AND pasajeros > {k} * despachos, 1"
        for c, k in cfg["limpieza"]["max_pax_por_despacho"].items()
    )
    return f"sum(case({casos}, true, 0))"


def descargar_rutas_diario(cfg: dict, pares: list[tuple[str, str]]) -> pd.DataFrame:
    lista = ", ".join(f"'{o}-{d}'" for o, d in pares)
    partes = []
    anio0 = int(cfg["fuente"]["fecha_inicio"][:4])
    anio1 = int(cfg["fuente"]["fecha_fin"][:4])
    for anio in range(anio0, anio1 + 1):  # por año para no exceder el tiempo de la API
        f0 = max(f"{anio}-01-01", cfg["fuente"]["fecha_inicio"])
        f1 = min(f"{anio}-12-31", cfg["fuente"]["fecha_fin"])
        q = (
            "SELECT municipio_origen_ruta AS origen, municipio_destino_ruta AS destino, "
            "fecha_despacho AS fecha, clase_vehiculo, tipo_despacho, "
            # Ojo: los alias no pueden llamarse igual que las columnas (``despachos``),
            # porque SoQL los usaría dentro de ``case(...)`` como agregados.
            "count(*) AS n_registros, sum(despachos) AS total_despachos, "
            "sum(pasajeros) AS total_pasajeros, "
            "sum(case(pasajeros = 0, 1, true, 0)) AS registros_cero_pax, "
            "sum(case(pasajeros = 0, despachos, true, 0)) AS despachos_cero_pax, "
            "sum(case(despachos = 0, 1, true, 0)) AS registros_cero_despachos, "
            f"{_expr_sobrecupo(cfg)} AS registros_sobrecupo "
            f"WHERE municipio_origen_ruta || '-' || municipio_destino_ruta IN ({lista}) "
            f"AND fecha_despacho >= '{f0}' AND fecha_despacho <= '{f1}' "
            "GROUP BY origen, destino, fecha, clase_vehiculo, tipo_despacho "
            "ORDER BY origen, destino, fecha, clase_vehiculo, tipo_despacho"
        )
        t0 = time.time()
        df = _soql(cfg, cfg["fuente"]["dataset_id"], q).rename(
            columns={"total_despachos": "despachos", "total_pasajeros": "pasajeros"})
        print(f"   {anio}: {len(df):>7} filas ({time.time() - t0:.0f} s)")
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def descargar_divipola(cfg: dict) -> pd.DataFrame:
    q = "SELECT cod_dpto, dpto, cod_mpio, nom_mpio, tipo_municipio, longitud, latitud ORDER BY cod_mpio"
    return _soql(cfg, cfg["fuente"]["divipola_id"], q)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--forzar", action="store_true", help="re-descargar aunque exista")
    args = ap.parse_args(argv)

    cfg = cargar_config(args.config)
    raw = ruta_abs(cfg, "raw")

    def hacer(nombre, fn):
        p = raw / nombre
        if p.exists() and not args.forzar:
            print(f"[ok] {nombre} ya existe (use --forzar para re-descargar)")
            return
        print(f"[..] descargando {nombre}")
        t0 = time.time()
        res = fn()
        if isinstance(res, pd.DataFrame):
            res.to_csv(p, index=False)
            print(f"[ok] {nombre}: {len(res)} filas en {time.time() - t0:.0f} s")

    hacer("metadata_eh75-8ah6.json", lambda: descargar_metadata(cfg, raw / "metadata_eh75-8ah6.json"))
    hacer("divipola.csv", lambda: descargar_divipola(cfg))
    hacer("resumen_anual.csv", lambda: descargar_resumen_anual(cfg))
    hacer("sistema_diario.csv", lambda: descargar_sistema_diario(cfg))
    hacer("ranking_rutas.csv", lambda: descargar_ranking(cfg))
    ranking = pd.read_csv(raw / "ranking_rutas.csv", dtype=str)
    pares = rutas_candidatas(cfg, ranking)
    print(f"     {len(pares)} rutas candidatas")
    hacer("rutas_diario_clase.csv.gz", lambda: descargar_rutas_diario(cfg, pares))


if __name__ == "__main__":
    main()
