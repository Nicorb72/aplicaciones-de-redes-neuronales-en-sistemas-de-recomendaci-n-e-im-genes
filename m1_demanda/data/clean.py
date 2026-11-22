"""Limpieza de los datos de demanda del Módulo 1.

Responsable: Persona 1.

Reglas de limpieza (todas parametrizadas en configs/m1_datos.yaml, sección ``limpieza``)
------------------------------------------------------------------------------------
R1. Fechas fuera de rango. Se descartan registros con fecha anterior al inicio
    declarado por la fuente (2021-01-01) o posterior a la fecha de corte; incluye
    fechas imposibles (años 2202, 2222, 6202) que aparecen en el conjunto.
R2. Días con reporte incompleto del SISTEMA. Si el número de registros nacionales
    de un día es < ``umbral_cobertura_sistema`` x mediana global, el día se trata
    como FALTANTE (NaN) en todas las rutas, nunca como cero. Así se excluye el
    hueco casi total de 2024 y diciembre de 2025.
R3. Variable objetivo. Solo despachos de tipo ORIGEN: pasajeros que abordan en la
    terminal de origen de la ruta. Los pasajeros que abordan en terminales de
    tránsito se guardan aparte (``pasajeros_transito``) pero no forman el objetivo,
    para no contar dos veces la misma ruta.
R4. Despachos con cero pasajeros. Se conservan (son vehículos que salieron) pero se
    cuentan aparte (``despachos_cero_pax``) y se calcula ``despachos_efectivos``.
    No afectan la suma de pasajeros.
R5. Día de ruta sin registros (o con 0 pasajeros) cuando el sistema sí reportó:
    se interpreta como falta de reporte de la terminal (las rutas elegidas tienen
    >1 000 pasajeros/día en promedio, un cero real es inverosímil) -> NaN.
R6. Atípicos. (a) Caída extrema: pasajeros < ``fraccion_caida_extrema`` x mediana
    del mismo día de la semana -> falla de reporte, salvo en los propios festivos
    y 24/31-dic (las rutas pendulares caen >90 % el 1-ene). (b) z robusto del log(1+pasajeros) frente a la mediana del mismo día de
    la semana en +-k semanas (escala MAD). Si |z| > umbral y el día NO está cerca de
    un evento de calendario (festivo, puente, Semana Santa, fin de año, elecciones) se marca
    como atípico y se trata como faltante. Los picos de festivos se conservan: son
    demanda real, justamente lo que se quiere pronosticar.
R7. Imputación. Huecos de hasta ``max_hueco_imputable_dias`` dentro de un segmento
    se imputan con el promedio del mismo día de la semana anterior y siguiente
    (ingenuo estacional bilateral); si falta alguno, con la mediana del mismo día
    de la semana en +-4 semanas; en último caso, interpolación lineal.
    Se marca ``imputado = True``. Los huecos largos (2024, dic-2025) NO se imputan:
    la serie se divide en segmentos.
R8. Eventos no previsibles (Paro Nacional 2021) se marcan con
    ``evento_no_previsible``; no se modifican los valores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from m1_demanda.data.utils import id_ruta, ruta_abs


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_crudos(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = ruta_abs(cfg, "raw")
    rutas = pd.read_csv(raw / "rutas_diario_clase.csv.gz",
                        dtype={"origen": str, "destino": str, "clase_vehiculo": str})
    rutas["fecha"] = pd.to_datetime(rutas.fecha.str[:10])
    sistema = pd.read_csv(raw / "sistema_diario.csv")
    sistema["fecha"] = pd.to_datetime(sistema.fecha.str[:10])
    divipola = pd.read_csv(raw / "divipola.csv", dtype=str)
    return rutas, sistema, divipola


# ---------------------------------------------------------------------------
# R2: validez de cada día a nivel sistema
# ---------------------------------------------------------------------------
def validez_sistema(cfg: dict, sistema: pd.DataFrame) -> pd.DataFrame:
    lim = cfg["limpieza"]
    fechas = pd.date_range(cfg["fuente"]["fecha_inicio"], lim["fecha_corte_datos"], freq="D")
    n = sistema.groupby("fecha").n_registros.sum().reindex(fechas, fill_value=0)
    mediana = float(n[n > 0].median())
    out = pd.DataFrame({
        "fecha": fechas,
        "n_registros_sistema": n.to_numpy(),
        "ratio_sistema": (n / mediana).to_numpy(),
    })
    out["dia_valido_sistema"] = out.ratio_sistema >= lim["umbral_cobertura_sistema"]
    # Huecos cortos del sistema (<= max_hueco_imputable_dias) rodeados de días válidos
    # no parten la serie: se imputarán después. Los largos (2024, dic-2025) sí la parten.
    v = out.dia_valido_sistema
    racha = v.ne(v.shift()).cumsum()
    largo = v.groupby(racha).transform("size")
    interior = (racha != racha.iloc[0]) & (racha != racha.iloc[-1])
    hueco_corto = ~v & (largo <= lim["max_hueco_imputable_dias"]) & interior
    en_segmento = v | hueco_corto
    # Segmentos: bloques consecutivos (1, 2, 3, ...). 0 = fuera de todo segmento.
    cambio = en_segmento.ne(en_segmento.shift()).cumsum()
    ids = {c: i + 1 for i, c in enumerate(cambio[en_segmento].unique())}
    out["segmento"] = np.where(en_segmento, cambio.map(ids), 0).astype(int)
    # Los segmentos de menos de 28 días no sirven para modelar: se anulan.
    tam = out.groupby("segmento").fecha.transform("size")
    out.loc[(out.segmento > 0) & (tam < 28), "segmento"] = 0
    ids = {s: i + 1 for i, s in enumerate(sorted(out.segmento[out.segmento > 0].unique()))}
    out["segmento"] = out.segmento.map(lambda s: ids.get(s, 0))
    out.attrs["mediana_registros_dia"] = mediana
    return out


# ---------------------------------------------------------------------------
# R3-R5: agregación por ruta y día
# ---------------------------------------------------------------------------
def agregar_ruta_dia(cfg: dict, rutas_raw: pd.DataFrame) -> pd.DataFrame:
    pares = [tuple(p) for c in cfg["seleccion_rutas"]["corredores"] for p in (c, c[::-1])]
    sel = rutas_raw[[(o, d) in set(pares) for o, d in zip(rutas_raw.origen, rutas_raw.destino)]]
    obj = cfg["limpieza"]["tipo_despacho_objetivo"]
    num = ["n_registros", "despachos", "pasajeros", "registros_cero_pax",
           "despachos_cero_pax", "registros_cero_despachos", "registros_sobrecupo"]
    o = sel[sel.tipo_despacho == obj].groupby(["origen", "destino", "fecha"])[num].sum()
    t = (sel[sel.tipo_despacho != obj].groupby(["origen", "destino", "fecha"])
         [["pasajeros", "despachos"]].sum()
         .rename(columns={"pasajeros": "pasajeros_transito", "despachos": "despachos_transito"}))
    df = o.join(t, how="outer").reset_index()
    df["ruta_id"] = [id_ruta(a, b) for a, b in zip(df.origen, df.destino)]
    return df


def mezcla_clase_vehiculo(cfg: dict, rutas_raw: pd.DataFrame, hasta: str) -> pd.DataFrame:
    """Participación de cada clase de vehículo en los despachos de origen de cada ruta
    y pasajeros medios por despacho (insumo para el dimensionamiento de flota).
    Solo con datos hasta ``hasta`` (fin de entrenamiento) para no filtrar el futuro."""
    pares = {tuple(p) for c in cfg["seleccion_rutas"]["corredores"] for p in (c, c[::-1])}
    sel = rutas_raw[[(o, d) in pares for o, d in zip(rutas_raw.origen, rutas_raw.destino)]]
    sel = sel[(sel.tipo_despacho == cfg["limpieza"]["tipo_despacho_objetivo"])
              & (sel.fecha <= hasta)]
    g = sel.groupby(["origen", "destino", "clase_vehiculo"])[
        ["despachos", "pasajeros", "despachos_cero_pax"]].sum().reset_index()
    g["ruta_id"] = [id_ruta(a, b) for a, b in zip(g.origen, g.destino)]
    g["participacion_despachos"] = g.despachos / g.groupby("ruta_id").despachos.transform("sum")
    g["pax_por_despacho"] = g.pasajeros / g.despachos.replace(0, np.nan)
    g["pax_por_despacho_efectivo"] = g.pasajeros / (g.despachos - g.despachos_cero_pax).replace(0, np.nan)
    return g.sort_values(["ruta_id", "despachos"], ascending=[True, False])


# ---------------------------------------------------------------------------
# R6: atípicos
# ---------------------------------------------------------------------------
def z_robusto_estacional(y: pd.Series, semanas: int) -> tuple[pd.Series, pd.Series]:
    """z = (log1p(y) - mediana) / (1.4826 * MAD), con mediana y MAD calculadas sobre el
    mismo día de la semana en +-``semanas`` semanas (excluyendo el propio día)."""
    ly = np.log1p(y)
    vecinos = pd.concat({k: ly.shift(7 * k) for k in range(-semanas, semanas + 1) if k != 0}, axis=1)
    med = vecinos.median(axis=1)
    mad = vecinos.sub(med, axis=0).abs().median(axis=1) * 1.4826
    mad = mad.clip(lower=0.05)  # evita dividir por ~0 en series muy estables
    return (ly - med) / mad, np.expm1(med)


def cerca_de_evento(cal: pd.DataFrame, radio: int = 2) -> pd.Series:
    ev = (cal.festivo | cal.puente | cal.semana_santa | cal.temporada_fin_anio
          | cal.vispera_puente | cal.dia_despues_puente | cal.receso_octubre | cal.eleccion)
    return ev.rolling(2 * radio + 1, center=True, min_periods=1).max().astype(bool)


# ---------------------------------------------------------------------------
# R7: imputación de huecos cortos
# ---------------------------------------------------------------------------
def imputar_huecos(y: pd.Series, segmento: pd.Series, max_hueco: int) -> tuple[pd.Series, pd.Series]:
    out = y.copy()
    for s in segmento[segmento > 0].unique():
        m = segmento == s
        ys = y[m]
        falt = ys.isna()
        if not falt.any():
            continue
        # longitud de cada racha de faltantes
        racha = falt.ne(falt.shift()).cumsum()
        largo = falt.groupby(racha).transform("sum")
        imputables = falt & (largo <= max_hueco)
        # 1) promedio del mismo día de la semana anterior y siguiente;
        # 2) si falta alguno, mediana del mismo día de la semana en +-4 semanas;
        # 3) en último caso, interpolación lineal.
        est = pd.concat([ys.shift(7), ys.shift(-7)], axis=1).mean(axis=1, skipna=False)
        med4 = pd.concat([ys.shift(7 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)], axis=1).median(axis=1)
        lin = ys.interpolate(limit_area="inside")
        rel = est.fillna(med4).fillna(lin)
        out.loc[ys.index[imputables]] = rel[imputables].round()
    imputado = y.isna() & out.notna()
    return out, imputado


# ---------------------------------------------------------------------------
# Pipeline de limpieza por ruta
# ---------------------------------------------------------------------------
def limpiar(cfg: dict, ruta_dia: pd.DataFrame, valid: pd.DataFrame,
            cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lim = cfg["limpieza"]
    fechas = valid.fecha
    cal_i = cal.set_index("fecha").reindex(fechas)
    cerca = cerca_de_evento(cal_i.reset_index()).to_numpy()
    eventos = np.zeros(len(fechas), dtype=bool)
    for ev in lim.get("eventos_no_previsibles", []):
        eventos |= ((fechas >= ev["inicio"]) & (fechas <= ev["fin"])).to_numpy()

    partes, log = [], []
    for rid, g in ruta_dia.groupby("ruta_id"):
        g = g.set_index("fecha").reindex(fechas)          # R1: rango de fechas
        d = pd.DataFrame({"fecha": fechas.to_numpy()})
        d["ruta_id"] = rid
        d["origen"], d["destino"] = rid.split("-")
        for c in ["n_registros", "despachos", "pasajeros", "despachos_cero_pax",
                  "registros_cero_pax", "registros_sobrecupo", "pasajeros_transito",
                  "despachos_transito"]:
            d[c] = g[c].to_numpy()
        d["pasajeros_crudo"] = d.pasajeros
        d["dia_valido_sistema"] = valid.dia_valido_sistema.to_numpy()
        d["segmento"] = valid.segmento.to_numpy()
        d["evento_no_previsible"] = eventos

        y = d.pasajeros.astype(float)
        # R2
        y[~d.dia_valido_sistema] = np.nan
        # R5
        falta = d.dia_valido_sistema & (d.pasajeros_crudo.isna() | (d.pasajeros_crudo == 0))
        y[falta] = np.nan
        d["falta_reporte_ruta"] = falta
        # R6
        z, ref = z_robusto_estacional(y, lim["ventana_atipicos_semanas"])
        z = z.to_numpy()
        d["z_robusto"] = z
        # En los propios festivos (y 24/31-dic) las rutas pendulares caen >90 % de
        # verdad (1-ene, Viernes Santo): ahí no se aplica la regla de caída extrema.
        dia_festivo = (cal_i.festivo.to_numpy()
                       | fechas.dt.strftime("%m-%d").isin(["12-24", "12-31"]).to_numpy())
        caida = (y < lim["fraccion_caida_extrema"] * ref).to_numpy() & ~dia_festivo
        atip = (((np.abs(z) > lim["z_robusto_umbral"]) & ~cerca) | caida) & ~eventos
        d["atipico"] = atip
        y[atip] = np.nan
        d["pasajeros"] = y
        # R7
        d["pasajeros_imp"], d["imputado"] = imputar_huecos(
            y, d.segmento, lim["max_hueco_imputable_dias"])
        d["despachos_efectivos"] = d.despachos - d.despachos_cero_pax
        d["pax_por_despacho"] = d.pasajeros / d.despachos.replace(0, np.nan)
        partes.append(d)

        v = d.dia_valido_sistema
        log.append({
            "ruta_id": rid,
            "dias_periodo": len(d),
            "dias_invalidos_sistema": int((~v).sum()),
            "dias_falta_reporte_ruta": int(falta.sum()),
            "dias_atipicos": int(atip.sum()),
            "dias_imputados": int(d.imputado.sum()),
            "dias_nan_final": int(d.pasajeros_imp[v].isna().sum()),
            "pasajeros_validos": float(d.pasajeros[v].sum()),
            "pct_despachos_cero_pax": float(100 * d.despachos_cero_pax[v].sum() / d.despachos[v].sum()),
            "pct_registros_sobrecupo": float(100 * d.registros_sobrecupo[v].sum() / d.n_registros[v].sum()),
            "pct_pax_transito_sobre_origen": float(100 * d.pasajeros_transito[v].sum() / d.pasajeros_crudo[v].sum()),
        })
    return pd.concat(partes, ignore_index=True), pd.DataFrame(log)
