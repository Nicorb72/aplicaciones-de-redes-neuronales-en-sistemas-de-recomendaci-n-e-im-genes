"""Análisis descriptivo e hipótesis de demanda (Módulo 1).

Responsable: Persona 1.

Genera, a partir de los datos procesados (sin internet):
    reports/figures/m1/eda_*.png     figuras numerables para el reporte
    reports/tables/m1_*.csv          tablas de resumen e hipótesis

Uso
    python -m m1_demanda.eda [--config configs/m1_datos.yaml] [--ruta-descomposicion 50001-11001]

Método para medir efectos de calendario
---------------------------------------
Para cada día t de una ruta se define una LÍNEA BASE "normal":
    b[t] = mediana de y[t + 7k], k = +-1..+-4, tomando solo días SIN evento
           (sin festivo, puente, Semana Santa, fin de año, receso, elecciones, paro).
El índice relativo es  I[t] = y[t] / b[t].  I = 1.30 significa "30 % más que un
día normal del mismo día de la semana en el último/próximo mes". Esto controla la
estacionalidad semanal y la tendencia local, y aísla el efecto del evento.
Para cada hipótesis se prueba H0: mediana(log I) = 0 con la prueba de rangos con
signo de Wilcoxon (no paramétrica: no supone normalidad).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.stattools import acf

from m1_demanda.data.utils import cargar_config, ruta_abs

# Paleta categórica validada (slots 1-2) + neutros. Ver skill de visualización.
AZUL, NARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRIS, GRIS_CLARO = "#0b0b0b", "#52514e", "#a3a29c", "#e8e7e3"
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIS_CLARO, "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.2,
})


# ---------------------------------------------------------------------------
def cargar(cfg):
    proc = ruta_abs(cfg, "processed")
    p = pd.read_csv(proc / "series_diarias.csv.gz", parse_dates=["fecha"])
    rutas = pd.read_csv(proc / "rutas.csv", dtype=str)
    valid = pd.read_csv(proc / "validez_sistema.csv", parse_dates=["fecha"])
    return p, rutas, valid


def sombrear_huecos(ax, valid):
    """Sombrea en gris los periodos sin reporte del sistema (segmento 0)."""
    inval = valid.segmento == 0
    bloques = inval.ne(inval.shift()).cumsum()
    for _, g in valid[inval].groupby(bloques[inval]):
        ax.axvspan(g.fecha.min(), g.fecha.max(), color=GRIS_CLARO, lw=0, zorder=0)


def sombrear_test(ax, cfg):
    a, b = cfg["particiones"]["test"]
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=AZUL, alpha=0.07, lw=0, zorder=0)


def anotar_celdas(ax, valores, mostrar, centro, escala, fmt="{:.2f}", fontsize=7.5):
    """Escribe el valor en cada celda con tinta oscura o blanca según el fondo."""
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            v = valores[i, j]
            if np.isnan(v):
                continue
            oscuro = abs(v - centro) > 0.6 * escala
            ax.text(j, i, fmt.format(mostrar[i, j]), ha="center", va="center",
                    fontsize=fontsize, color="white" if oscuro else TINTA)


def guardar(fig, cfg, nombre):
    ruta = ruta_abs(cfg, "figuras") / nombre
    fig.savefig(ruta)
    plt.close(fig)
    print(f"[fig] {ruta.relative_to(ruta.parents[3])}")


# ---------------------------------------------------------------------------
# Índice relativo frente a línea base sin eventos
# ---------------------------------------------------------------------------
def columna_evento(p: pd.DataFrame) -> pd.Series:
    return (p.festivo | p.puente | p.semana_santa | p.temporada_fin_anio | p.receso_octubre
            | p.eleccion | p.vispera_puente | p.dia_despues_puente | p.evento_no_previsible)


def indice_relativo(p: pd.DataFrame) -> pd.Series:
    p = p.sort_values(["ruta_id", "fecha"])
    ev = columna_evento(p)
    out = []
    for _, g in p.groupby("ruta_id", sort=False):
        y = g.pasajeros
        y_normal = y.where(~ev.loc[g.index])
        vec = pd.concat([y_normal.shift(7 * k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)], axis=1)
        base = vec.median(axis=1).where(vec.notna().sum(axis=1) >= 3)
        out.append(y / base)
    return pd.concat(out).reindex(p.index)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_cobertura(valid, cfg):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(valid.fecha, valid.ratio_sistema, color=AZUL, lw=0.8)
    ax.axhline(cfg["limpieza"]["umbral_cobertura_sistema"], color=NARANJA, ls="--", lw=1)
    ax.text(valid.fecha.iloc[5], cfg["limpieza"]["umbral_cobertura_sistema"] + 0.04,
            f"umbral = {cfg['limpieza']['umbral_cobertura_sistema']:.2f}", color=TINTA2)
    sombrear_huecos(ax, valid)
    ax.set_ylabel("registros / mediana")
    ax.set_title("Registros diarios de todo el sistema relativos a la mediana (gris: periodo excluido)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    guardar(fig, cfg, "eda_01_cobertura_sistema.png")


def fig_series(p, rutas, valid, cfg):
    corredores = rutas.groupby("corredor", sort=True)
    fig, axs = plt.subplots(len(corredores), 1, figsize=(11, 2.2 * len(corredores)), sharex=True)
    for ax, (_, cr) in zip(axs, corredores):
        for color, (_, r) in zip([AZUL, NARANJA], cr.iterrows()):
            g = p[p.ruta_id == r.ruta_id].set_index("fecha").pasajeros_imp
            ax.plot(g.index, g, color=color, lw=0.35, alpha=0.45)
            ax.plot(g.index, g.rolling(7, center=True, min_periods=4).mean(), color=color,
                    lw=1.3, label=r.nombre)
        sombrear_huecos(ax, valid)
        sombrear_test(ax, cfg)
        ax.legend(loc="upper left", ncol=2)
        ax.set_ylabel("pasajeros/día")
        ax.set_ylim(bottom=0)
    axs[0].set_title("Pasajeros diarios por ruta (línea fina: diario; gruesa: media móvil 7 días; "
                     "azul claro: prueba; gris: sin datos)")
    axs[-1].xaxis.set_major_locator(mdates.YearLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    guardar(fig, cfg, "eda_02_series_rutas.png")


def fig_descomposicion(p, rutas, cfg, ruta_id):
    g = p[(p.ruta_id == ruta_id) & (p.segmento == 1)].set_index("fecha").pasajeros_imp
    g = g.interpolate(limit_direction="both")
    res = MSTL(np.log1p(g), periods=(7, 365), stl_kwargs={"seasonal_deg": 0}).fit()
    nombre = rutas.set_index("ruta_id").nombre[ruta_id]
    comps = [("log(1 + pasajeros)", res.observed), ("Tendencia", res.trend),
             ("Estacionalidad semanal", res.seasonal["seasonal_7"]),
             ("Estacionalidad anual", res.seasonal["seasonal_365"]), ("Residuo", res.resid)]
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    for ax, (t, s) in zip(axs, comps):
        ax.plot(s.index, s, color=AZUL, lw=0.7 if t != "Tendencia" else 1.5)
        ax.set_ylabel(t, fontsize=8)
    axs[0].set_title(f"Descomposición MSTL (periodos 7 y 365 días) — {nombre}, 2021–2023")
    guardar(fig, cfg, f"eda_03_descomposicion_{ruta_id}.png")
    # zoom de la estacionalidad semanal en 3 semanas
    var = {k: float(np.var(v)) for k, v in
           [("semanal", res.seasonal["seasonal_7"]), ("anual", res.seasonal["seasonal_365"]),
            ("tendencia", res.trend - res.trend.mean()), ("residuo", res.resid)]}
    tot = sum(var.values())
    return {k: round(100 * v / tot, 1) for k, v in var.items()}


def fig_dia_semana(p, rutas, cfg):
    q = p[(p.segmento > 0) & ~columna_evento(p)].copy()
    # índice por día de semana relativo a la media semanal centrada
    q["media28"] = q.groupby("ruta_id").pasajeros.transform(
        lambda s: s.rolling(28, center=True, min_periods=14).mean())
    q["rel"] = q.pasajeros / q.media28
    tab = q.groupby(["ruta_id", "dia_semana"]).rel.mean().unstack()
    tab.index = rutas.set_index("ruta_id").nombre.reindex(tab.index)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    vmax = float(np.nanmax(np.abs(tab.to_numpy() - 1)))
    im = ax.imshow(tab.to_numpy(), cmap="RdBu_r", vmin=1 - vmax, vmax=1 + vmax, aspect="auto")
    ax.set_xticks(range(7), DIAS)
    ax.set_yticks(range(len(tab)), tab.index)
    ax.grid(False)
    anotar_celdas(ax, tab.to_numpy(), tab.to_numpy(), 1, vmax)
    fig.colorbar(im, ax=ax, shrink=0.8, label="pasajeros / media móvil 28 días")
    ax.set_title("Perfil semanal: demanda relativa por día de la semana (días sin evento)")
    guardar(fig, cfg, "eda_04_perfil_dia_semana.png")
    return tab


def fig_mensual(p, rutas, cfg):
    q = p[(p.segmento > 0) & ~p.evento_no_previsible].copy()
    q["anio"] = q.fecha.dt.year
    m = q.groupby(["ruta_id", "anio", "mes"]).pasajeros.mean().reset_index()
    m["rel"] = m.pasajeros / m.groupby(["ruta_id", "anio"]).pasajeros.transform("mean")
    # solo años completos 2021-2023 para el perfil anual
    tab = m[m.anio.between(2022, 2023)].groupby(["ruta_id", "mes"]).rel.mean().unstack()
    tab.index = rutas.set_index("ruta_id").nombre.reindex(tab.index)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vmax = float(np.nanmax(np.abs(tab.to_numpy() - 1)))
    im = ax.imshow(tab.to_numpy(), cmap="RdBu_r", vmin=1 - vmax, vmax=1 + vmax, aspect="auto")
    ax.set_xticks(range(12), MESES)
    ax.set_yticks(range(len(tab)), tab.index)
    ax.grid(False)
    anotar_celdas(ax, tab.to_numpy(), tab.to_numpy(), 1, vmax, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="media mensual / media anual")
    ax.set_title("Estacionalidad anual: demanda media de cada mes relativa a la del año (2022–2023)")
    guardar(fig, cfg, "eda_05_estacionalidad_mensual.png")
    return tab


def fig_eventos(p, rutas, cfg):
    tipos = {
        "Víspera de puente": p.vispera_puente,
        "Primer día de puente": p.inicio_puente_flag,
        "Último día de puente": p.ultimo_dia_puente,
        "Festivo (no último)": p.festivo & ~p.ultimo_dia_puente,
        "Día después de puente": p.dia_despues_puente,
        "Elecciones": p.eleccion,
    }
    filas = []
    for t, m in tipos.items():
        for rid, g in p[m & p.indice.notna()].groupby("ruta_id"):
            filas.append({"evento": t, "ruta_id": rid, "mediana": g.indice.median(), "n": len(g)})
    tab = pd.DataFrame(filas).pivot(index="ruta_id", columns="evento", values="mediana")[list(tipos)]
    tab.index = rutas.set_index("ruta_id").nombre.reindex(tab.index)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    lv = np.log(tab.to_numpy().astype(float))
    vmax = float(np.nanmax(np.abs(lv)))
    im = ax.imshow(lv, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tipos)), list(tipos), rotation=20, ha="right")
    ax.set_yticks(range(len(tab)), tab.index)
    ax.grid(False)
    anotar_celdas(ax, lv, tab.to_numpy().astype(float), 0, vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log(índice relativo)")
    ax.set_title("Efecto de festivos, puentes y elecciones: mediana del índice relativo\n"
                 "(pasajeros / línea base del mismo día de la semana sin evento)")
    guardar(fig, cfg, "eda_06_efecto_festivos_puentes.png")
    return tab


def fig_semana_santa_fin_anio(p, rutas, cfg):
    from m1_demanda.data.calendar_features import domingo_pascua
    corr = rutas[rutas.corredor == "1"]  # Villavicencio <-> Bogotá
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for color, (_, r) in zip([AZUL, NARANJA], corr.iterrows()):
        g = p[p.ruta_id == r.ruta_id].set_index("fecha")
        filas = []
        for a in (2021, 2022, 2023, 2025):
            dp = pd.Timestamp(domingo_pascua(a))
            for k in range(-10, 4):
                f = dp + pd.Timedelta(days=k)
                filas.append({"k": k, "anio": a, "i": g.indice.get(f, np.nan)})
        d = pd.DataFrame(filas).groupby("k").i.median()
        axs[0].plot(d.index, d, color=color, marker="o", ms=3, label=r.nombre)
        filas = []
        for a in (2021, 2022, 2023):
            for k in range(-14, 21):
                f = pd.Timestamp(f"{a}-12-25") + pd.Timedelta(days=k)
                filas.append({"k": k, "i": g.indice.get(f, np.nan)})
        d = pd.DataFrame(filas).groupby("k").i.median()
        axs[1].plot(d.index, d, color=color, marker="o", ms=3, label=r.nombre)
    for ax in axs:
        ax.axhline(1, color=GRIS, lw=1)
        ax.legend(loc="upper left")
    axs[0].axvline(0, color=GRIS, ls=":")
    axs[0].set_xlabel("días respecto al Domingo de Pascua (−7 = Domingo de Ramos)")
    axs[0].set_ylabel("índice relativo (mediana entre años)")
    axs[0].set_title("Semana Santa (2021, 2022, 2023, 2025)")
    axs[1].axvline(0, color=GRIS, ls=":")
    axs[1].axvline(7, color=GRIS, ls=":")
    axs[1].set_xlabel("días respecto al 25 de diciembre (7 = 1 de enero)")
    axs[1].set_title("Navidad y fin de año (2021, 2022, 2023)")
    guardar(fig, cfg, "eda_07_semana_santa_fin_anio.png")


def fig_recuperacion(p, rutas, valid, cfg):
    q = p[p.segmento > 0].copy()
    q["mes_ini"] = q.fecha.dt.to_period("M").dt.to_timestamp()
    m = q.groupby(["ruta_id", "mes_ini"]).pasajeros.mean().reset_index()
    base = m[m.mes_ini.dt.year == 2023].groupby("ruta_id").pasajeros.mean()
    m["rel"] = m.pasajeros / m.ruta_id.map(base)
    tot = m.groupby("mes_ini").rel.median()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for rid, g in m.groupby("ruta_id"):
        ax.plot(g.mes_ini, g.rel, color=GRIS, lw=0.6, alpha=0.7)
    ax.plot(tot.index, tot, color=AZUL, lw=2, marker="o", ms=3, label="mediana de las 10 rutas")
    ax.axhline(1, color=TINTA2, lw=0.8, ls="--")
    sombrear_huecos(ax, valid)
    ev = cfg["limpieza"]["eventos_no_previsibles"][0]
    ax.axvspan(pd.Timestamp(ev["inicio"]), pd.Timestamp(ev["fin"]), color=NARANJA, alpha=0.15, lw=0)
    ax.text(pd.Timestamp(ev["inicio"]), 0.3, " Paro Nacional\n 2021", color=TINTA2, fontsize=8)
    ax.set_ylabel("media mensual / media 2023")
    ax.set_title("Recuperación pospandemia: demanda mensual relativa al promedio de 2023 (gris: cada ruta)")
    ax.legend(loc="lower right")
    guardar(fig, cfg, "eda_08_recuperacion_pospandemia.png")
    anual = q[q.fecha.dt.month <= 11].groupby([q.fecha.dt.year, "ruta_id"]).pasajeros.mean().unstack(0)
    return (anual.div(anual[2023], axis=0)).round(3)


def fig_ida_regreso(p, rutas, cfg):
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.9))
    filas = []
    for ax, (c, cr) in zip(axs, rutas.groupby("corredor")):
        ida, reg = cr.ruta_id.tolist()
        a = p[p.ruta_id == ida].set_index("fecha")
        b = p[p.ruta_id == reg].set_index("fecha")
        semana = pd.DataFrame({"ida": a.pasajeros, "reg": b.pasajeros}).dropna()
        semana = semana.resample("W").sum(min_count=7).dropna()
        ax.scatter(semana.ida, semana.reg, s=8, color=AZUL, alpha=0.6, lw=0)
        lim = [0, semana.max().max() * 1.05]
        ax.plot(lim, lim, color=GRIS, ls="--", lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(cr.nombre.iloc[0].replace(" → ", " ↔ "), fontsize=8.5)
        ax.set_xlabel("ida (pax/semana)")
        r = np.corrcoef(semana.ida, semana.reg)[0, 1]
        ax.text(0.04, 0.9, f"r = {r:.2f}\nreg/ida = {semana.reg.sum() / semana.ida.sum():.2f}",
                transform=ax.transAxes, fontsize=8, color=TINTA2)
        filas.append({"corredor": cr.nombre.iloc[0], "ruta_ida": ida, "ruta_regreso": reg,
                      "correlacion_semanal": round(r, 3),
                      "razon_regreso_ida": round(semana.reg.sum() / semana.ida.sum(), 3)})
    axs[0].set_ylabel("regreso (pax/semana)")
    fig.suptitle("Ida vs. regreso: pasajeros semanales en origen (línea: igualdad)", fontweight="bold")
    guardar(fig, cfg, "eda_09_ida_vs_regreso.png")
    return pd.DataFrame(filas)


def fig_calidad(p, rutas, cfg):
    q = p[p.dia_valido_sistema].copy()
    g = q.groupby("ruta_id").agg(desp=("despachos", "sum"), desp0=("despachos_cero_pax", "sum"),
                                 pax=("pasajeros_crudo", "sum"), tr=("pasajeros_transito", "sum"))
    g["pct0"] = 100 * g.desp0 / g.desp
    g["pct_tr"] = 100 * g.tr / g.pax
    g.index = rutas.set_index("ruta_id").nombre.reindex(g.index)
    g = g.sort_values("pct0")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    axs[0].barh(g.index, g.pct0, color=AZUL, height=0.6)
    axs[0].set_xlabel("% de despachos con 0 pasajeros")
    axs[0].set_title("Despachos que salen sin pasajeros registrados")
    axs[1].barh(g.index, g.pct_tr, color=AZUL, height=0.6)
    axs[1].set_xlabel("pasajeros en terminales de tránsito / pasajeros en origen (%)")
    axs[1].set_title("Pasajeros que suben en terminales de tránsito")
    for ax, col in zip(axs, ["pct0", "pct_tr"]):
        for i, v in enumerate(g[col]):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=7.5, color=TINTA2)
        ax.grid(axis="y", visible=False)
    guardar(fig, cfg, "eda_10_calidad_despachos_transito.png")


def fig_acf(p, rutas, cfg, ruta_id):
    g = p[(p.ruta_id == ruta_id) & (p.particion == "train")].pasajeros_imp.interpolate()
    a = acf(np.log1p(g), nlags=400, missing="drop")
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.vlines(range(len(a)), 0, a, color=AZUL, lw=0.8)
    ax.axhline(1.96 / np.sqrt(len(g)), color=NARANJA, ls="--", lw=0.8)
    ax.axhline(-1.96 / np.sqrt(len(g)), color=NARANJA, ls="--", lw=0.8)
    for k in (7, 14, 28, 364):
        ax.annotate(f"{k}", (k, a[k]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5, color=TINTA2)
    nombre = rutas.set_index("ruta_id").nombre[ruta_id]
    ax.set_title(f"Autocorrelación de log(1+pasajeros) — {nombre} (entrenamiento)")
    ax.set_xlabel("rezago (días)")
    guardar(fig, cfg, f"eda_11_acf_{ruta_id}.png")


# ---------------------------------------------------------------------------
# Hipótesis
# ---------------------------------------------------------------------------
def probar_hipotesis(p, tab_dow, tab_mes, anual, rutas):
    """Devuelve una tabla con estadístico, p-valor y veredicto por hipótesis."""
    q = p[p.segmento > 0]
    filas = []

    def wilcoxon(mascara, nombre, hipotesis, esperado_mayor=True):
        x = np.log(q.loc[mascara & q.indice.notna(), "indice"])
        if len(x) < 5:
            return
        alt = "greater" if esperado_mayor else "less"
        res = stats.wilcoxon(x, alternative=alt)
        med = float(np.exp(np.median(x)))
        por_ruta = q[mascara & q.indice.notna()].groupby("ruta_id").indice.median()
        filas.append({
            "id": nombre, "hipotesis": hipotesis, "n_dias_ruta": len(x),
            "indice_mediano": round(med, 3),
            "rutas_en_direccion_esperada": int(((por_ruta > 1) if esperado_mayor else (por_ruta < 1)).sum()),
            "estadistico": float(res.statistic), "p_valor": float(res.pvalue),
            "veredicto": "Confirmada" if res.pvalue < 0.01 else "No confirmada",
        })

    # H1 día de la semana: Kruskal-Wallis del índice semanal por ruta
    qq = q[~columna_evento(q)].copy()
    qq["rel"] = qq.pasajeros / qq.groupby("ruta_id").pasajeros.transform(
        lambda s: s.rolling(28, center=True, min_periods=14).mean())
    pv = []
    for rid, g in qq.dropna(subset=["rel"]).groupby("ruta_id"):
        pv.append(stats.kruskal(*[gg.rel for _, gg in g.groupby("dia_semana")]).pvalue)
    amp = (tab_dow.max(axis=1) / tab_dow.min(axis=1))
    filas.append({"id": "H1", "hipotesis": "El día de la semana determina la demanda, con patrón según el sentido: salidas de ciudad grande pican viernes, regresos domingo/lunes, rutas pendulares caen el fin de semana",
                  "n_dias_ruta": len(qq), "indice_mediano": round(float(amp.median()), 3),
                  "rutas_en_direccion_esperada": int((np.array(pv) < 0.01).sum()),
                  "estadistico": np.nan, "p_valor": float(max(pv)),
                  "veredicto": "Confirmada" if max(pv) < 0.01 else "Parcial",
                  })
    wilcoxon(q.vispera_puente, "H2a", "La víspera de un puente festivo aumenta la demanda")
    wilcoxon(q.ultimo_dia_puente, "H2b", "El último día de un puente (retorno) aumenta la demanda")
    wilcoxon(q.festivo & ~q.ultimo_dia_puente, "H2c", "Los festivos que no son día de retorno reducen la demanda",
             esperado_mayor=False)
    wilcoxon(q.semana_santa, "H3", "La Semana Santa aumenta la demanda")
    wilcoxon(q.temporada_fin_anio & ~q.festivo, "H4", "La temporada de fin de año (15-dic a 12-ene) aumenta la demanda")
    wilcoxon(q.vacaciones_mitad_anio & ~q.festivo & ~q.puente, "H5a",
             "Las vacaciones escolares de mitad de año aumentan la demanda")
    wilcoxon(q.receso_octubre & ~q.puente, "H5b", "El receso escolar de octubre aumenta la demanda")
    wilcoxon(q.eleccion, "H7", "Las jornadas electorales reducen la demanda", esperado_mayor=False)
    # H6 recuperación pospandemia: nivel 2021 < 2022 <= 2023 ; 2025 vs 2023
    r21, r22, r25 = anual[2021], anual[2022], anual[2025]
    filas.append({"id": "H6", "hipotesis": "Recuperación pospandemia: la demanda crece de 2021 a 2023 y se estabiliza",
                  "n_dias_ruta": np.nan,
                  "indice_mediano": round(float(r21.median()), 3),
                  "rutas_en_direccion_esperada": int((r21 < 1).sum()),
                  "estadistico": np.nan,
                  "p_valor": float(stats.wilcoxon(np.log(r21), alternative="less").pvalue),
                  "veredicto": "Confirmada" if (r21 < 1).mean() >= 0.8 else "Parcial"})
    t = pd.DataFrame(filas)
    t["p_valor"] = t.p_valor.map(lambda v: f"{v:.2e}")
    extra = {"nivel_2022_vs_2023_mediana": round(float(r22.median()), 3),
             "nivel_2025_vs_2023_mediana": round(float(r25.median()), 3)}
    return t, extra


# ---------------------------------------------------------------------------
def tablas_resumen(p, rutas, cfg):
    raw = ruta_abs(cfg, "raw")
    anual = pd.read_csv(raw / "resumen_anual.csv")
    meta = json.loads((raw / "metadata_eh75-8ah6.json").read_text(encoding="utf-8"))
    dicc = pd.DataFrame([{"variable": c["fieldName"], "tipo_publicado": c["dataTypeName"],
                          "descripcion_diccionario": c.get("description", "")}
                         for c in meta["columns"] if not c["fieldName"].startswith(":")])
    tablas = ruta_abs(cfg, "tablas")
    dicc.to_csv(tablas / "m1_diccionario_variables.csv", index=False)
    anual.to_csv(tablas / "m1_registros_por_anio.csv", index=False)

    q = p[p.segmento > 0]
    est = q.groupby(["ruta_id", "particion"]).pasajeros.agg(
        ["count", "mean", "std", "min", "median", "max"]).round(1).reset_index()
    est = est.merge(rutas[["ruta_id", "nombre"]], on="ruta_id")
    est.to_csv(tablas / "m1_estadisticas_rutas.csv", index=False)
    return {"registros_totales": int(anual.n_registros.sum()),
            "variables": len(dicc),
            "fecha_min_registrada": str(anual.anio.min()), "fecha_max_registrada": str(anual.anio.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="EDA del Módulo 1")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ruta-descomposicion", default="50001-11001")
    args = ap.parse_args(argv)
    cfg = cargar_config(args.config)

    p, rutas, valid = cargar(cfg)
    p = p.sort_values(["ruta_id", "fecha"]).reset_index(drop=True)
    p["inicio_puente_flag"] = p.puente & ~p.groupby("ruta_id").puente.shift(1, fill_value=False)
    p["indice"] = indice_relativo(p)

    resumen = tablas_resumen(p, rutas, cfg)
    fig_cobertura(valid, cfg)
    fig_series(p, rutas, valid, cfg)
    var_desc = fig_descomposicion(p, rutas, cfg, args.ruta_descomposicion)
    tab_dow = fig_dia_semana(p, rutas, cfg)
    tab_mes = fig_mensual(p, rutas, cfg)
    tab_ev = fig_eventos(p, rutas, cfg)
    fig_semana_santa_fin_anio(p, rutas, cfg)
    anual = fig_recuperacion(p, rutas, valid, cfg)
    ida_reg = fig_ida_regreso(p, rutas, cfg)
    fig_calidad(p, rutas, cfg)
    fig_acf(p, rutas, cfg, args.ruta_descomposicion)

    hip, extra = probar_hipotesis(p, tab_dow, tab_mes, anual, rutas)
    tablas = ruta_abs(cfg, "tablas")
    hip.to_csv(tablas / "m1_hipotesis.csv", index=False)
    tab_dow.round(3).to_csv(tablas / "m1_indice_dia_semana.csv")
    tab_mes.round(3).to_csv(tablas / "m1_indice_mensual.csv")
    tab_ev.round(3).to_csv(tablas / "m1_indice_eventos.csv")
    anual.to_csv(tablas / "m1_nivel_anual_vs_2023.csv")
    ida_reg.to_csv(tablas / "m1_ida_vs_regreso.csv", index=False)

    resumen.update({"varianza_descomposicion_pct": var_desc, **extra})
    (tablas / "m1_resumen_eda.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    pd.set_option("display.width", 220, "display.max_colwidth", 70)
    print(hip.to_string(index=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
