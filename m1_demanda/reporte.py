"""Propósito: Figuras y tablas del reporte del Módulo 1 que exige el enunciado (Persona 2).
Responsable: Persona 2.

Genera, con los modelos ya entrenados (``models/m1_demanda/<familia>_final.*``) y los baselines:

* ``m1_pronostico_rutas_<origen>.png``  pronóstico frente a la demanda real de LAS 10 rutas, con la banda del P90,
  para un origen de prueba en temporada alta (receso de octubre y puentes; fin de año).
* ``m1_flota_conductores_dia.png``      vehículos y conductores-día requeridos por día en los próximos 30 días,
  por corredor, planeando con la media y con el P90.
* ``m1_sensibilidad_flota.png``         flota frente a la demanda (±10 % y ±20 %) y frente al parámetro operativo más
  influyente; ``m1_sensibilidad_flota.csv``.
* ``m1_error_tramos.csv``               métricas por tramo de horizonte (días 1–7, 8–14 y 15–30) de cada modelo.
* ``m1_cajas_semillas.png``             MASE de las 30 semillas de cada red frente a los baselines.
* ``m1_busqueda_resultados.png``        desempeño en validación de las configuraciones probadas.
* ``m1_curvas_aprendizaje_<familia>.png`` pérdida de entrenamiento y validación por época, con la parada temprana.
* ``m1_arquitectura_<familia>.png``     capas, dimensiones y parámetros entrenables de la red final.

Uso: ``python -m m1_demanda.reporte`` (cada figura se omite con un aviso si falta el modelo que necesita).
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common.config_loader import cargar_yaml
from m1_demanda import figuras
from m1_demanda.baselines import GLMPoisson, IngenuoEstacional
from m1_demanda.data.utils import RAIZ, cargar_config, ruta_abs
from m1_demanda.evaluate import escala_mase, evaluar_modelo, por_tramos
from m1_demanda.operacion import forecast_value as fv
from m1_demanda.operacion import parametros as pa
from m1_demanda.operacion.planificar import planificar
from m1_demanda.operacion.sensitivity import orden_por_efecto, sensibilidad
from m1_demanda.train import MODELOS, cargar_modelo, cfg_con_ventana
from m1_demanda.ventanas import CONFIG_M1, Panel, cargar_panel, construir_ventanas, ventanas_particion

ORIGENES_TEMPORADA_ALTA = {
    "2023-10-15": "receso de octubre y puentes de octubre–noviembre",
    "2023-12-02": "fin de año (Navidad y Año Nuevo)",
}
AZUL, NARANJA, TINTA, GRIS = figuras.AZUL, figuras.NARANJA, figuras.TINTA, figuras.GRIS


def cargar_modelos(panel: Panel, cfg_m1: dict, cfg_datos: dict) -> dict:
    """Ingenuo, GLM y las redes finales que existan en ``models/m1_demanda/``."""
    out = {"ingenuo": IngenuoEstacional(0.9).ajustar(panel, cfg_m1, cfg_datos),
           "glm": GLMPoisson(0.9).ajustar(panel, cfg_m1, cfg_datos)}
    for fam in ("mlp", "rnn"):
        base = MODELOS / f"{fam}_final"
        if base.with_suffix(".pt").exists():
            out[fam] = cargar_modelo(base, n_rutas=len(panel.rutas))
    return out


def _ventanas_para(modelo, panel: Panel, cfg_m1: dict, origenes) -> "object":
    cfg = cfg_con_ventana(cfg_m1, int(modelo.hp["window"])) if hasattr(modelo, "hp") else cfg_m1
    return construir_ventanas(panel, cfg, list(origenes))


# ---------------------------------------------------------------------------
# Pronóstico vs real (las 10 rutas, con banda del P90)
# ---------------------------------------------------------------------------
def fig_pronostico_rutas(panel: Panel, cfg_m1: dict, modelos: dict, origen: str, descripcion: str):
    """5×2: una celda por ruta (ida arriba, regreso abajo en cada corredor). Banda = media a P90 de la red final."""
    H = int(cfg_m1["horizon"])
    i0 = panel.idx(origen)
    fechas = panel.fechas[i0:i0 + H]
    red = next((m for m in ("rnn", "mlp") if m in modelos), None)
    pron: dict[str, dict[int, tuple]] = {}
    for nombre, m in modelos.items():
        v = _ventanas_para(m, panel, cfg_m1, [origen])
        mu, q = m.predecir(panel, v)
        pron[nombre] = {int(r): (mu[k], q[k]) for k, r in enumerate(v.ruta_idx)}
    fig, ejes = plt.subplots(5, 2, figsize=(12.5, 15), sharex=True)
    alta = panel.alta[i0:i0 + H]
    for corredor in range(5):
        for col, sentido in enumerate(("ida", "regreso")):
            ax = ejes[corredor, col]
            fila = panel.rutas[(panel.rutas.corredor == corredor + 1) & (panel.rutas.sentido == sentido)].iloc[0]
            r = int(fila.name)
            ax.fill_between(fechas, 0, 1, where=alta, transform=ax.get_xaxis_transform(), color=figuras.GRIS_CLARO,
                            lw=0, zorder=0)
            obs = panel.y_obs[i0:i0 + H, r].copy()
            obs[panel.imputado[i0:i0 + H, r]] = np.nan
            ax.plot(fechas, obs, color=TINTA, lw=1.8, label="observado", zorder=5)
            for nombre, c in (("ingenuo", figuras.GRIS), ("glm", figuras.TINTA2)):
                if nombre in pron and r in pron[nombre]:
                    ax.plot(fechas, pron[nombre][r][0], color=c, lw=1.0, ls="--" if nombre == "ingenuo" else "-",
                            label=nombre)
            if red and r in pron[red]:
                mu, q = pron[red][r]
                ax.plot(fechas, mu, color=AZUL, lw=1.6, label=f"{red} (media)")
                ax.fill_between(fechas, mu, q, color=AZUL, alpha=0.18, lw=0, label=f"{red}: banda media–P90")
            ax.set_title(fila["nombre"], fontsize=9)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            if col == 0:
                ax.set_ylabel("Pasajeros/día", fontsize=8)
    h, l = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"Pronóstico a 30 días desde {origen} — {descripcion} (gris = temporada alta)", y=1.0 + 0.0,
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


# ---------------------------------------------------------------------------
# Flota y conductores por día
# ---------------------------------------------------------------------------
def tabla_flota_dia(modelo_nombre: str | None = None) -> pd.DataFrame:
    """Vehículos y conductores-día por corredor y día en los próximos 30 días, plan con media y con P90."""
    from m1_demanda import predict

    filas = []
    for q, etiqueta in ((None, "media"), (0.9, "P90")):
        for r in predict.list_routes().query("sentido == 'ida'").itertuples():
            d = predict.forecast(r.ruta_id, q)
            filas.append(pd.DataFrame({"fecha": d["fecha"], "corredor": r.corredor, "plan": etiqueta,
                                       "vehiculos": d["vehiculos"], "conductores_dia": d["conductores"],
                                       "modelo": d.attrs["modelo"], "flota": d.attrs["flota_corredor"],
                                       "plantilla": d.attrs["plantilla_conductores"]}))
    return pd.concat(filas, ignore_index=True)


def fig_flota_conductores(tabla: pd.DataFrame, nombres: dict[int, str]):
    corredores = sorted(tabla["corredor"].unique())
    fig, ejes = plt.subplots(2, len(corredores), figsize=(15.5, 6.0), sharex="col")
    for j, c in enumerate(corredores):
        for fila, (col, titulo) in enumerate((("vehiculos", "Vehículos"), ("conductores_dia", "Conductores-día"))):
            ax = ejes[fila, j]
            for plan, color in (("media", figuras.GRIS), ("P90", AZUL)):
                d = tabla[(tabla.corredor == c) & (tabla.plan == plan)]
                ax.step(d["fecha"], d[col], where="mid", color=color, lw=1.6, label=f"plan con {plan}")
            f = tabla[(tabla.corredor == c) & (tabla.plan == "P90")].iloc[0]
            # El total del plan con P90 va en el título (fuera del área de trazado, para no tapar las series).
            if fila == 0:
                ax.set_title(f"{nombres[c]}\nflota (P90): {int(f['flota'])} vehículos", fontsize=9)
            else:
                ax.set_title(f"plantilla (P90): {int(f['plantilla'])} conductores", fontsize=8.5, fontweight="normal")
            if j == 0:
                ax.set_ylabel(titulo)
            ax.tick_params(axis="x", labelsize=7, rotation=35)
    h, l = ejes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Vehículos y conductores-día requeridos por día en los próximos 30 días (origen de producción)",
                 fontweight="bold", fontsize=10, x=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Sensibilidad de la flota
# ---------------------------------------------------------------------------
def curva_sensibilidad(panel: Panel, v, plan: np.ndarray, params: dict, mezcla: pd.DataFrame,
                       factores=(0.8, 0.9, 1.0, 1.1, 1.2)) -> pd.DataFrame:
    """Flota total frente al factor sobre la demanda pronosticada y sobre ``target_occupancy`` y ``capacity_by_class``."""
    filas = []
    base = fv.simular(panel, v, {"p": plan}, params, mezcla)["flota_total"].mean()
    for nombre in ("demanda_pronosticada", "target_occupancy", "capacity_by_class", "cycle_time", "operating_hours"):
        for f in factores:
            if nombre == "demanda_pronosticada":
                s = fv.simular(panel, v, {"p": plan * f}, params, mezcla)
            else:
                s = fv.simular(panel, v, {"p": plan}, pa.escalar_parametro(params, nombre, f), mezcla)
            flota = float(s["flota_total"].mean())
            filas.append({"parametro": nombre, "factor": f, "flota_total": flota, "cambio_pct": 100 * (flota / base - 1)})
    return pd.DataFrame(filas)


def fig_sensibilidad_flota(curvas: pd.DataFrame, plan_nombre: str):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    d = curvas[curvas.parametro == "demanda_pronosticada"].sort_values("factor")
    a1.plot((d.factor - 1) * 100, d.cambio_pct, marker="o", color=AZUL, lw=2)
    a1.axhline(0, color=GRIS, lw=0.8)
    a1.axvline(0, color=GRIS, lw=0.8)
    a1.set_xlabel("Variación de la demanda frente al pronóstico (%)")
    a1.set_ylabel("Cambio de la flota total (%)")
    a1.set_title("Si la demanda real difiere del pronóstico")
    for _, r in d.iterrows():
        a1.annotate(f"{r.cambio_pct:+.1f}", ((r.factor - 1) * 100, r.cambio_pct), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5)
    # target_occupancy y capacity_by_class dan EXACTAMENTE la misma curva: los despachos dependen solo del producto
    # φ·C. Se dibuja una gruesa y la otra encima, discontinua, para que se vean las dos.
    estilos = {
        "capacity_by_class": dict(color=figuras.VERDE_AGUA, lw=5.5, alpha=0.55, marker="o", ms=8,
                                  label="capacity_by_class"),
        "target_occupancy": dict(color=NARANJA, lw=1.6, ls="--", marker="s", ms=4.5,
                                 label="target_occupancy (idéntica a capacity_by_class)"),
        "cycle_time": dict(color=figuras.TINTA2, lw=1.6, marker="o", ms=5, label="cycle_time"),
        "operating_hours": dict(color=figuras.GRIS, lw=1.6, marker="o", ms=5, label="operating_hours"),
    }
    for p, kw in estilos.items():
        g = curvas[curvas.parametro == p].sort_values("factor")
        a2.plot((g.factor - 1) * 100, g.cambio_pct, **kw)
    a2.axhline(0, color=GRIS, lw=0.8)
    a2.axvline(0, color=GRIS, lw=0.8)
    a2.set_xlabel("Variación del parámetro operativo (%)")
    a2.set_title("Parámetros operativos (la flota depende de φ·C, de T y de H_op)")
    a2.legend(fontsize=7.5, loc="upper center")
    fig.suptitle(f"Sensibilidad de la flota total — plan {plan_nombre}", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Error por tramos de horizonte
# ---------------------------------------------------------------------------
def tabla_tramos(panel: Panel, cfg_m1: dict, cfg_datos: dict, modelos: dict, particion: str = "test") -> pd.DataFrame:
    esc = escala_mase(panel, cfg_datos)
    ids = list(panel.rutas["ruta_id"])
    filas = []
    for nombre, m in modelos.items():
        cfg = cfg_con_ventana(cfg_m1, int(m.hp["window"])) if hasattr(m, "hp") else cfg_m1
        _, _, v, mu, q = evaluar_modelo(m, panel, cfg, cfg_datos, particion)
        t = por_tramos(v, mu, q, esc, ids, tau=float(cfg_m1["quantile"]))
        t.insert(0, "particion", particion)
        t.insert(0, "modelo", nombre)
        filas.append(t)
    return pd.concat(filas, ignore_index=True)


# ---------------------------------------------------------------------------
# Figuras comunes: cajas de las 30 semillas y resultados de la búsqueda
# ---------------------------------------------------------------------------
def fig_cajas_semillas(semillas: pd.DataFrame, baselines: dict[str, dict[str, float]], metrica: str = "mase"):
    """Diagramas de caja de ``metrica`` entre semillas, por configuración final y partición (validación y prueba)."""
    t = semillas[semillas.subconjunto == "total"]
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, part in zip(ejes, ("val", "test")):
        grupos = [(f"{fam} #{r}", g[metrica].to_numpy()) for (fam, r), g in t[t.particion == part].groupby(["familia", "config_rank"])]
        bp = ax.boxplot([g for _, g in grupos], tick_labels=[n for n, _ in grupos], patch_artist=True, widths=0.55)
        for patch, (n, _) in zip(bp["boxes"], grupos):
            patch.set_facecolor(AZUL if n.startswith("mlp") else NARANJA)
            patch.set_alpha(0.55)
        for nombre, estilo in (("ingenuo", ":"), ("glm", "--")):
            if nombre in baselines and part in baselines[nombre]:
                ax.axhline(baselines[nombre][part], color=figuras.TINTA2 if nombre == "glm" else GRIS, ls=estilo, lw=1.4,
                           label=nombre)
        ax.set_title(f"MASE en {part} (30 semillas por configuración)")
        ax.set_ylabel("MASE")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_busqueda(tablas: dict[str, pd.DataFrame], baselines_val: dict[str, float]):
    """MASE de validación de cada configuración probada, ordenado, por familia; líneas de los baselines."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for fam, color in (("mlp", AZUL), ("rnn", NARANJA)):
        if fam in tablas:
            d = tablas[fam].sort_values("mase").reset_index(drop=True)
            ax.plot(np.arange(1, len(d) + 1), d["mase"], marker="o", color=color, lw=1.4,
                    label=f"{fam} ({len(d)} configuraciones)")
    for nombre, estilo in (("ingenuo", ":"), ("glm", "--")):
        if nombre in baselines_val:
            ax.axhline(baselines_val[nombre], color=figuras.TINTA2 if nombre == "glm" else GRIS, ls=estilo, lw=1.4,
                       label=f"{nombre} (1 evaluación)" if nombre == "glm" else "ingenuo (0 hiperparámetros)")
    ax.set_xlabel("Configuración (ordenadas de mejor a peor)")
    ax.set_ylabel("MASE en validación (1 semilla)")
    ax.set_title("Búsqueda aleatoria de hiperparámetros: desempeño en validación")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Arquitectura (generada desde la red real) y curvas de aprendizaje
# ---------------------------------------------------------------------------
def tabla_arquitectura(pron, ejemplo) -> pd.DataFrame:
    """Capas de primer nivel de la red con la forma de su salida y sus parámetros entrenables (vía hooks)."""
    modelo, filas, hooks = pron.modelo.eval(), [], []

    def forma_con_n(t) -> str:
        """Forma con la dimensión de lote reemplazada por N (el tamaño del lote de ejemplo no es de la red)."""
        return "(" + ", ".join(["N"] + [str(d) for d in t.shape[1:]]) + ")" if hasattr(t, "shape") else "—"

    def gancho(nombre):
        def f(mod, entrada, salida):
            s = salida[0] if isinstance(salida, tuple) else salida
            filas.append({"capa": nombre, "tipo": type(mod).__name__, "salida": forma_con_n(s),
                          "parametros": int(sum(p.numel() for p in mod.parameters() if p.requires_grad))})
        return f

    for nombre, mod in modelo.named_modules():
        if nombre and len(list(mod.children())) == 0:
            hooks.append(mod.register_forward_hook(gancho(nombre)))
    with torch.no_grad():
        modelo(*ejemplo)
    for h in hooks:
        h.remove()
    # Entradas de la red (las dimensiones de entrada que pide el enunciado), antes de las capas.
    usa_cal_hist = bool(pron.hp.get("use_cal_hist", True)) if pron.familia == "mlp" else True
    nombres_in = ["x_hist", "cal_hist", "cal_fut", "ruta_idx"]
    entradas = []
    for n_, t in zip(nombres_in, ejemplo):
        nota = " (no se usa)" if (n_ == "cal_hist" and not usa_cal_hist) else ""
        entradas.append({"capa": f"entrada: {n_}{nota}", "tipo": "tensor", "salida": forma_con_n(t), "parametros": 0})
    return pd.DataFrame(entradas + filas)


def fig_arquitectura(tabla: pd.DataFrame, familia: str, descripcion: str, total: int):
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(tabla) + 1.9))
    ax.axis("off")
    filas = [[r.capa, r.tipo, r.salida, f"{r.parametros:,}"] for r in tabla.itertuples()]
    filas.append(["TOTAL entrenables", "", "", f"{total:,}"])
    t = ax.table(cellText=filas, colLabels=["Capa", "Tipo", "Forma de salida (N = lote)", "Parámetros"], loc="center",
                 cellLoc="left", colLoc="left", colWidths=[0.30, 0.20, 0.32, 0.18])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.35)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(figuras.GRIS_CLARO)
        if i == 0:
            c.set_facecolor(figuras.GRIS_CLARO)
            c.set_text_props(fontweight="bold")
        if i == len(filas):
            c.set_text_props(fontweight="bold")
    ax.set_title(f"Arquitectura de la red final ({familia}): {descripcion}", fontsize=9.5, fontweight="bold", loc="left")
    fig.tight_layout()
    return fig


def fig_curvas_aprendizaje(historial: list[dict], mejor_epoca: int, familia: str):
    h = pd.DataFrame(historial)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(h["epoca"], h["train"], color=AZUL, label="entrenamiento")
    ax.plot(h["epoca"], h["val_total"], color=NARANJA, label="validación")
    ax.axvline(mejor_epoca, color=TINTA, ls="--", lw=1.0)
    # El texto va a la IZQUIERDA de la línea (hay espacio libre sobre la curva de entrenamiento), no fuera del gráfico.
    ax.annotate(f"parada temprana: se conserva\nla época {mejor_epoca}", (mejor_epoca, h["val_total"].min()),
                textcoords="offset points", xytext=(-150, 38), fontsize=8, arrowprops={"arrowstyle": "->", "color": TINTA})
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida (Poisson + pinball)")
    ax.set_title(f"Curvas de aprendizaje de la red final ({familia})")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reproducir la semilla guardada: historial por época y verificación bit a bit
# ---------------------------------------------------------------------------
def reentrenar_semilla_guardada(familia: str, panel: Panel, cfg_m1: dict, cfg_datos: dict):
    """Re-entrena la configuración y la semilla de ``<familia>_final`` y devuelve ``(resultado, coincide)``.

    ``coincide`` es True si los pesos re-entrenados son idénticos a los guardados (en CPU el entrenamiento es
    determinista). Sirve para obtener el historial por época (curvas de aprendizaje) sin guardarlo para cada semilla.
    """
    import json

    from m1_demanda.train import entrenar

    sel = json.loads((MODELOS / f"{familia}_final_seleccion.json").read_text(encoding="utf-8"))
    res = entrenar(familia, sel["hp"], cfg_m1, cfg_datos, panel, int(sel["semilla_guardada"]), "cpu")
    guardado = cargar_modelo(MODELOS / f"{familia}_final", n_rutas=len(panel.rutas))
    igual = all(torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in
                zip(res.pronosticador.modelo.state_dict().values(), guardado.modelo.state_dict().values()))
    return res, bool(igual), sel


def _descripcion_arquitectura(pron) -> str:
    hp = pron.hp
    if pron.familia == "mlp":
        return (f"MLP, ventana {hp['window']}, {hp['layers']} capas de {hp['units']} unidades (ReLU, dropout "
                f"{hp['regularization']['dropout']:.2f}), embedding de ruta de {hp['route_embedding_dim']}")
    r = hp["rnn"]
    return (f"{r['cell'].upper()} de {r['layers']} capa(s) con {r['hidden']} unidades ocultas, ventana {hp['window']}, "
            f"capa densa de {hp['units']} (ReLU), embedding de ruta de {hp['route_embedding_dim']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Figuras y tablas del reporte del Módulo 1.")
    ap.add_argument("--config", default=CONFIG_M1)
    ap.add_argument("--sin-reentrenar", action="store_true",
                    help="omite las curvas de aprendizaje (que reentrenan la semilla guardada)")
    args = ap.parse_args(argv)
    cfg_m1, cfg_datos = cargar_yaml(args.config), cargar_config()
    panel = cargar_panel(cfg_m1, cfg_datos)
    tablas = ruta_abs(cfg_datos, "tablas")
    modelos = cargar_modelos(panel, cfg_m1, cfg_datos)
    print("Modelos disponibles:", ", ".join(modelos))

    # 1. Error por tramos de horizonte (días 1–7, 8–14 y 15–30)
    t = tabla_tramos(panel, cfg_m1, cfg_datos, modelos)
    t.to_csv(tablas / "m1_error_tramos.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 200):
        print(t[["modelo", "tramo", "mae", "mase", "cobertura"]].round(3).to_string(index=False))

    # 2. Pronóstico vs observado, 10 rutas, temporada alta
    for origen, desc in ORIGENES_TEMPORADA_ALTA.items():
        figuras.guardar(fig_pronostico_rutas(panel, cfg_m1, modelos, origen, desc), cfg_datos,
                        f"m1_pronostico_rutas_{origen}.png")

    # 3. Flota y conductores por día
    tf = tabla_flota_dia()
    tf.to_csv(tablas / "m1_flota_conductores_dia.csv", index=False, encoding="utf-8")
    nombres = {int(r.corredor): r.nombre.split(" → ")[0] + " ↔ " + r.nombre.split(" → ")[1]
               for r in panel.rutas[panel.rutas.sentido == "ida"].itertuples()}
    figuras.guardar(fig_flota_conductores(tf, nombres), cfg_datos, "m1_flota_conductores_dia.png")

    # 4. Sensibilidad de la flota (plan P90 del mejor modelo disponible, prueba)
    params = pa.cargar_parametros()
    mezcla = pa.cargar_mezcla(params)
    v = ventanas_particion(panel, cfg_m1, cfg_datos, "test")
    plan_nombre = next((m for m in ("rnn", "mlp") if m in modelos), "glm")
    m = modelos[plan_nombre]
    vw = ventanas_particion(panel, cfg_con_ventana(cfg_m1, int(m.hp["window"])) if hasattr(m, "hp") else cfg_m1,
                            cfg_datos, "test")
    _, q90 = m.predecir(panel, vw)
    curvas = curva_sensibilidad(panel, v, q90, params, mezcla)
    curvas.insert(0, "plan", f"{plan_nombre}_p90")
    curvas.to_csv(tablas / "m1_sensibilidad_flota.csv", index=False, encoding="utf-8")
    figuras.guardar(fig_sensibilidad_flota(curvas, f"{plan_nombre}_p90"), cfg_datos, "m1_sensibilidad_flota.png")

    # 5. Resultados de la búsqueda y cajas de las 30 semillas
    busq = {f: pd.read_csv(tablas / f"m1_busqueda_{f}.csv") for f in ("mlp", "rnn") if (tablas / f"m1_busqueda_{f}.csv").exists()}
    base_val = {}
    for nombre in ("ingenuo", "glm"):
        tt, *_ = evaluar_modelo(modelos[nombre], panel, cfg_m1, cfg_datos, "val")
        base_val[nombre] = float(tt[(tt.subconjunto == "total") & (tt.ruta_id == "TODAS")].mase.iloc[0])
    figuras.guardar(fig_busqueda(busq, base_val), cfg_datos, "m1_busqueda_resultados.png")
    archivos_sem = [tablas / f"m1_semillas_{f}.csv" for f in ("mlp", "rnn") if (tablas / f"m1_semillas_{f}.csv").exists()]
    if archivos_sem:
        sem = pd.concat([pd.read_csv(a) for a in archivos_sem], ignore_index=True)
        base_eval = {}
        for nombre in ("ingenuo", "glm"):
            base_eval[nombre] = {}
            for part in ("val", "test"):
                tt, *_ = evaluar_modelo(modelos[nombre], panel, cfg_m1, cfg_datos, part)
                base_eval[nombre][part] = float(tt[(tt.subconjunto == "total") & (tt.ruta_id == "TODAS")].mase.iloc[0])
        figuras.guardar(fig_cajas_semillas(sem, base_eval), cfg_datos, "m1_cajas_semillas.png")

    # 6. Arquitectura y curvas de aprendizaje de la red final de cada familia
    for fam in ("mlp", "rnn"):
        if fam not in modelos:
            continue
        pron = modelos[fam]
        cfg_w = cfg_con_ventana(cfg_m1, int(pron.hp["window"]))
        vv = ventanas_particion(panel, cfg_w, cfg_datos, "val").subconjunto(slice(0, 2))
        ejemplo = (torch.as_tensor(vv.x_hist), torch.as_tensor(vv.cal_hist), torch.as_tensor(vv.cal_fut),
                   torch.as_tensor(vv.ruta_idx, dtype=torch.long))
        ta = tabla_arquitectura(pron, ejemplo)
        ta.to_csv(tablas / f"m1_arquitectura_{fam}.csv", index=False, encoding="utf-8")
        figuras.guardar(fig_arquitectura(ta, fam, _descripcion_arquitectura(pron), pron.n_parametros), cfg_datos,
                        f"m1_arquitectura_{fam}.png")
        if not args.sin_reentrenar:
            res, igual, sel = reentrenar_semilla_guardada(fam, panel, cfg_m1, cfg_datos)
            print(f"{fam}: semilla {sel['semilla_guardada']} reentrenada; pesos idénticos a los guardados: {igual}")
            figuras.guardar(fig_curvas_aprendizaje(res.historial, res.mejor_epoca, fam), cfg_datos,
                            f"m1_curvas_aprendizaje_{fam}.png")
            pd.DataFrame(res.historial).to_csv(tablas / f"m1_curvas_aprendizaje_{fam}.csv", index=False, encoding="utf-8")
    print(f"\nEscrito en {tablas.relative_to(RAIZ)} y {ruta_abs(cfg_datos, 'figuras').relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
