"""Propósito: Punto de entrada por línea de comandos del Módulo 1.
Responsable: Persona 2.

Ejecutar siempre como módulo, desde la raíz del repositorio::

    python -m m1_demanda.run --mode evaluate --modelo ingenuo
    python -m m1_demanda.run --mode evaluate --modelo glm
    python -m m1_demanda.run --mode demo

Modos
-----
evaluate  Evalúa un modelo con origen rodante a 30 días (validación y prueba, y opcionalmente
          post_hueco) y escribe ``reports/tables/m1_metricas_<modelo>.csv`` y
          ``reports/tables/m1_horizonte_<modelo>.csv``.
train, search, demo  Se agregan en las tareas siguientes del plan (docs/plan_persona2.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.config_loader import cargar_yaml
from common.seeds import elegir_dispositivo, fijar_semillas
from m1_demanda.data.utils import RAIZ, cargar_config, ruta_abs
from m1_demanda.evaluate import evaluar_modelo
from m1_demanda.ventanas import cargar_panel

MODOS = ("evaluate", "train", "search", "final", "demo")
MODELOS = ("ingenuo", "glm", "mlp", "rnn")
CONFIG_M1 = "configs/m1_demanda.yaml"
CONFIG_GLOBAL = "configs/global.yaml"


def construir_modelo(nombre: str, panel, cfg_m1: dict, cfg_datos: dict):
    """Crea y deja listo (ajustado) el modelo base pedido."""
    if nombre == "ingenuo":
        from m1_demanda.baselines import IngenuoEstacional
        return IngenuoEstacional(cuantil=float(cfg_m1["quantile"])).ajustar(panel, cfg_m1, cfg_datos)
    if nombre == "glm":
        from m1_demanda.baselines import GLMPoisson
        return GLMPoisson(cuantil=float(cfg_m1["quantile"])).ajustar(panel, cfg_m1, cfg_datos)
    if nombre in ("mlp", "rnn"):
        from m1_demanda.train import MODELOS, cargar_modelo
        # La red que se defiende es la FINAL (mejor semilla por validación de la corrida con 30 semillas).
        # `<modelo>.pt` es solo la corrida suelta de `--mode train` (una semilla) y se usa si aún no hay final.
        base = MODELOS / f"{nombre}_final"
        if not base.with_suffix(".pt").exists():
            base = MODELOS / nombre
            print(f"AVISO: no hay {nombre}_final; se evalúa la corrida suelta de una semilla ({base.name}.pt).")
        if not base.with_suffix(".pt").exists():
            raise FileNotFoundError(f"No hay un modelo guardado de {nombre}; ejecute primero "
                                    f"`python -m m1_demanda.run --mode train --modelo {nombre}` (una semilla) o "
                                    f"`--mode final` (30 semillas).")
        return cargar_modelo(base, n_rutas=len(panel.rutas), device="cpu")
    raise ValueError(f"Modelo desconocido: {nombre!r}")


def modo_evaluate(args, cfg_m1: dict, cfg_datos: dict) -> Path:
    panel = cargar_panel(cfg_m1, cfg_datos)
    modelo = construir_modelo(args.modelo, panel, cfg_m1, cfg_datos)
    tablas, horizontes = [], []
    for part in args.particiones:
        tabla, hor, _, _, _ = evaluar_modelo(modelo, panel, cfg_m1, cfg_datos, part)
        tabla.insert(0, "particion", part)
        hor.insert(0, "particion", part)
        tablas.append(tabla)
        horizontes.append(hor)
    metricas = pd.concat(tablas, ignore_index=True)
    metricas.insert(0, "modelo", args.modelo)
    horizonte = pd.concat(horizontes, ignore_index=True)
    horizonte.insert(0, "modelo", args.modelo)

    salida = ruta_abs(cfg_datos, "tablas")
    ruta = salida / f"m1_metricas_{args.modelo}.csv"
    metricas.to_csv(ruta, index=False, encoding="utf-8")
    horizonte.to_csv(salida / f"m1_horizonte_{args.modelo}.csv", index=False, encoding="utf-8")

    resumen = metricas[(metricas.ruta_id == "TODAS")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(resumen.drop(columns=["ruta_id"]).round(3).to_string(index=False))
    print(f"\nEscrito: {ruta.relative_to(RAIZ)}")
    return ruta


def modo_train_ingenuo(cfg_m1: dict, cfg_datos: dict) -> Path:
    """El ingenuo no tiene parámetros: 'entrenar' es calibrar su P90 con validación y guardarlo."""
    panel = cargar_panel(cfg_m1, cfg_datos)
    modelo = construir_modelo("ingenuo", panel, cfg_m1, cfg_datos)
    ruta = modelo.guardar(RAIZ / "models" / "m1_demanda" / "ingenuo.json")
    print(f"Ingenuo calibrado con validación. P90/media por horizonte: "
          f"h1={modelo.razon_p90[0]:.3f} ... h30={modelo.razon_p90[-1]:.3f}")
    print(f"Escrito: {ruta.relative_to(RAIZ)}")
    return ruta


def modo_train_red(args, cfg_m1: dict, cfg_datos: dict) -> Path:
    """Entrena una red con los hiperparámetros del YAML y UNA semilla; guarda ``models/m1_demanda/<modelo>``."""
    from m1_demanda.train import MODELOS, entrenar, guardar_modelo, hp_desde_config

    panel = cargar_panel(cfg_m1, cfg_datos)
    hp = hp_desde_config(cfg_m1)
    res = entrenar(args.modelo, hp, cfg_m1, cfg_datos, panel, args.seed, elegir_dispositivo(args.device),
                   verbose=args.verbose)
    m = res.pronosticador.meta
    print(f"{args.modelo}: {res.n_parametros:,} parámetros entrenables | mejor época {res.mejor_epoca} de "
          f"{m['epocas_corridas']} | val {res.mejor_val:.4f} | {res.segundos:.1f} s "
          f"({res.segundos_por_epoca * 1000:.0f} ms/época) en {m['dispositivo']}")
    pt, _ = guardar_modelo(res.pronosticador, panel.nombres_cal, MODELOS / args.modelo)
    print(f"Escrito: {pt.relative_to(RAIZ)}")
    return pt


def modo_search(args, cfg_m1: dict, cfg_datos: dict) -> Path:
    """Búsqueda aleatoria con presupuesto; escribe las tablas y ``models/m1_demanda/mejor_config_<modelo>.json``."""
    from m1_demanda.search import buscar, guardar_busqueda

    panel = cargar_panel(cfg_m1, cfg_datos)
    reg, tabla, _ = buscar(args.modelo, cfg_m1, cfg_datos, panel, presupuesto=args.presupuesto, semilla=args.seed,
                           device=elegir_dispositivo(args.device), verbose=True)
    csv, mejor = guardar_busqueda(args.modelo, reg, tabla, cfg_m1, args.seed, ruta_abs(cfg_datos, "tablas"))
    r = reg.resumen()
    print(f"\n{args.modelo}: {r['n_configuraciones']} configuraciones evaluadas en {r['tiempo_total_s']:.0f} s "
          f"(presupuesto {r['presupuesto']}).")
    print(f"Escrito: {csv.relative_to(RAIZ)} y {mejor.relative_to(RAIZ)}")
    return mejor


def _baselines_evaluados(cfg_m1: dict, cfg_datos: dict, panel) -> tuple[pd.DataFrame, dict, dict]:
    """Evalúa ingenuo y GLM en val/test/post_hueco. Devuelve (tabla total, horizonte por modelo, cobertura por modelo)."""
    filas, hor, cob = [], {}, {}
    for nombre in ("ingenuo", "glm"):
        modelo = construir_modelo(nombre, panel, cfg_m1, cfg_datos)
        for part in ("val", "test", "post_hueco"):
            tabla, h, *_ = evaluar_modelo(modelo, panel, cfg_m1, cfg_datos, part)
            t = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id == "TODAS")].iloc[0]
            filas.append({"modelo": nombre, "particion": part, "n_parametros": int(getattr(modelo, "n_parametros", 0)),
                          **{m: float(t[m]) for m in ("mase", "mae", "rmse", "cobertura", "pinball_escalada", "pinball")}})
            if part == "test":
                hor[nombre] = h
                cob[nombre] = tabla[(tabla.subconjunto == "total") & (tabla.ruta_id != "TODAS")].set_index("ruta_id")["cobertura"]
    return pd.DataFrame(filas), hor, cob


def modo_final(args, cfg_m1: dict, cfg_datos: dict, cfg_global: dict) -> Path:
    """30 semillas por configuración final, tabla de comparación justa, figuras y modelo guardado para la app."""
    from common.seeds import lista_semillas
    from m1_demanda import figuras
    from m1_demanda.final import correr_familia, guardar_final, tabla_comparacion

    panel = cargar_panel(cfg_m1, cfg_datos)
    semillas = lista_semillas(cfg_global["seed"], args.semillas or cfg_global["num_seeds"])
    familias = [args.modelo] if args.modelo in ("mlp", "rnn") else ["mlp", "rnn"]
    tablas_dir = ruta_abs(cfg_datos, "tablas")
    device = elegir_dispositivo(args.device)

    todas, extras, finales = [], {"horizonte": [], "ruta": []}, {}
    presupuestos = {"ingenuo": 0, "glm": 1}
    for fam in familias:
        tabla, ex, mejor, info = correr_familia(fam, cfg_m1, cfg_datos, panel, semillas, device, verbose=True,
                                                top_k=args.top_k)
        tabla.to_csv(tablas_dir / f"m1_semillas_{fam}.csv", index=False, encoding="utf-8")
        guardar_final(fam, mejor, info, panel)
        todas.append(tabla)
        for k in extras:
            extras[k].append(ex[k])
        finales[fam] = info
        presupuestos[fam] = int(cfg_m1["search"]["budget"])

    semillas_df = pd.concat(todas, ignore_index=True)
    from m1_demanda.final import elegir_familia_final, escribir_modelo_final
    eleccion = elegir_familia_final(semillas_df, finales)
    print(f"Red para la app: {eleccion['familia']} (MASE medio de validación {eleccion['mase_val_media']:.3f})")
    escribir_modelo_final(eleccion)
    base, hor_base, cob_base = _baselines_evaluados(cfg_m1, cfg_datos, panel)
    comparacion = tabla_comparacion(semillas_df, base, presupuestos)
    salida = tablas_dir / "m1_comparacion.csv"
    comparacion.to_csv(salida, index=False, encoding="utf-8")

    # --- figuras (configuración final de cada familia, partición de prueba) ---
    hor_ex, ruta_ex = pd.concat(extras["horizonte"]), pd.concat(extras["ruta"])
    series, cobertura = dict(hor_base), dict(cob_base)
    for fam, info in finales.items():
        r = info["config_final_rank"]
        h = hor_ex[(hor_ex.familia == fam) & (hor_ex.config_rank == r) & (hor_ex.particion == "test")]
        g = h.groupby("horizonte")["mase"]
        series[fam] = pd.DataFrame({"horizonte": g.mean().index, "mase": g.mean().to_numpy(), "mase_desv": g.std().to_numpy()})
        c = ruta_ex[(ruta_ex.familia == fam) & (ruta_ex.config_rank == r) & (ruta_ex.particion == "test")]
        cobertura[fam] = c.groupby("ruta_id")["cobertura"].mean()
    figuras.guardar(figuras.fig_error_por_horizonte(series), cfg_datos, "m1_error_por_horizonte.png")
    figuras.guardar(figuras.fig_cobertura_por_ruta(cobertura), cfg_datos, "m1_cobertura_por_ruta.png")
    _figura_pronostico(panel, cfg_m1, cfg_datos, finales, figuras)

    t = comparacion[(comparacion.particion == "test")][["modelo", "n_parametros", "mase_media", "mase_desv", "cobertura_media"]]
    with pd.option_context("display.width", 200):
        print(t.round(3).to_string(index=False))
    print(f"\nEscrito: {salida.relative_to(RAIZ)} y modelos *_final en models/m1_demanda/")
    return salida


def _figura_pronostico(panel, cfg_m1: dict, cfg_datos: dict, finales: dict, figuras,
                       ruta_id: str = "50001-11001", origen: str = "2023-12-02") -> None:
    """Pronóstico vs observado de una ruta desde el último origen de prueba (cubre hasta el 31 de diciembre).

    Usa los modelos ``<familia>_final`` ya guardados (la semilla de menor pérdida de VALIDACIÓN).
    """
    import numpy as np

    from m1_demanda.train import MODELOS, cargar_modelo, cfg_con_ventana
    from m1_demanda.ventanas import construir_ventanas

    idx = int(panel.rutas.index[panel.rutas["ruta_id"] == ruta_id][0])
    i0 = panel.idx(origen)
    fechas = panel.fechas[i0:i0 + int(cfg_m1["horizon"])]
    observado = panel.y_obs[i0:i0 + int(cfg_m1["horizon"]), idx]
    pronosticos = {}
    for nombre in ("ingenuo", "glm"):
        m = construir_modelo(nombre, panel, cfg_m1, cfg_datos)
        v = construir_ventanas(panel, cfg_m1, [origen], rutas=[idx])
        mu, q = m.predecir(panel, v)
        pronosticos[nombre] = (mu[0], q[0])
    for fam in finales:
        m = cargar_modelo(MODELOS / f"{fam}_final", n_rutas=len(panel.rutas))
        cfg = cfg_con_ventana(cfg_m1, int(m.hp["window"]))
        v = construir_ventanas(panel, cfg, [origen], rutas=[idx])
        mu, q = m.predecir(panel, v)
        pronosticos[fam] = (mu[0], q[0])
    nombre_ruta = panel.rutas.loc[idx, "nombre"]
    fig = figuras.fig_pronostico_vs_observado(fechas, np.asarray(observado), pronosticos,
                                              f"Pronóstico desde {origen}", nombre_ruta)
    figuras.guardar(fig, cfg_datos, "m1_pronostico_vs_observado.png")


def modo_demo(args) -> None:
    """Demo de menos de 2 minutos: SOLO carga artefactos (no entrena ni descarga nada).

    Pronostica las 10 rutas en el origen de producción y en un origen de prueba (2023-12-02, que cubre fin de
    año), y muestra la operación por corredor planeando con el P90.
    """
    import time

    from m1_demanda import predict

    t0 = time.perf_counter()
    rutas = predict.list_routes()
    print("=== Módulo 1: pronóstico a 30 días y operación (demo) ===")
    for etiqueta, origen in (("PRODUCCIÓN (día siguiente al último dato)", None), ("PRUEBA (reproduce un pronóstico)", "2023-12-02")):
        dfs = [predict.forecast(r, 0.9, origen=origen) for r in rutas["ruta_id"]]
        a = dfs[0].attrs
        print(f"\n--- {etiqueta}: origen {a['origen']} | modelo: {a['modelo']} ---")
        filas = []
        for r, d in zip(rutas.itertuples(), dfs):
            fila = {"ruta": r.nombre, "pax_30d (media)": d["pronostico"].sum(), "pax_30d (P90)": d["p90"].sum(),
                    "despachos/día (prom)": d["despachos"].mean()}
            if d["real"].notna().any():
                fila["pax_30d (real)"] = d["real"].sum()
            filas.append(fila)
        with pd.option_context("display.width", 200, "display.float_format", "{:,.0f}".format):
            print(pd.DataFrame(filas).to_string(index=False))
        oper = pd.DataFrame({
            "corredor": [r.corredor for r in rutas.itertuples() if r.sentido == "ida"],
            "flota": [d.attrs["flota_corredor"] for r, d in zip(rutas.itertuples(), dfs) if r.sentido == "ida"],
            "plantilla": [d.attrs["plantilla_conductores"] for r, d in zip(rutas.itertuples(), dfs) if r.sentido == "ida"],
            "vehículos/día (pico)": [d["vehiculos"].max() for r, d in zip(rutas.itertuples(), dfs) if r.sentido == "ida"],
        })
        print("\nOperación por corredor (plan con P90):")
        print(oper.to_string(index=False))
        avisos = sorted({w for d in dfs for w in d.attrs["advertencias"]})
        for w in avisos:
            print(f"  ! {w}")
    print(f"\nDemo terminada en {time.perf_counter() - t0:.1f} s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Módulo 1: pronóstico de demanda y operación.")
    ap.add_argument("--mode", choices=MODOS, required=True)
    ap.add_argument("--modelo", choices=MODELOS, default="ingenuo")
    ap.add_argument("--config", default=CONFIG_M1, help="YAML de modelos del módulo 1")
    ap.add_argument("--config-datos", default=None, help="YAML de datos (Persona 1); por defecto el suyo")
    ap.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default=None,
                    help="dispositivo de PyTorch (por defecto el de configs/global.yaml)")
    ap.add_argument("--seed", type=int, default=None, help="semilla (por defecto la de configs/global.yaml)")
    ap.add_argument("--verbose", action="store_true", help="imprime la pérdida por época")
    ap.add_argument("--semillas", type=int, default=None, help="número de semillas del modo final (por defecto num_seeds)")
    ap.add_argument("--top-k", type=int, default=None, dest="top_k",
                    help="cuántas de las mejores configuraciones correr en el modo final (por defecto search.top_k)")
    ap.add_argument("--presupuesto", type=int, default=None,
                    help="número de configuraciones de la búsqueda (por defecto search.budget)")
    ap.add_argument("--particiones", nargs="+", default=["val", "test"],
                    choices=("val", "test", "post_hueco"), help="particiones a evaluar")
    args = ap.parse_args(argv)

    cfg_global = cargar_yaml(CONFIG_GLOBAL)
    cfg_m1 = cargar_yaml(args.config)
    cfg_datos = cargar_config(args.config_datos)
    args.device = args.device or cfg_global["device"]
    args.seed = cfg_global["seed"] if args.seed is None else args.seed
    fijar_semillas(args.seed)
    elegir_dispositivo(args.device)

    if args.mode == "evaluate":
        modo_evaluate(args, cfg_m1, cfg_datos)
    elif args.mode == "train" and args.modelo == "ingenuo":
        modo_train_ingenuo(cfg_m1, cfg_datos)
    elif args.mode == "train" and args.modelo in ("mlp", "rnn"):
        modo_train_red(args, cfg_m1, cfg_datos)
    elif args.mode == "search" and args.modelo in ("mlp", "rnn"):
        modo_search(args, cfg_m1, cfg_datos)
    elif args.mode == "final":
        modo_final(args, cfg_m1, cfg_datos, cfg_global)
    elif args.mode == "demo":
        modo_demo(args)
    else:
        raise NotImplementedError(f"El modo {args.mode!r} se implementa en una tarea posterior del plan.")


if __name__ == "__main__":
    main()
