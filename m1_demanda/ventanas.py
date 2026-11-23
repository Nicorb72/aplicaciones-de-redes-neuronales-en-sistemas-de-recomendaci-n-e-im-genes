"""Propósito: Constructor de ventanas sin fuga de información para el Módulo 1.
Responsable: Persona 2.

Para un origen ``o`` (primer día pronosticado) y una ruta ``r`` se construye:

* ``x_hist``  (L,)    log((y_t + 1) / (ℓ + 1)) de los L días ``o-L .. o-1`` (``pasajeros_imp``).
                      El +1 evita el log de cero; ℓ > 0 se exige.
* ``cal_hist`` (L, F) calendario de esos L días.
* ``cal_fut``  (H, F) calendario de los H días objetivo ``o .. o+H-1``. Es válido porque los
                      festivos, puentes, temporadas y elecciones se conocen por ley o
                      calendario oficial con meses de anticipación (docs/m1_datos_y_eda.md, 2.4).
* ``nivel``   ℓ = media de ``pasajeros_imp`` en los ``level_days`` días previos a ``o``.
                      Entra como offset: log μ = log ℓ + f(x).
* ``y``, ``mask`` (H,) objetivo observado y máscara de días válidos para pérdida y métricas.

Control de fuga (todo validado en ``tests/test_m1_ventanas.py``)
----------------------------------------------------------------
1. Ninguna fecha de entrada es mayor o igual que el origen.
2. Las columnas de entrada salen de ``calendar_features`` y se validan contra la lista
   ``forbidden_inputs`` y el prefijo ``despachos`` (``validar_columnas_entrada``).
3. Se descartan las ventanas cuyo historial o cuyos objetivos cruzan segmentos, o cuyo
   historial tiene NaN (huecos largos de Villavicencio→Bogotá en 2022 y Duitama→Sogamoso en 2026).
4. El Paro Nacional 2021 (``evento_no_previsible``) no es entrada; sus días objetivo tienen
   peso 0 y, si ``mask.exclude_unforeseeable_inputs``, las ventanas que lo tienen en el
   historial se descartan.
5. La máscara del objetivo exige ``pasajeros`` no nulo y ``imputado`` falso.
6. Los orígenes de cada partición los genera ``rolling_origin`` con la regla de la Persona 1;
   el historial puede usar datos anteriores al origen aunque sean de otra partición (datos ya
   revelados), pero los objetivos siempre caen dentro de la partición.

Uso: ``python -m m1_demanda.ventanas [--config configs/m1_demanda.yaml]``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common.config_loader import cargar_yaml
from m1_demanda.data.utils import RAIZ, cargar_config
from m1_demanda.rolling_origin import origenes_particion

CONFIG_M1 = "configs/m1_demanda.yaml"
PREFIJOS_PROHIBIDOS = ("despachos",)
DOW_ONEHOT = [f"dow_{k}" for k in range(7)]


# ---------------------------------------------------------------------------
# Validación de columnas
# ---------------------------------------------------------------------------
def validar_columnas_entrada(columnas: list[str], cfg_m1: dict) -> None:
    """Falla si alguna columna de entrada no se conoce 30 días antes (o es bandera de limpieza)."""
    prohibidas = set(cfg_m1["forbidden_inputs"])
    malas = [c for c in columnas if c in prohibidas or c.startswith(PREFIJOS_PROHIBIDOS)]
    if malas:
        raise ValueError(
            "Columnas prohibidas como entrada (no se conocen 30 días antes del día objetivo): "
            + ", ".join(malas)
        )


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    """Series y calendario alineados en un índice diario continuo (calendario 2020-2027)."""
    fechas: pd.DatetimeIndex
    rutas: pd.DataFrame            # catálogo (orden = índice de ruta)
    y_imp: np.ndarray              # (T, R) pasajeros_imp, NaN fuera de datos
    y_obs: np.ndarray              # (T, R) pasajeros (limpio), NaN donde falta
    imputado: np.ndarray           # (T, R) bool
    segmento: np.ndarray           # (T, R) int, 0 = sin datos válidos
    evento: np.ndarray             # (T, R) bool, evento no previsible
    calendario: np.ndarray         # (T, F) float32
    alta: np.ndarray               # (T,) bool, temporada_alta
    nombres_cal: list[str] = field(default_factory=list)

    @property
    def ultimo_dato(self) -> pd.Timestamp:
        con_dato = np.where(np.isfinite(self.y_imp).any(axis=1))[0]
        return self.fechas[con_dato.max()]

    def idx(self, fecha) -> int:
        return int((pd.Timestamp(fecha) - self.fechas[0]).days)


def _matriz_calendario(cal: pd.DataFrame, nombres: list[str], tope_dias: int) -> np.ndarray:
    cols = []
    for n in nombres:
        if n in DOW_ONEHOT:
            cols.append((cal["dia_semana"].to_numpy() == int(n.split("_")[1])).astype(np.float32))
        elif n in ("dias_hasta_festivo", "dias_desde_festivo"):
            cols.append(np.minimum(cal[n].to_numpy(dtype=float), tope_dias).astype(np.float32) / tope_dias)
        else:
            if n not in cal.columns:
                raise KeyError(f"Variable de calendario {n!r} no está en calendario_colombia.csv")
            cols.append(cal[n].to_numpy().astype(np.float32))
    return np.stack(cols, axis=1)


def cargar_panel(cfg_m1: dict | None = None, cfg_datos: dict | None = None) -> Panel:
    """Lee los datos procesados de la Persona 1 y arma el ``Panel`` (sin modificarlos)."""
    cfg_m1 = cfg_m1 or cargar_yaml(CONFIG_M1)
    cfg_datos = cfg_datos or cargar_config()
    validar_columnas_entrada(cfg_m1["calendar_features"], cfg_m1)
    proc = RAIZ / cfg_datos["rutas_salida"]["processed"]
    ext = RAIZ / cfg_datos["rutas_salida"]["external"]

    rutas = pd.read_csv(proc / "rutas.csv", dtype={"ruta_id": str, "origen": str, "destino": str,
                                                    "ruta_regreso": str})
    panel = pd.read_csv(proc / "series_diarias.csv.gz", parse_dates=["fecha"], dtype={"ruta_id": str},
                        usecols=["fecha", "ruta_id", "pasajeros", "pasajeros_imp", "imputado",
                                 "segmento", "evento_no_previsible"])
    cal = pd.read_csv(ext / "calendario_colombia.csv", parse_dates=["fecha"]).sort_values("fecha")
    fechas = pd.DatetimeIndex(cal["fecha"])
    if not (fechas == pd.date_range(fechas[0], fechas[-1], freq="D")).all():
        raise ValueError("El calendario externo debe ser un índice diario continuo.")

    def matriz(col: str, relleno, dtype):
        w = panel.pivot(index="fecha", columns="ruta_id", values=col)
        w = w.reindex(index=fechas, columns=rutas["ruta_id"])
        return w.fillna(relleno).to_numpy(dtype=dtype) if relleno is not None else w.to_numpy(dtype=dtype)

    nombres = list(cfg_m1["calendar_features"])
    return Panel(
        fechas=fechas,
        rutas=rutas,
        y_imp=matriz("pasajeros_imp", None, float),
        y_obs=matriz("pasajeros", None, float),
        imputado=matriz("imputado", False, bool),
        segmento=matriz("segmento", 0, int),
        evento=matriz("evento_no_previsible", False, bool),
        calendario=_matriz_calendario(cal, nombres, int(cfg_m1["calendar_day_cap"])),
        alta=cal["temporada_alta"].to_numpy(dtype=bool),
        nombres_cal=nombres,
    )


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------
@dataclass
class Ventanas:
    origen: np.ndarray      # (N,) datetime64[D]
    ruta_idx: np.ndarray    # (N,) int
    x_hist: np.ndarray      # (N, L)
    cal_hist: np.ndarray    # (N, L, F)
    cal_fut: np.ndarray     # (N, H, F)
    nivel: np.ndarray       # (N,)
    y: np.ndarray           # (N, H) objetivo (0 donde mask = 0)
    mask: np.ndarray        # (N, H) 1.0 donde el día cuenta para pérdida y métricas
    alta: np.ndarray        # (N, H) bool, temporada alta del día objetivo

    def __len__(self) -> int:
        return int(self.origen.shape[0])

    def subconjunto(self, sel) -> "Ventanas":
        return Ventanas(**{k: v[sel] for k, v in self.__dict__.items()})


def construir_ventanas(panel: Panel, cfg_m1: dict, origenes, *, rutas: list[int] | None = None,
                       con_objetivo: bool = True) -> Ventanas:
    """Construye las ventanas válidas para los orígenes dados.

    ``con_objetivo=False`` es el modo producción: los días objetivo son futuros y desconocidos
    (``y = 0``, ``mask = 0``) y solo se exige que el historial sea válido.
    """
    L, H = int(cfg_m1["window"]), int(cfg_m1["horizon"])
    nl = int(cfg_m1["level_days"])
    if nl > L:
        raise ValueError("level_days no puede ser mayor que window.")
    m = cfg_m1["mask"]
    T = len(panel.fechas)
    R = panel.y_imp.shape[1]
    rutas = list(range(R)) if rutas is None else rutas

    reg: dict[str, list] = {k: [] for k in Ventanas.__dataclass_fields__}
    for o in origenes:
        i = panel.idx(o)
        if i - L < 0 or i + H > T:
            continue
        for r in rutas:
            hist = panel.y_imp[i - L:i, r]
            if not np.isfinite(hist).all():
                continue
            fin = i + H if con_objetivo else i
            seg = panel.segmento[i - L:fin, r]
            if seg[0] == 0 or (seg != seg[0]).any():
                continue
            if m["exclude_unforeseeable_inputs"] and panel.evento[i - L:i, r].any():
                continue
            nivel = float(panel.y_imp[i - nl:i, r].mean())
            if nivel <= 0:
                continue
            if con_objetivo:
                obs = panel.y_obs[i:i + H, r]
                valido = np.isfinite(obs) & ~panel.imputado[i:i + H, r]
                if m["exclude_unforeseeable_targets"]:
                    valido &= ~panel.evento[i:i + H, r]
                if valido.sum() < int(m["min_valid_targets"]):
                    continue
                y = np.where(valido, obs, 0.0)
            else:
                valido = np.zeros(H, dtype=bool)
                y = np.zeros(H)
            reg["origen"].append(np.datetime64(pd.Timestamp(o).date(), "D"))
            reg["ruta_idx"].append(r)
            reg["x_hist"].append(np.log((hist + 1.0) / (nivel + 1.0)))
            reg["cal_hist"].append(panel.calendario[i - L:i])
            reg["cal_fut"].append(panel.calendario[i:i + H])
            reg["nivel"].append(nivel)
            reg["y"].append(y)
            reg["mask"].append(valido.astype(np.float32))
            reg["alta"].append(panel.alta[i:i + H])

    F = panel.calendario.shape[1]
    if not reg["origen"]:
        z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
        return Ventanas(np.array([], dtype="datetime64[D]"), np.zeros(0, dtype=int), z(0, L), z(0, L, F),
                        z(0, H, F), z(0), z(0, H), z(0, H), np.zeros((0, H), dtype=bool))
    return Ventanas(
        origen=np.array(reg["origen"], dtype="datetime64[D]"),
        ruta_idx=np.array(reg["ruta_idx"], dtype=int),
        x_hist=np.array(reg["x_hist"], dtype=np.float32),
        cal_hist=np.array(reg["cal_hist"], dtype=np.float32),
        cal_fut=np.array(reg["cal_fut"], dtype=np.float32),
        nivel=np.array(reg["nivel"], dtype=np.float32),
        y=np.array(reg["y"], dtype=np.float32),
        mask=np.array(reg["mask"], dtype=np.float32),
        alta=np.array(reg["alta"], dtype=bool),
    )


def ventanas_particion(panel: Panel, cfg_m1: dict, cfg_datos: dict, particion: str,
                       paso_dias: int | None = None) -> Ventanas:
    """Ventanas de una partición ('train', 'val', 'test' o 'post_hueco') con la regla de la Persona 1."""
    if paso_dias is None:
        paso_dias = cfg_m1["train_stride_days"] if particion == "train" else cfg_m1["eval_stride_days"]
    origenes = origenes_particion(cfg_datos, particion, paso_dias=int(paso_dias), horizonte=int(cfg_m1["horizon"]))
    return construir_ventanas(panel, cfg_m1, origenes)


def origen_produccion(panel: Panel) -> pd.Timestamp:
    """Día siguiente al último dato disponible (p. ej. 2026-03-31)."""
    return panel.ultimo_dato + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Resumen de las ventanas del Módulo 1.")
    ap.add_argument("--config", default=CONFIG_M1, help="YAML de modelos del módulo 1")
    args = ap.parse_args(argv)
    cfg_m1 = cargar_yaml(Path(args.config))
    cfg_datos = cargar_config()
    panel = cargar_panel(cfg_m1, cfg_datos)
    print(f"Panel: {len(panel.fechas)} días, {panel.y_imp.shape[1]} rutas, {len(panel.nombres_cal)} variables de calendario")
    print(f"Ventana L={cfg_m1['window']}, horizonte H={cfg_m1['horizon']}, nivel sobre {cfg_m1['level_days']} días")
    for p in ("train", "val", "test", "post_hueco"):
        v = ventanas_particion(panel, cfg_m1, cfg_datos, p)
        extra = f"días objetivo válidos: {int(v.mask.sum())} de {v.mask.size}" if len(v) else "sin ventanas"
        print(f"{p:>10}: {len(v):6d} ventanas  x_hist{v.x_hist.shape}  cal_fut{v.cal_fut.shape}  {extra}")
    prod = origen_produccion(panel)
    vp = construir_ventanas(panel, cfg_m1, [prod], con_objetivo=False)
    print(f"Producción: origen {prod.date()} -> {len(vp)} de {panel.y_imp.shape[1]} rutas con historial válido")


if __name__ == "__main__":
    main()
