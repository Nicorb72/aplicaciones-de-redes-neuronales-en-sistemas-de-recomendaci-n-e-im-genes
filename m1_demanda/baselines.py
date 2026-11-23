"""Propósito: Modelos base del Módulo 1 (ingenuo estacional y, más adelante, GLM Poisson).
Responsable: Persona 2.

Ingenuo estacional (mismo día de la semana anterior)
----------------------------------------------------
Para el día objetivo ``t = o + k`` (k = 0, ..., H-1; k = 0 es el origen) se usa el día más
reciente con el mismo día de la semana **estrictamente anterior al origen**:

    rezago(k) = 7 · (⌊k / 7⌋ + 1)         ŷ_{o+k} = y_{o+k − rezago(k)} = y_{o − 7 + (k mod 7)}

Es decir, el ingenuo repite la última semana observada. Usar ``y_{t−7}`` con ``t > o + 6``
sería fuga de información (ese dato no se conoce en el origen).

P90 del ingenuo: cuantil empírico 0,9 del cociente ``y / ŷ`` por horizonte, estimado con los
orígenes de VALIDACIÓN (nunca prueba), y ``q = ŷ · max(cociente_P90, 1)`` para mantener
``q >= ŷ``. Presupuesto de búsqueda: 0 hiperparámetros.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m1_demanda.ventanas import Panel, Ventanas, ventanas_particion


def rezago_ingenuo(k: int) -> int:
    """Rezago (en días) del ingenuo estacional para el día objetivo ``o + k`` (k desde 0)."""
    return 7 * (k // 7 + 1)


class IngenuoEstacional:
    """Repite la última semana observada antes del origen. Sin parámetros entrenables."""

    nombre = "ingenuo"
    n_parametros = 0
    n_configuraciones = 0

    def __init__(self, cuantil: float = 0.9):
        self.cuantil = float(cuantil)
        self.razon_p90: np.ndarray | None = None   # (H,) razón y/ŷ al cuantil, por horizonte

    # -- pronóstico puntual -------------------------------------------------
    @staticmethod
    def puntual(panel: Panel, v: Ventanas) -> np.ndarray:
        H = v.y.shape[1]
        mu = np.empty((len(v), H), dtype=float)
        for n in range(len(v)):
            i = panel.idx(v.origen[n])
            r = v.ruta_idx[n]
            for k in range(H):
                mu[n, k] = panel.y_imp[i + k - rezago_ingenuo(k), r]
        if not np.isfinite(mu).all():
            raise ValueError("El ingenuo encontró NaN en el último semana observada.")
        return mu

    # -- calibración del P90 (solo validación) ---------------------------------
    def ajustar(self, panel: Panel, cfg_m1: dict, cfg_datos: dict) -> "IngenuoEstacional":
        v = ventanas_particion(panel, cfg_m1, cfg_datos, "val")
        mu = self.puntual(panel, v)
        H = mu.shape[1]
        razon = np.ones(H)
        for k in range(H):
            ok = (v.mask[:, k] > 0) & (mu[:, k] > 0)
            if ok.sum() < 10:
                raise ValueError(f"Muy pocas ventanas de validación en el horizonte {k + 1}.")
            razon[k] = max(float(np.quantile(v.y[ok, k] / mu[ok, k], self.cuantil)), 1.0)
        self.razon_p90 = razon
        return self

    def predecir(self, panel: Panel, v: Ventanas) -> tuple[np.ndarray, np.ndarray]:
        if self.razon_p90 is None:
            raise RuntimeError("Llame ajustar() antes de predecir() para calibrar el P90.")
        mu = self.puntual(panel, v)
        return mu, mu * self.razon_p90[None, :]

    # -- una ruta, un origen (lo usa predict.py) --------------------------------
    def predecir_ruta(self, panel: Panel, ruta_idx: int, origen, horizonte: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Pronóstico de una ruta desde ``origen`` usando SOLO los 7 días anteriores al origen.

        A diferencia de ``predecir`` no exige la ventana completa de entrada: sirve de respaldo cuando
        el historial largo tiene huecos. Lanza ``ValueError`` si la última semana tiene NaN.
        """
        if self.razon_p90 is None:
            raise RuntimeError("Llame ajustar() o cargar() antes de predecir.")
        if horizonte != len(self.razon_p90):
            raise ValueError(f"El P90 está calibrado para {len(self.razon_p90)} días, no {horizonte}.")
        i = panel.idx(origen)
        if i - 7 < 0 or i + horizonte > len(panel.fechas):
            raise ValueError(f"El origen {pd.Timestamp(origen).date()} queda fuera del calendario disponible.")
        semana = panel.y_imp[i - 7:i, ruta_idx]
        if not np.isfinite(semana).all():
            raise ValueError("La última semana antes del origen tiene datos faltantes: no se puede usar el ingenuo.")
        mu = np.array([panel.y_imp[i + k - rezago_ingenuo(k), ruta_idx] for k in range(horizonte)], dtype=float)
        return mu, mu * self.razon_p90

    # -- persistencia ---------------------------------------------------------------
    def guardar(self, ruta: str | Path) -> Path:
        if self.razon_p90 is None:
            raise RuntimeError("Nada que guardar: el modelo no está calibrado.")
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps({
            "modelo": self.nombre, "cuantil": self.cuantil, "n_parametros": 0,
            "razon_p90": [float(x) for x in self.razon_p90],
            "nota": "Calibrado con orígenes de VALIDACIÓN (m1_demanda/baselines.py).",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: str | Path) -> "IngenuoEstacional":
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        m = cls(cuantil=d["cuantil"])
        m.razon_p90 = np.array(d["razon_p90"], dtype=float)
        return m


# ===========================================================================
# GLM Poisson por ruta (regresión lineal en escala log con rezagos y calendario)
# ===========================================================================
SEMANAS_REZAGO = 4
# Variables de calendario redundantes: son combinación lineal exacta de dow_0..dow_6 y el intercepto.
CAL_REDUNDANTES = ("dow_sin", "dow_cos", "fin_de_semana", "dow_0")


class GLMPoisson:
    """GLM Poisson con enlace log y offset ``log ℓ``, un modelo por ruta.

        log μ_{r,o,k} = log ℓ_{r,o} + β_r · x_{o,k}

    Entradas ``x_{o,k}`` para el día objetivo ``k`` (k = 0..H-1) del origen ``o`` (todas conocidas en ``o``):

    * intercepto;
    * calendario del día objetivo (se conoce por ley; sin las variables redundantes);
    * rezagos: ``log((y+1)/(ℓ+1))`` del MISMO día de la semana de las 4 semanas anteriores
      (``x_hist[L − 7w + (k mod 7)]``, w = 1..4): son los rezagos relevantes para ese día;
    * tendencia: ``log`` de la media de los últimos 7 días relativa a ℓ;
    * horizonte ``k`` en one-hot (H − 1 columnas).

    Pérdida: Poisson con cada ventana pesada por 1/ℓ, la MISMA que usan las redes (``losses.py``), para que
    la comparación aísle la arquitectura y no la función de pérdida. Es la misma información que usa la MLP (un subconjunto de su entrada); la diferencia es que aquí la
    relación es lineal en escala log y un modelo por ruta, así que la comparación con la MLP aísla el aporte
    de la no linealidad y del modelo global. Un solo hiperparámetro configurado (presupuesto: 1).

    P90: igual que el ingenuo, cuantil empírico 0,9 de y/μ por horizonte con orígenes de VALIDACIÓN.
    """

    nombre = "glm"
    n_configuraciones = 1

    def __init__(self, cuantil: float = 0.9, tope_log_razon: float = 3.0):
        self.cuantil = float(cuantil)
        self.tope = float(tope_log_razon)      # protege exp() ante extrapolaciones: μ/ℓ ∈ [e^-3, e^3]
        self.betas: dict[int, np.ndarray] = {}
        self.columnas: list[str] = []
        self.razon_p90: np.ndarray | None = None
        self._idx_cal: list[int] | None = None
        self.convergio: dict[int, bool] = {}

    @property
    def n_parametros(self) -> int:
        return int(sum(len(b) for b in self.betas.values()))

    # -- diseño ------------------------------------------------------------------
    def _seleccion_calendario(self, nombres: list[str]) -> list[int]:
        return [i for i, n in enumerate(nombres) if n not in CAL_REDUNDANTES]

    def disenio(self, v: Ventanas, nombres_cal: list[str]) -> np.ndarray:
        """Matriz de diseño (N, H, P) con las entradas descritas arriba."""
        N, H, _ = v.cal_fut.shape
        L = v.x_hist.shape[1]
        if L < 7 * SEMANAS_REZAGO:
            raise ValueError(f"La ventana ({L}) debe tener al menos {7 * SEMANAS_REZAGO} días para los rezagos.")
        idx = self._seleccion_calendario(nombres_cal)
        k = np.arange(H)
        cal = v.cal_fut[:, :, idx]
        lags = np.stack([v.x_hist[:, L - 7 * w + (k % 7)] for w in range(1, SEMANAS_REZAGO + 1)], axis=-1)
        tend = np.log(np.exp(v.x_hist[:, -7:]).mean(axis=1))
        tend = np.broadcast_to(tend[:, None, None], (N, H, 1))
        hor = np.broadcast_to(np.eye(H, dtype=np.float32)[None, :, 1:], (N, H, H - 1))
        unos = np.ones((N, H, 1), dtype=np.float32)
        X = np.concatenate([unos, cal, lags, tend, hor], axis=-1).astype(float)
        self.columnas = (["intercepto"] + [nombres_cal[i] for i in idx]
                         + [f"lag_{w}sem" for w in range(1, SEMANAS_REZAGO + 1)]
                         + ["tendencia_7d"] + [f"h_{j + 1}" for j in range(1, H)])
        return X

    # -- ajuste ------------------------------------------------------------------
    def ajustar(self, panel: Panel, cfg_m1: dict, cfg_datos: dict) -> "GLMPoisson":
        nombres = list(panel.nombres_cal)
        tr = ventanas_particion(panel, cfg_m1, cfg_datos, "train")
        X = self.disenio(tr, nombres)
        for r in range(len(panel.rutas)):
            sel = tr.ruta_idx == r
            m = tr.mask[sel] > 0
            Xr = X[sel][m]
            yr = tr.y[sel][m].astype(float)
            niv = np.broadcast_to(tr.nivel[sel].astype(float)[:, None], tr.y[sel].shape)[m]
            # Misma pérdida que la red (m1_demanda/losses.py): Poisson con cada ventana pesada por 1/ℓ.
            self.betas[r], self.convergio[r] = self._ajustar_ruta(Xr, yr, np.log(niv), 1.0 / niv)
        self._calibrar_p90(panel, cfg_m1, cfg_datos)
        return self

    @staticmethod
    def _ajustar_ruta(X: np.ndarray, y: np.ndarray, offset: np.ndarray,
                      pesos: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        """Ajusta un GLM Poisson (enlace log) con ``offset`` y pesos opcionales; devuelve ``(beta, convergió)``."""
        import statsmodels.api as sm
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset,
                         freq_weights=pesos).fit(maxiter=100)
        return np.asarray(res.params, dtype=float), bool(res.converged)

    def _mu(self, v: Ventanas, nombres: list[str]) -> np.ndarray:
        X = self.disenio(v, nombres)
        eta = np.zeros(v.y.shape)
        for r, beta in self.betas.items():
            sel = v.ruta_idx == r
            if sel.any():
                eta[sel] = X[sel] @ beta
        return v.nivel[:, None].astype(float) * np.exp(np.clip(eta, -self.tope, self.tope))

    def _calibrar_p90(self, panel: Panel, cfg_m1: dict, cfg_datos: dict) -> None:
        v = ventanas_particion(panel, cfg_m1, cfg_datos, "val")
        mu = self._mu(v, list(panel.nombres_cal))
        H = mu.shape[1]
        razon = np.ones(H)
        for k in range(H):
            ok = (v.mask[:, k] > 0) & (mu[:, k] > 0)
            razon[k] = max(float(np.quantile(v.y[ok, k] / mu[ok, k], self.cuantil)), 1.0)
        self.razon_p90 = razon

    def predecir(self, panel: Panel, v: Ventanas) -> tuple[np.ndarray, np.ndarray]:
        if self.razon_p90 is None or not self.betas:
            raise RuntimeError("Llame ajustar() antes de predecir().")
        mu = self._mu(v, list(panel.nombres_cal))
        return mu, mu * self.razon_p90[None, :]
