"""Propósito: Entrenamiento, validación, persistencia y pronóstico de las redes del Módulo 1.
Responsable: Persona 2.

Sigue el estilo del curso: ``Dataset`` + ``DataLoader``, funciones ``entrenar_epoca`` y ``validar``
explícitas, ``torch.random.manual_seed`` por semilla y conteo de parámetros con ``numel``.

Protocolo (decisión D7 de docs/plan_persona2.md)
------------------------------------------------
* Los pesos se entrenan UNA vez con los orígenes de ENTRENAMIENTO (paso de 1 día).
* La **parada temprana** vigila la pérdida combinada en VALIDACIÓN (orígenes de paso 1 día, más ventanas
  y por tanto menos ruido). Se conserva el estado de la mejor época.
* Nunca se usa la partición de prueba aquí. La selección de hiperparámetros (``search.py``) usa orígenes
  semanales de validación.
* La red aprende ``f`` y ``g``; el pronóstico es ``μ = ℓ·exp(f)`` y ``q = μ·exp(softplus(g))``.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.seeds import fijar_semillas
from m1_demanda.data.utils import RAIZ
from m1_demanda.losses import PerdidaCombinada, media_y_cuantil
from m1_demanda.nn_models import construir_modelo, contar_parametros
from m1_demanda.ventanas import Panel, Ventanas, ventanas_particion

MODELOS = RAIZ / "models" / "m1_demanda"


# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
CLAVES_HP = ("window", "layers", "units", "regularization", "lr", "batch_size", "epochs", "patience",
             "route_embedding_dim", "quantile", "quantile_weight", "optimizer", "rnn")


def hp_desde_config(cfg_m1: dict, **cambios) -> dict:
    """Extrae los hiperparámetros de ``configs/m1_demanda.yaml`` y aplica ``cambios`` (p. ej. de la búsqueda).

    Los cambios a diccionarios (``regularization``, ``rnn``) se fusionan con los valores por defecto.
    """
    hp = {k: copy.deepcopy(cfg_m1[k]) for k in CLAVES_HP if k in cfg_m1}
    for k, v in cambios.items():
        if isinstance(v, dict) and isinstance(hp.get(k), dict):
            hp[k] = {**hp[k], **v}
        else:
            hp[k] = v
    hp["horizon"] = int(cfg_m1["horizon"])
    return hp


def cfg_con_ventana(cfg_m1: dict, ventana: int) -> dict:
    """Copia de la configuración con otra longitud de ventana (la búsqueda prueba 28 y 56 días)."""
    cfg = copy.deepcopy(cfg_m1)
    cfg["window"] = int(ventana)
    cfg["level_days"] = min(int(cfg_m1["level_days"]), int(ventana))
    return cfg


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
class DatasetVentanas(Dataset):
    """Envuelve ``Ventanas`` como tensores; cada elemento es un diccionario de tensores."""

    def __init__(self, v: Ventanas):
        self.t = {
            "x_hist": torch.as_tensor(v.x_hist), "cal_hist": torch.as_tensor(v.cal_hist),
            "cal_fut": torch.as_tensor(v.cal_fut), "ruta_idx": torch.as_tensor(v.ruta_idx, dtype=torch.long),
            "nivel": torch.as_tensor(v.nivel), "y": torch.as_tensor(v.y), "mask": torch.as_tensor(v.mask),
        }
        self.n = len(v)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {k: t[i] for k, t in self.t.items()}


def _a_device(lote: dict, device: torch.device) -> dict:
    return {k: t.to(device) for k, t in lote.items()}


def _salida(modelo: nn.Module, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return modelo(b["x_hist"], b["cal_hist"], b["cal_fut"], b["ruta_idx"])


# ---------------------------------------------------------------------------
# Bucles (estilo del curso)
# ---------------------------------------------------------------------------
def entrenar_epoca(dataloader: DataLoader, modelo: nn.Module, perdida: PerdidaCombinada,
                   optimizer: torch.optim.Optimizer, device: torch.device, clip: float = 1.0) -> float:
    """Una época de entrenamiento; devuelve la pérdida media ponderada por número de días válidos."""
    modelo.train()
    suma, n = 0.0, 0.0
    for lote in dataloader:
        b = _a_device(lote, device)
        f, g = _salida(modelo, b)
        total, _, _ = perdida(f, g, b["nivel"], b["y"], b["mask"])
        optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), clip)
        optimizer.step()
        validos = float(b["mask"].sum())
        suma += float(total.detach()) * validos
        n += validos
    return suma / max(n, 1.0)


def validar(dataloader: DataLoader, modelo: nn.Module, perdida: PerdidaCombinada,
            device: torch.device) -> dict:
    """Pérdida de validación (total, Poisson y pinball) sin calcular gradientes."""
    modelo.eval()
    suma = np.zeros(3)
    n = 0.0
    with torch.no_grad():
        for lote in dataloader:
            b = _a_device(lote, device)
            f, g = _salida(modelo, b)
            total, lp, lq = perdida(f, g, b["nivel"], b["y"], b["mask"])
            validos = float(b["mask"].sum())
            suma += np.array([float(total), float(lp), float(lq)]) * validos
            n += validos
    t, p, q = suma / max(n, 1.0)
    return {"total": float(t), "poisson": float(p), "pinball": float(q)}


# ---------------------------------------------------------------------------
# Pronosticador (interfaz de evaluate.py)
# ---------------------------------------------------------------------------
class PronosticadorRed:
    """Envuelve una red entrenada con la interfaz ``predecir(panel, ventanas) -> (mu, q)``."""

    def __init__(self, modelo: nn.Module, familia: str, hp: dict, device: torch.device, meta: dict | None = None):
        self.modelo = modelo.to(device).eval()
        self.familia, self.hp, self.device = familia, hp, device
        self.meta = meta or {}
        self.nombre = familia
        self.n_parametros = contar_parametros(modelo)

    def predecir(self, panel: Panel, v: Ventanas, lote: int = 512) -> tuple[np.ndarray, np.ndarray]:
        if int(v.x_hist.shape[1]) != int(self.hp["window"]):
            raise ValueError(f"El modelo espera ventanas de {self.hp['window']} días, no {v.x_hist.shape[1]}.")
        mus, qs = [], []
        self.modelo.eval()
        with torch.no_grad():
            for lo in range(0, len(v), lote):
                s = slice(lo, lo + lote)
                b = {"x_hist": torch.as_tensor(v.x_hist[s]), "cal_hist": torch.as_tensor(v.cal_hist[s]),
                     "cal_fut": torch.as_tensor(v.cal_fut[s]),
                     "ruta_idx": torch.as_tensor(v.ruta_idx[s], dtype=torch.long),
                     "nivel": torch.as_tensor(v.nivel[s])}
                b = _a_device(b, self.device)
                f, g = _salida(self.modelo, b)
                mu, q = media_y_cuantil(f, g, b["nivel"])
                # MPS no soporta float64: primero a CPU y luego a double.
                mus.append(mu.cpu().double().numpy())
                qs.append(q.cpu().double().numpy())
        return np.concatenate(mus), np.concatenate(qs)


# ---------------------------------------------------------------------------
# Entrenamiento completo
# ---------------------------------------------------------------------------
@dataclass
class ResultadoEntrenamiento:
    pronosticador: PronosticadorRed
    historial: list[dict] = field(default_factory=list)
    mejor_epoca: int = 0
    mejor_val: float = float("nan")
    segundos: float = 0.0
    segundos_por_epoca: float = 0.0
    n_parametros: int = 0


def _hp_dataloader(ds: Dataset, batch_size: int, shuffle: bool, semilla: int) -> DataLoader:
    g = torch.Generator().manual_seed(int(semilla))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, generator=g if shuffle else None)


def entrenar(familia: str, hp: dict, cfg_m1: dict, cfg_datos: dict, panel: Panel, semilla: int,
             device: torch.device | str = "cpu", verbose: bool = False,
             ventanas_train: Ventanas | None = None, ventanas_val: Ventanas | None = None) -> ResultadoEntrenamiento:
    """Entrena una red con parada temprana sobre la validación y devuelve la mejor época.

    Reconstruye las ventanas si ``hp['window']`` difiere de la configuración. Todo el aleatorio (inicialización,
    barajado, dropout) sale de ``semilla``.
    """
    device = torch.device(device) if isinstance(device, str) else device
    cfg = cfg_con_ventana(cfg_m1, hp["window"]) if int(hp["window"]) != int(cfg_m1["window"]) else cfg_m1
    tr = ventanas_train or ventanas_particion(panel, cfg, cfg_datos, "train")
    va = ventanas_val or ventanas_particion(panel, cfg, cfg_datos, "val", paso_dias=1)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("No hay ventanas de entrenamiento o de validación con esta configuración.")

    fijar_semillas(semilla)
    modelo = construir_modelo(familia, hp, n_rutas=len(panel.rutas), n_cal=len(panel.nombres_cal),
                              horizonte=int(hp["horizon"])).to(device)
    n_par = contar_parametros(modelo)
    perdida = PerdidaCombinada(tau=float(hp["quantile"]), peso_cuantil=float(hp["quantile_weight"]))
    reg = hp["regularization"]
    optimizer = torch.optim.Adam(modelo.parameters(), lr=float(hp["lr"]), weight_decay=float(reg["weight_decay"]))
    dl_tr = _hp_dataloader(DatasetVentanas(tr), hp["batch_size"], True, semilla)
    dl_va = _hp_dataloader(DatasetVentanas(va), 512, False, semilla)

    mejor, mejor_estado, mejor_ep, sin_mejora = float("inf"), None, 0, 0
    historial: list[dict] = []
    t0 = time.perf_counter()
    for ep in range(1, int(hp["epochs"]) + 1):
        l_tr = entrenar_epoca(dl_tr, modelo, perdida, optimizer, device)
        v = validar(dl_va, modelo, perdida, device)
        historial.append({"epoca": ep, "train": l_tr, **{f"val_{k}": x for k, x in v.items()}})
        if verbose:
            print(f"época {ep:3d}  train {l_tr:.4f}  val {v['total']:.4f} (poisson {v['poisson']:.4f}, pinball {v['pinball']:.4f})")
        if v["total"] < mejor - 1e-6:
            mejor, mejor_ep, sin_mejora = v["total"], ep, 0
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= int(hp["patience"]):
                break
    segundos = time.perf_counter() - t0
    modelo.load_state_dict(mejor_estado)
    meta = {"familia": familia, "semilla": int(semilla), "dispositivo": str(device), "mejor_epoca": mejor_ep,
            "mejor_val_perdida": mejor, "epocas_corridas": len(historial), "n_parametros": n_par,
            "segundos_entrenamiento": segundos}
    return ResultadoEntrenamiento(
        pronosticador=PronosticadorRed(modelo, familia, hp, device, meta), historial=historial,
        mejor_epoca=mejor_ep, mejor_val=mejor, segundos=segundos,
        segundos_por_epoca=segundos / max(len(historial), 1), n_parametros=n_par)


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def guardar_modelo(p: PronosticadorRed, nombres_cal: list[str], ruta_base: str | Path) -> tuple[Path, Path]:
    """Escribe ``<base>.pt`` (``state_dict``) y ``<base>.json`` (hiperparámetros y metadatos)."""
    base = Path(ruta_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    pt, js = base.with_suffix(".pt"), base.with_suffix(".json")
    torch.save({k: v.detach().cpu() for k, v in p.modelo.state_dict().items()}, pt)
    js.write_text(json.dumps({"familia": p.familia, "hp": p.hp, "n_cal": len(nombres_cal),
                              "nombres_cal": nombres_cal, "meta": p.meta}, indent=2, ensure_ascii=False,
                             default=float), encoding="utf-8")
    return pt, js


def cargar_modelo(ruta_base: str | Path, n_rutas: int, device: torch.device | str = "cpu") -> PronosticadorRed:
    """Reconstruye una red guardada con ``guardar_modelo``. No entrena ni usa internet."""
    base = Path(ruta_base)
    d = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    device = torch.device(device) if isinstance(device, str) else device
    modelo = construir_modelo(d["familia"], d["hp"], n_rutas=n_rutas, n_cal=int(d["n_cal"]),
                              horizonte=int(d["hp"]["horizon"]))
    modelo.load_state_dict(torch.load(base.with_suffix(".pt"), map_location="cpu", weights_only=True))
    return PronosticadorRed(modelo, d["familia"], d["hp"], device, d.get("meta", {}))
