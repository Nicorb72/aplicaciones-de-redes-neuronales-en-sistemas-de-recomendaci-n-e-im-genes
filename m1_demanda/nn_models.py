"""Propósito: Redes neuronales del Módulo 1 (MLP y codificador recurrente) con dos cabezas: media y P90.
Responsable: Persona 2.

Entradas de cada modelo (por ventana; ver ``ventanas.py``), todas conocidas en el origen ``o``:

* ``x_hist``   (N, L)      log((y+1)/(ℓ+1)) de los últimos L días antes de ``o``.
* ``cal_hist`` (N, L, F)   calendario de esos L días.
* ``cal_fut``  (N, H, F)   calendario de los H días objetivo (se conoce por ley).
* ``ruta_idx`` (N,)        índice de ruta (embedding: el patrón semanal y el efecto de los puentes
                           dependen de la ruta; docs/m1_datos_y_eda.md, H1 y H2).

Salidas: ``f`` y ``g`` de forma (N, H) (sin restricción). Con el nivel local ℓ:
``μ = ℓ·exp(f)`` y ``q = μ·exp(softplus(g)) ≥ μ`` (``losses.media_y_cuantil``).

Modelo global (una sola red para las 10 rutas) con embedding de ruta (decisión D3). Estilo del curso:
clases ``nn.Module`` con ``nn.Sequential`` y conteo de parámetros con ``sum(p.numel() ...)``.
"""
from __future__ import annotations

import torch
from torch import nn

# g = -1  ->  softplus(g) ≈ 0,31  ->  q/μ ≈ 1,37 al inicio (un P90 razonable, no el doble de la media)
SESGO_INICIAL_G = -1.0


def contar_parametros(modelo: nn.Module) -> int:
    """Número de parámetros ENTRENABLES (solo ``requires_grad``; incluye el embedding)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


class _Cabezas(nn.Module):
    """Dos capas lineales que producen ``f`` (media) y ``g`` (cuantil) para los H días objetivo."""

    def __init__(self, entrada: int, horizonte: int):
        super().__init__()
        self.media = nn.Linear(entrada, horizonte)
        self.cuantil = nn.Linear(entrada, horizonte)
        # Arranque con f ≈ 0 (μ ≈ ℓ, "el mismo nivel de siempre") y un P90 moderado.
        nn.init.zeros_(self.media.weight)
        nn.init.zeros_(self.media.bias)
        nn.init.normal_(self.cuantil.weight, std=0.01)
        nn.init.constant_(self.cuantil.bias, SESGO_INICIAL_G)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.media(h), self.cuantil(h)


class MLPDemanda(nn.Module):
    """MLP con ventanas de rezagos: aplana historial + calendario + embedding de ruta.

    ``capas`` capas ocultas de ``unidades`` neuronas con ReLU y dropout. Parámetros: ``dim_embedding``,
    ``capas``, ``unidades``, ``dropout``. Con ``usar_cal_hist=False`` no se entrega el calendario del
    historial (ablación).
    """

    def __init__(self, n_rutas: int, ventana: int, horizonte: int, n_cal: int, dim_embedding: int = 3,
                 capas: int = 2, unidades: int = 64, dropout: float = 0.1, usar_cal_hist: bool = True):
        super().__init__()
        if capas < 1 or unidades < 1:
            raise ValueError("capas y unidades deben ser al menos 1.")
        self.ventana, self.horizonte, self.n_cal = ventana, horizonte, n_cal
        self.usar_cal_hist = usar_cal_hist
        self.embedding = nn.Embedding(n_rutas, dim_embedding)
        entrada = ventana + (ventana * n_cal if usar_cal_hist else 0) + horizonte * n_cal + dim_embedding
        bloques: list[nn.Module] = []
        previa = entrada
        for _ in range(capas):
            bloques += [nn.Linear(previa, unidades), nn.ReLU(), nn.Dropout(dropout)]
            previa = unidades
        self.tronco = nn.Sequential(*bloques)
        self.cabezas = _Cabezas(unidades, horizonte)

    def forward(self, x_hist: torch.Tensor, cal_hist: torch.Tensor, cal_fut: torch.Tensor,
                ruta_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        partes = [x_hist]
        if self.usar_cal_hist:
            partes.append(cal_hist.flatten(1))
        partes += [cal_fut.flatten(1), self.embedding(ruta_idx)]
        return self.cabezas(self.tronco(torch.cat(partes, dim=1)))


class RNNDemanda(nn.Module):
    """Codificador recurrente (LSTM o GRU) del pasado + calendario futuro + embedding de ruta.

    1. El codificador recorre los L días anteriores al origen. En cada paso recibe ``[x_t, calendario_t]``
       (1 + F valores) y se usa su ESTADO OCULTO FINAL de la última capa (``h_n``). Con ``batch_first=True``
       la entrada es (N, L, 1+F); ``nn.LSTM`` y ``nn.GRU`` usan ``batch_first=False`` por defecto, y olvidarlo
       mezcla silenciosamente el lote con el tiempo.
    2. Ese estado se concatena con el calendario futuro aplanado (H·F) y el embedding de ruta.
    3. Una capa densa (``unidades``, ReLU, dropout) y las dos cabezas ``f`` (media) y ``g`` (P90).

    El calendario futuro NO entra al codificador, así que cambiarlo no altera el estado del pasado
    (``codificar`` lo expone para comprobarlo). Hiperparámetros: ``celda`` ("lstm" o "gru"), ``oculto``,
    ``capas_rnn``, ``dim_embedding``, ``unidades`` y ``dropout``.
    """

    def __init__(self, n_rutas: int, ventana: int, horizonte: int, n_cal: int, celda: str = "gru",
                 oculto: int = 32, capas_rnn: int = 1, dim_embedding: int = 3, unidades: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        celda = celda.lower()
        if celda not in ("lstm", "gru"):
            raise ValueError("celda debe ser 'lstm' o 'gru'.")
        if oculto < 1 or capas_rnn < 1 or unidades < 1:
            raise ValueError("oculto, capas_rnn y unidades deben ser al menos 1.")
        self.celda, self.ventana, self.horizonte, self.n_cal = celda, ventana, horizonte, n_cal
        clase = nn.LSTM if celda == "lstm" else nn.GRU
        self.codificador = clase(input_size=1 + n_cal, hidden_size=oculto, num_layers=capas_rnn, batch_first=True,
                                 dropout=dropout if capas_rnn > 1 else 0.0)
        self.embedding = nn.Embedding(n_rutas, dim_embedding)
        self.tronco = nn.Sequential(nn.Linear(oculto + horizonte * n_cal + dim_embedding, unidades),
                                    nn.ReLU(), nn.Dropout(dropout))
        self.cabezas = _Cabezas(unidades, horizonte)

    def codificar(self, x_hist: torch.Tensor, cal_hist: torch.Tensor) -> torch.Tensor:
        """Estado oculto final (N, oculto) de la última capa tras recorrer el historial."""
        secuencia = torch.cat([x_hist.unsqueeze(-1), cal_hist], dim=-1)          # (N, L, 1 + F)
        _, estado = self.codificador(secuencia)
        h_n = estado[0] if self.celda == "lstm" else estado                        # LSTM devuelve (h_n, c_n)
        return h_n[-1]

    def forward(self, x_hist: torch.Tensor, cal_hist: torch.Tensor, cal_fut: torch.Tensor,
                ruta_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([self.codificar(x_hist, cal_hist), cal_fut.flatten(1), self.embedding(ruta_idx)], dim=1)
        return self.cabezas(self.tronco(z))


def construir_modelo(familia: str, hp: dict, n_rutas: int, n_cal: int, horizonte: int) -> nn.Module:
    """Crea la red de una familia (``"mlp"`` o ``"rnn"``) con los hiperparámetros ``hp``."""
    if familia == "rnn":
        r = hp["rnn"]
        return RNNDemanda(
            n_rutas=n_rutas, ventana=int(hp["window"]), horizonte=horizonte, n_cal=n_cal, celda=str(r["cell"]),
            oculto=int(r["hidden"]), capas_rnn=int(r["layers"]), dim_embedding=int(hp["route_embedding_dim"]),
            unidades=int(hp["units"]), dropout=float(hp["regularization"]["dropout"]),
        )
    if familia == "mlp":
        return MLPDemanda(
            n_rutas=n_rutas, ventana=int(hp["window"]), horizonte=horizonte, n_cal=n_cal,
            dim_embedding=int(hp["route_embedding_dim"]), capas=int(hp["layers"]), unidades=int(hp["units"]),
            dropout=float(hp["regularization"]["dropout"]), usar_cal_hist=bool(hp.get("use_cal_hist", True)),
        )
    raise ValueError(f"Familia {familia!r} desconocida (use 'mlp' o 'rnn').")
