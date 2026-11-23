"""Propósito: Funciones de pérdida del Módulo 1 (conteos, cuantil alto y su combinación).
Responsable: Persona 2.

Parametrización del modelo (decisiones D1, D2 y D5 de docs/plan_persona2.md)
-----------------------------------------------------------------------------
La red produce, para cada día objetivo, dos números ``f`` y ``g`` (sin restricción):

    μ = ℓ · exp(f)                         media (pronóstico puntual), siempre > 0
    q = μ · exp(softplus(g))               cuantil alto P90, siempre q ≥ μ (no hay cruce de cuantiles)

donde ℓ es el nivel local (media de los 28 días previos al origen), que entra como *offset*.

Pérdida
-------
    L = L_Poisson(μ) + λ · L_pinball,τ(q),        τ = 0,9

* **Poisson** sobre la escala relativa ``y/ℓ``: ``exp(f) − (y/ℓ)·f`` (se omite el término constante
  ``log y!``). Es exactamente ``nn.PoissonNLLLoss(log_input=True)`` aplicada a ``f`` y ``y/ℓ``; equivale
  a la verosimilitud de Poisson de ``y`` con cada ventana pesada por 1/ℓ, de modo que las 10 rutas pesan
  parecido (el MASE promedia por ruta). Sigue siendo una cuasi-verosimilitud consistente para la media
  aunque haya sobredispersión (varianza > media), que sí la hay (ver ``sobredispersion.py``).
  Cuidado: con ``log_input=True`` la entrada es el LOGARITMO de la media (``f``), no la media.
* **Pinball** (Koenker y Bassett, 1978): ``L_τ(y, q) = max(τ(y − q), (τ − 1)(y − q))``, también sobre
  ``y/ℓ``. Con τ = 0,9, quedarse corto (q < y) cuesta 9 veces más que pasarse en la misma cantidad.
  Si se invierte el signo de (y − q) se estima el P10 en vez del P90.
* Ambas se promedian solo sobre los días con ``mask = 1`` (observados, no imputados, fuera del Paro Nacional).

Opcional (ablación, no se usa por defecto): binomial negativa tipo NB2 (``var = μ + α μ²``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

TAU_POR_DEFECTO = 0.9


# ---------------------------------------------------------------------------
# Parametrización
# ---------------------------------------------------------------------------
def media_y_cuantil(f: torch.Tensor, g: torch.Tensor, nivel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve ``(μ, q)`` en pasajeros a partir de las salidas ``f`` y ``g`` y el nivel ℓ.

    ``nivel`` tiene forma (N,) o (N, 1). Garantiza ``q ≥ μ > 0``.
    """
    ell = nivel.reshape(-1, 1) if nivel.dim() == 1 else nivel
    mu = ell * torch.exp(f)
    q = mu * torch.exp(F.softplus(g))
    return mu, q


# ---------------------------------------------------------------------------
# Pérdidas elemento a elemento (devuelven un tensor del mismo tamaño que la entrada)
# ---------------------------------------------------------------------------
def poisson_nll(f: torch.Tensor, y_rel: torch.Tensor) -> torch.Tensor:
    """Poisson NLL con enlace log: ``exp(f) − y_rel·f``. ``f`` es log de la media relativa (log μ/ℓ)."""
    return F.poisson_nll_loss(f, y_rel, log_input=True, full=False, reduction="none")


def pinball(y: torch.Tensor, q: torch.Tensor, tau: float = TAU_POR_DEFECTO) -> torch.Tensor:
    """Pérdida cuantílica ``max(τ(y − q), (τ − 1)(y − q))``."""
    d = y - q
    return torch.maximum(tau * d, (tau - 1.0) * d)


def nb2_nll(y: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Binomial negativa NB2 (``var = μ + α μ²``), NLL completa.

    ``−[lgamma(y+1/α) − lgamma(1/α) − lgamma(y+1) + (1/α)·log(1/(1+αμ)) + y·log(αμ/(1+αμ))]``.
    Equivale a ``scipy.stats.nbinom(n=1/α, p=1/(1+αμ))`` y a ``torch.distributions.NegativeBinomial``
    con ``total_count = 1/α`` y ``probs = αμ/(1+αμ)`` (en torch ``probs`` es la prob. de "éxito" y la
    media es ``total_count·probs/(1−probs)``; ver tests/test_m1_perdidas.py).
    """
    r = 1.0 / alpha
    ll = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
          + r * torch.log(r / (r + mu)) + y * torch.log(mu / (r + mu)))
    return -ll


# ---------------------------------------------------------------------------
# Pérdida combinada
# ---------------------------------------------------------------------------
class PerdidaCombinada(nn.Module):
    """``L = L_Poisson + λ · L_pinball`` promediada sobre los días válidos.

    Entradas (todas de forma (N, H)): ``f``, ``g`` salidas de la red; ``nivel`` (N,) = ℓ; ``y`` en pasajeros;
    ``mask`` en {0, 1}. Devuelve ``(total, poisson, pinball)`` como escalares.
    """

    def __init__(self, tau: float = TAU_POR_DEFECTO, peso_cuantil: float = 1.0):
        super().__init__()
        if not 0.0 < tau < 1.0:
            raise ValueError("tau debe estar en (0, 1).")
        if peso_cuantil < 0:
            raise ValueError("peso_cuantil (λ) no puede ser negativo.")
        self.tau = float(tau)
        self.peso_cuantil = float(peso_cuantil)

    def forward(self, f: torch.Tensor, g: torch.Tensor, nivel: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ell = nivel.reshape(-1, 1)
        y_rel = y / ell
        q_rel = torch.exp(f) * torch.exp(F.softplus(g))           # q/ℓ
        n = mask.sum().clamp_min(1.0)
        l_poisson = (poisson_nll(f, y_rel) * mask).sum() / n
        l_pinball = (pinball(y_rel, q_rel, self.tau) * mask).sum() / n
        return l_poisson + self.peso_cuantil * l_pinball, l_poisson, l_pinball
