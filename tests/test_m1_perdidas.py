"""Pruebas de losses.py y sobredispersion.py (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_perdidas.py -q
"""
import numpy as np
import pytest
import torch
from scipy import stats
from torch import nn

from m1_demanda.losses import PerdidaCombinada, media_y_cuantil, nb2_nll, pinball, poisson_nll


# --- Poisson -----------------------------------------------------------------------
def test_poisson_coincide_con_la_formula_manual_y_con_nn_PoissonNLLLoss():
    f = torch.tensor([[-0.5, 0.0, 0.7]])
    y_rel = torch.tensor([[0.4, 1.0, 2.5]])
    manual = torch.exp(f) - y_rel * f
    torch.testing.assert_close(poisson_nll(f, y_rel), manual)
    ref = nn.PoissonNLLLoss(log_input=True, full=False, reduction="none")(f, y_rel)
    torch.testing.assert_close(poisson_nll(f, y_rel), ref)


def test_poisson_log_input_espera_el_logaritmo_de_la_media():
    """El mínimo de exp(f) − y·f está en f = log(y), no en f = y (error típico: pasar la media)."""
    y = torch.tensor(3.0)
    fs = torch.linspace(-2, 4, 6001)
    f_opt = fs[torch.argmin(torch.exp(fs) - y * fs)]
    assert f_opt.item() == pytest.approx(np.log(3.0), abs=1e-3)


def test_poisson_minimiza_en_la_media_de_los_datos():
    rng = np.random.default_rng(0)
    y = torch.tensor(rng.poisson(7.0, size=5000), dtype=torch.float64)
    f = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    opt = torch.optim.LBFGS([f], lr=1.0, max_iter=200)

    def cierre():
        opt.zero_grad()
        perdida = poisson_nll(f.expand_as(y), y).mean()
        perdida.backward()
        return perdida

    opt.step(cierre)
    assert float(torch.exp(f.detach())) == pytest.approx(float(y.mean()), rel=1e-4)


# --- Pinball ------------------------------------------------------------------------
def test_pinball_a_mano_y_signo():
    y, q = torch.tensor([10.0, 10.0]), torch.tensor([8.0, 12.0])
    torch.testing.assert_close(pinball(y, q, 0.9), torch.tensor([1.8, 0.2]))
    # con el signo invertido se obtendría el cuantil 0,1: aquí el costo de quedarse corto sería 0,2
    torch.testing.assert_close(pinball(y, q, 0.1), torch.tensor([0.2, 1.8]))


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_minimizar_pinball_sobre_una_constante_da_el_cuantil_empirico(tau):
    rng = np.random.default_rng(1)
    y = torch.tensor(rng.gamma(2.0, 100.0, size=4000))
    candidatos = torch.linspace(float(y.min()), float(y.max()), 3000, dtype=y.dtype)
    perdidas = torch.stack([pinball(y, c.expand_as(y), tau).mean() for c in candidatos])
    assert float(candidatos[perdidas.argmin()]) == pytest.approx(float(np.quantile(y.numpy(), tau)), rel=0.02)


def test_p90_aprendido_con_pinball_cubre_90_por_ciento():
    """Prueba sintética del plan: la cobertura del cuantil aprendido debe ser ≈ 0,90 (con 0,10 sería signo invertido)."""
    rng = np.random.default_rng(2)
    y = torch.tensor(rng.poisson(rng.uniform(50, 400, size=20000)), dtype=torch.float32)
    q = torch.nn.Parameter(torch.full((1,), 100.0))
    opt = torch.optim.Adam([q], lr=1.0)
    for _ in range(600):
        opt.zero_grad()
        pinball(y, q.expand_as(y), 0.9).mean().backward()
        opt.step()
    cobertura = float((y <= q.detach()).float().mean())
    assert cobertura == pytest.approx(0.9, abs=0.02)


# --- media y cuantil -------------------------------------------------------------------
def test_media_y_cuantil_garantiza_q_mayor_o_igual_que_mu_y_positivos():
    f = torch.randn(50, 30) * 3
    g = torch.randn(50, 30) * 10                       # incluye valores muy negativos
    nivel = torch.rand(50) * 1000 + 1
    mu, q = media_y_cuantil(f, g, nivel)
    assert (mu > 0).all() and (q >= mu).all()
    torch.testing.assert_close(mu, nivel[:, None] * torch.exp(f))


# --- pérdida combinada --------------------------------------------------------------------
def test_perdida_combinada_suma_y_respeta_la_mascara():
    torch.manual_seed(0)
    f, g = torch.randn(4, 5), torch.randn(4, 5)
    nivel = torch.tensor([100.0, 200.0, 300.0, 400.0])
    y = torch.rand(4, 5) * 500
    mask = torch.tensor([[1, 1, 0, 1, 0]] * 4, dtype=torch.float32)
    perd = PerdidaCombinada(tau=0.9, peso_cuantil=2.0)
    total, lp, lq = perd(f, g, nivel, y, mask)
    assert float(total) == pytest.approx(float(lp) + 2.0 * float(lq))
    # los días con máscara 0 no influyen: cambiarlos no altera la pérdida
    y2 = y.clone()
    y2[:, [2, 4]] = 1e6
    total2, _, _ = perd(f, g, nivel, y2, mask)
    torch.testing.assert_close(total, total2)
    # a mano para un elemento: ℓ = 100, y = 50 -> y_rel 0,5
    l1, lp1, _ = perd(torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([100.0]), torch.tensor([[50.0]]), torch.ones(1, 1))
    assert float(lp1) == pytest.approx(1.0)       # exp(0) − 0,5·0 = 1
    assert float(l1) >= float(lp1)


def test_perdida_combinada_pinball_a_mano():
    # f = 0 -> μ/ℓ = 1; g = 0 -> softplus(0) = ln 2 -> q/ℓ = 1·exp(ln 2) = 2; y_rel = 3 -> pinball = 0,9·(3 − 2)
    perd = PerdidaCombinada(0.9, 1.0)
    _, _, lq = perd(torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([100.0]), torch.tensor([[300.0]]), torch.ones(1, 1))
    assert float(lq) == pytest.approx(0.9)


def test_perdida_combinada_sin_dias_validos_no_da_nan():
    perd = PerdidaCombinada()
    total, _, _ = perd(torch.zeros(2, 3), torch.zeros(2, 3), torch.ones(2), torch.ones(2, 3), torch.zeros(2, 3))
    assert torch.isfinite(total) and float(total) == 0.0


def test_perdida_combinada_parametros_invalidos():
    with pytest.raises(ValueError):
        PerdidaCombinada(tau=1.5)
    with pytest.raises(ValueError):
        PerdidaCombinada(peso_cuantil=-1)


def test_gradientes_finitos():
    f = torch.randn(8, 30, requires_grad=True)
    g = torch.randn(8, 30, requires_grad=True)
    total, _, _ = PerdidaCombinada()(f, g, torch.rand(8) * 500 + 10, torch.rand(8, 30) * 800, torch.ones(8, 30))
    total.backward()
    assert torch.isfinite(f.grad).all() and torch.isfinite(g.grad).all()


# --- binomial negativa (ablación opcional) ----------------------------------------------------
def test_nb2_coincide_con_scipy():
    mu, alpha = 120.0, 0.3
    y = np.arange(0, 400, 7)
    esperado = -stats.nbinom.logpmf(y, n=1 / alpha, p=1 / (1 + alpha * mu))
    obtenido = nb2_nll(torch.tensor(y, dtype=torch.float64), torch.tensor(mu, dtype=torch.float64),
                       torch.tensor(alpha, dtype=torch.float64))
    np.testing.assert_allclose(obtenido.numpy(), esperado, rtol=1e-9)


def test_torch_negativebinomial_probs_es_la_probabilidad_de_exito():
    """Punto 5 del checklist: en torch, media = total_count·probs/(1 − probs) (no `probs` = prob. de fallo)."""
    mu, alpha = 120.0, 0.3
    r = 1 / alpha
    probs = alpha * mu / (1 + alpha * mu)
    d = torch.distributions.NegativeBinomial(total_count=torch.tensor(r, dtype=torch.float64),
                                             probs=torch.tensor(probs, dtype=torch.float64))
    assert float(d.mean) == pytest.approx(mu)
    y = torch.tensor([0.0, 10.0, 120.0, 300.0], dtype=torch.float64)
    ref = nb2_nll(y, torch.tensor(mu, dtype=torch.float64), torch.tensor(alpha, dtype=torch.float64))
    torch.testing.assert_close(-d.log_prob(y), ref)


# --- sobredispersión ------------------------------------------------------------------------------
def test_tabla_de_sobredispersion_indica_conteo_muy_sobredispersado():
    import pandas as pd

    from m1_demanda.data.utils import RAIZ
    t = pd.read_csv(RAIZ / "reports/tables/m1_sobredispersion.csv", dtype={"ruta_id": str})
    assert len(t) == 11 and t.iloc[-1]["ruta_id"] == "TODAS"
    por_ruta = t.iloc[:-1]
    assert (por_ruta["razon_mediana"] > 1).all()
    assert (por_ruta["dispersion_pearson_glm"] > 1).all()
