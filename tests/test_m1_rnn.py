"""Pruebas del codificador recurrente (LSTM/GRU) del Módulo 1 (Persona 2).

Ejecutar:  python -m pytest tests/test_m1_rnn.py -q
"""
import numpy as np
import pytest
import torch

from common.config_loader import cargar_yaml
from m1_demanda import ventanas as vt
from m1_demanda.data.utils import cargar_config
from m1_demanda.evaluate import evaluar_modelo
from m1_demanda.losses import PerdidaCombinada, media_y_cuantil
from m1_demanda.nn_models import RNNDemanda, construir_modelo, contar_parametros
from m1_demanda.train import DatasetVentanas, cargar_modelo, entrenar, guardar_modelo, hp_desde_config


@pytest.fixture(scope="module")
def cfg_datos():
    return cargar_config()


@pytest.fixture(scope="module")
def cfg_m1():
    return cargar_yaml("configs/m1_demanda.yaml")


@pytest.fixture(scope="module")
def panel(cfg_m1, cfg_datos):
    return vt.cargar_panel(cfg_m1, cfg_datos)


def _entradas(b=4, L=28, H=30, F=27, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, L, generator=g), torch.randn(b, L, F, generator=g), torch.randn(b, H, F, generator=g),
            torch.randint(0, 10, (b,), generator=g))


@pytest.mark.parametrize("celda", ["gru", "lstm"])
def test_dimensiones_y_q_mayor_o_igual_que_mu(celda):
    m = RNNDemanda(10, 28, 30, 27, celda=celda, oculto=16, unidades=32)
    x, ch, cf, r = _entradas()
    f, g = m(x, ch, cf, r)
    assert f.shape == (4, 30) and g.shape == (4, 30)
    mu, q = media_y_cuantil(f, g, torch.rand(4) * 400 + 10)
    assert (mu > 0).all() and (q >= mu).all()


def test_conteo_de_parametros_a_mano():
    ent, hid, uni, H, F = 1 + 27, 32, 64, 30, 27
    cabezas = 2 * (uni * H + H)
    resto = 10 * 3 + (hid + H * F + 3) * uni + uni + cabezas
    gru = RNNDemanda(10, 28, 30, 27, "gru", hid, 1, 3, uni)
    lstm = RNNDemanda(10, 28, 30, 27, "lstm", hid, 1, 3, uni)
    assert contar_parametros(gru) == 3 * (ent * hid + hid * hid + 2 * hid) + resto
    assert contar_parametros(lstm) == 4 * (ent * hid + hid * hid + 2 * hid) + resto
    assert contar_parametros(gru) == sum(p.numel() for p in gru.parameters())


@pytest.mark.parametrize("celda", ["gru", "lstm"])
def test_batch_first_cada_muestra_es_independiente_del_lote(celda):
    """Si el codificador usara batch_first=False mezclaría el lote con el tiempo; el resultado de una muestra
    no puede depender de las demás ni del tamaño del lote (modo evaluación)."""
    m = RNNDemanda(10, 28, 30, 27, celda, 16, 1, 3, 32).eval()
    x, ch, cf, r = _entradas(b=6)
    f_lote, g_lote = m(x, ch, cf, r)
    for i in range(6):
        f1, g1 = m(x[i:i + 1], ch[i:i + 1], cf[i:i + 1], r[i:i + 1])
        torch.testing.assert_close(f1[0], f_lote[i], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(g1[0], g_lote[i], atol=1e-5, rtol=1e-5)
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(1))
    f_perm, _ = m(x[perm], ch[perm], cf[perm], r[perm])
    torch.testing.assert_close(f_perm, f_lote[perm], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("celda", ["gru", "lstm"])
def test_el_calendario_futuro_no_cambia_el_estado_del_codificador(celda):
    m = RNNDemanda(10, 28, 30, 27, celda, 16, 2, 3, 32).eval()
    x, ch, cf, r = _entradas()
    estado = m.codificar(x, ch)
    torch.testing.assert_close(estado, m.codificar(x, ch))                     # no depende de cal_fut ni de la ruta
    # La cabeza de la media arranca en cero a propósito (f = 0, μ = ℓ); se perturba para comprobar la dependencia.
    torch.nn.init.normal_(m.cabezas.media.weight, std=0.1)
    f1, _ = m(x, ch, cf, r)
    f2, _ = m(x, ch, cf + 5.0, r)
    assert not torch.allclose(f1, f2)                                          # el pronóstico sí usa el futuro
    assert estado.shape == (4, 16)


def test_el_historial_afecta_al_estado():
    m = RNNDemanda(10, 28, 30, 27, "gru", 16, 1, 3, 32).eval()
    x, ch, _, _ = _entradas()
    assert not torch.allclose(m.codificar(x, ch), m.codificar(x + 1.0, ch))


def test_parametros_invalidos():
    with pytest.raises(ValueError, match="celda"):
        RNNDemanda(10, 28, 30, 27, celda="rnn")
    with pytest.raises(ValueError):
        RNNDemanda(10, 28, 30, 27, oculto=0)


def test_construir_modelo_desde_hiperparametros(cfg_m1):
    hp = hp_desde_config(cfg_m1, rnn={"cell": "lstm", "hidden": 8})
    m = construir_modelo("rnn", hp, n_rutas=10, n_cal=27, horizonte=30)
    assert isinstance(m, RNNDemanda) and m.celda == "lstm" and m.codificador.hidden_size == 8
    assert hp["rnn"]["layers"] == cfg_m1["rnn"]["layers"]                   # los cambios se fusionan, no reemplazan
    with pytest.raises(ValueError):
        construir_modelo("transformer", hp, 10, 27, 30)


def test_sobreajusta_un_lote_pequeno(panel, cfg_m1, cfg_datos):
    torch.manual_seed(0)
    v = vt.ventanas_particion(panel, cfg_m1, cfg_datos, "val").subconjunto(slice(0, 24))
    b = DatasetVentanas(v).t
    m = RNNDemanda(len(panel.rutas), 28, 30, len(panel.nombres_cal), "gru", 32, 1, 3, 64, dropout=0.0)
    perdida = PerdidaCombinada(0.9, 1.0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def paso():
        f, g = m(b["x_hist"], b["cal_hist"], b["cal_fut"], b["ruta_idx"])
        return perdida(f, g, b["nivel"], b["y"], b["mask"])

    r = (b["y"] / b["nivel"][:, None]).clamp_min(1e-9)
    piso = float(((r * (1 - torch.log(r))) * b["mask"]).sum() / b["mask"].sum())
    inicial = [float(x.detach()) for x in paso()]
    for _ in range(1500):
        opt.zero_grad()
        paso()[0].backward()
        opt.step()
    final = [float(x.detach()) for x in paso()]
    assert final[1] >= piso - 1e-4
    assert (inicial[1] - final[1]) / (inicial[1] - piso) > 0.8
    assert final[2] < 0.5 * inicial[2]


@pytest.mark.parametrize("celda", ["gru", "lstm"])
def test_entrena_evalua_y_se_recarga(celda, panel, cfg_m1, cfg_datos, tmp_path):
    hp = hp_desde_config(cfg_m1, epochs=3, patience=5, rnn={"cell": celda, "hidden": 8})
    r = entrenar("rnn", hp, cfg_m1, cfg_datos, panel, semilla=11, device="cpu")
    assert r.historial[-1]["train"] <= r.historial[0]["train"] * 1.05
    tabla, hor, v, mu, q = evaluar_modelo(r.pronosticador, panel, cfg_m1, cfg_datos, "val")
    assert np.isfinite(mu).all() and (q >= mu).all() and len(hor) == 30
    guardar_modelo(r.pronosticador, panel.nombres_cal, tmp_path / "rnn")
    otro = cargar_modelo(tmp_path / "rnn", n_rutas=len(panel.rutas))
    mu2, q2 = otro.predecir(panel, v)
    np.testing.assert_allclose(mu, mu2, rtol=1e-6)
    np.testing.assert_allclose(q, q2, rtol=1e-6)
    assert otro.n_parametros == r.n_parametros and otro.hp["rnn"]["cell"] == celda


def test_misma_semilla_mismo_resultado(panel, cfg_m1, cfg_datos):
    hp = hp_desde_config(cfg_m1, epochs=2, patience=5, rnn={"hidden": 8})
    a = entrenar("rnn", hp, cfg_m1, cfg_datos, panel, semilla=3, device="cpu")
    b = entrenar("rnn", hp, cfg_m1, cfg_datos, panel, semilla=3, device="cpu")
    for pa, pb in zip(a.pronosticador.modelo.parameters(), b.pronosticador.modelo.parameters()):
        torch.testing.assert_close(pa, pb)
