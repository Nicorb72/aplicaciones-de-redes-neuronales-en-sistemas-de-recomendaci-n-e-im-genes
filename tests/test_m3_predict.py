"""Pruebas rápidas del contrato de recomendación del Módulo 3 (Persona 4).

Ejecutar desde la raíz del repositorio:
    python -m pytest tests/test_m3_predict.py -q

Utiliza datos sintéticos en memoria: no lee ni modifica train/val/test.csv,
no utiliza los resultados del conjunto de prueba y no entrena el modelo C5.
El contrato con app/ se comprobará cuando exista el adaptador de integración.
"""

import math

import pandas as pd
import pytest

from m3_recomendacion.models.popularity import PopularityRecommender
from m3_recomendacion.models.collaborative import CollaborativeRecommender
from m3_recomendacion.models.cold_start import ColdStartRecommender
from m3_recomendacion.experiments.train import NeuralRecommendationSystem


@pytest.fixture
def datos():
    catalogo = pd.DataFrame({
        "ItemID": [1, 2, 3, 4, 5],
        "Name": ["Goa", "Jaipur", "Kerala", "Leh", "Taj"],
        "Type": ["Beach", "City", "Nature", "Adventure", "Culture"],
    })
    usuarios = pd.DataFrame({
        "UserID": [10, 20, 30, 99],
        "Preferences": ["Nature", "City", "Nature", "City"],
    })
    train = pd.DataFrame({
        "UserID": [10, 10, 20, 20, 30, 30],
        "ItemID": [1, 2, 2, 3, 1, 4],
        "Rating": [5, 2, 5, 4, 4, 5],
        "Positiva": [True, False, True, True, True, True],
    })
    return train, catalogo, usuarios


def comprobar_ranking_puntuado(recomendaciones, catalogo, vistos, k):
    """Contrato vigente: [(ItemID, puntuación), ...] ordenado desc."""
    assert isinstance(recomendaciones, list)
    assert len(recomendaciones) <= k
    ids = []
    scores = []
    catalogo_ids = set(catalogo["ItemID"].astype(int))
    for item_id, score in recomendaciones:
        assert isinstance(item_id, int)
        assert item_id in catalogo_ids
        assert item_id not in vistos
        assert isinstance(score, (float, int))
        assert math.isfinite(score)
        ids.append(item_id)
        scores.append(score)
    assert len(ids) == len(set(ids)), "No debe repetir destinos"
    assert scores == sorted(scores, reverse=True), "El ranking debe estar ordenado"


def test_popularidad_excluye_vistos_y_respeta_k(datos):
    train, catalogo, _ = datos
    modelo = PopularityRecommender().fit(train, catalogo)
    recomendaciones = modelo.recommend(10, k=3)
    comprobar_ranking_puntuado(recomendaciones, catalogo, {1, 2}, 3)
    assert len(recomendaciones) == 3


def test_popularidad_es_determinista(datos):
    train, catalogo, _ = datos
    modelo = PopularityRecommender().fit(train, catalogo)
    assert modelo.recommend(10, k=3) == modelo.recommend(10, k=3)


def test_colaborativo_entrega_ranking_valido(datos):
    train, catalogo, _ = datos
    modelo = CollaborativeRecommender(epochs=2, seed=42).fit(train, catalogo)
    recomendaciones = modelo.recommend(10, k=3)
    comprobar_ranking_puntuado(recomendaciones, catalogo, {1, 2}, 3)
    assert len(recomendaciones) == 3


def test_colaborativo_rechaza_usuario_desconocido(datos):
    train, catalogo, _ = datos
    modelo = CollaborativeRecommender(epochs=1, seed=42).fit(train, catalogo)
    with pytest.raises(ValueError, match="Usuario desconocido"):
        modelo.recommend(99, k=3)


def test_neuronal_contrato_de_inferencia(datos):
    """Comprueba el cableado de predicción; no evalúa la calidad del modelo."""
    train, catalogo, usuarios = datos
    modelo = NeuralRecommendationSystem(
        embedding_dim=8, hidden_dim=16, dropout=0.1, seed=42
    )
    modelo.preparar(train, catalogo, usuarios)
    recomendaciones = modelo.recommend(10, k=3)
    comprobar_ranking_puntuado(recomendaciones, catalogo, {1, 2}, 3)
    assert len(recomendaciones) == 3


def test_neuronal_rechaza_usuario_desconocido(datos):
    train, catalogo, usuarios = datos
    modelo = NeuralRecommendationSystem(embedding_dim=8, hidden_dim=16)
    modelo.preparar(train, catalogo, usuarios)
    with pytest.raises(ValueError, match="Usuario desconocido"):
        modelo.recommend(99, k=3)


@pytest.mark.parametrize("strategy", ["global", "preferences"])
def test_cold_start_recomienda_usuario_nuevo(datos, strategy):
    train, catalogo, usuarios = datos
    modelo = ColdStartRecommender(strategy=strategy).fit(train, catalogo, usuarios)
    recomendaciones = modelo.recommend(99, k=3)
    assert isinstance(recomendaciones, list)
    assert len(recomendaciones) == 3
    assert len(set(recomendaciones)) == 3
    assert set(recomendaciones).issubset(set(catalogo["ItemID"]))
    assert recomendaciones == modelo.recommend(99, k=3)


def test_cold_start_rechaza_estrategia_invalida(datos):
    train, catalogo, usuarios = datos
    modelo = ColdStartRecommender(strategy="invalida").fit(train, catalogo, usuarios)
    with pytest.raises(ValueError, match="Estrategia desconocida"):
        modelo.recommend(99, k=3)
