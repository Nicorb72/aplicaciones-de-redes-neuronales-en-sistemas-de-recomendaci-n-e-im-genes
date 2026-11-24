# Módulo 3 — Recomendación turística: comandos actualizados

Ejecutar desde la raíz del repositorio (no desde `m3_recomendacion/`).

## Preparación

```powershell
python -m m3_recomendacion.data.explore
python -m m3_recomendacion.data.prepare
python -m m3_recomendacion.data.split
```

## Entrenamiento y evaluación

```powershell
python -m m3_recomendacion.experiments.train --embedding-dim 8 --hidden-dim 16 --dropout 0.1 --lr 0.001 --epochs 100 --seed 42 --patience 15 --k 3
python -m m3_recomendacion.evaluation.metrics_by_k
python -m m3_recomendacion.evaluation.evaluate_user_groups
python -m m3_recomendacion.models.cold_start
```

## Experimentación y figuras

```powershell
python -m m3_recomendacion.experiments.search_neural
python -m m3_recomendacion.experiments.stability_neural
python -m m3_recomendacion.experiments.ablation_stability
python -m m3_recomendacion.analysis.embedding_visualization
python -m m3_recomendacion.analysis.recommendation_concentration
python -m m3_recomendacion.analysis.campaign_simulation
```

Las rutas de datos permanecen en `data/m3_recomendacion/{raw,processed}` y las figuras en `reports/figures/m3`.

Se corrigieron únicamente importaciones y raíz del repositorio respecto de la reestructuración. Las pruebas ejecutadas incluyen importación de todos los módulos, entrenamiento de C5, evaluación por K y cold-start.
