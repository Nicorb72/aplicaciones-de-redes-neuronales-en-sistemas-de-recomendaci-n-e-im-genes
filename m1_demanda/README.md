# Módulo 1: Series de tiempo — Pronóstico de demanda y planeación de la operación

## Objetivo
Pronosticar los pasajeros por ruta y día para los próximos 30 días y, con ese pronóstico, dimensionar la flota y los conductores.

## Datos (Persona 1)
- **Fuente:** Superintendencia de Transporte, *Operación de pasajeros y despacho de vehículos en la modalidad de transporte de pasajeros por carretera*, datos.gov.co (`eh75-8ah6`). Municipios: DIVIPOLA del DANE (`gdxc-w37w`).
- **Rutas:** 5 corredores × 2 sentidos = 10 series: Villavicencio↔Bogotá, Girardot↔Bogotá, Pereira↔Armenia, Pasto↔Ipiales y Duitama↔Sogamoso.
- **Periodo válido:** 2021-01-01–2023-12-31, 2025-01-01–2025-11-29 y 2026-01-01–2026-03-30. La fuente casi no tiene datos de 2024 ni de diciembre de 2025.
- **Documentación completa** (caso de uso, diccionario, inconsistencias, limpieza, hipótesis): [`docs/m1_datos_y_eda.md`](../docs/m1_datos_y_eda.md).

```bash
make download-m1   # (opcional, requiere internet ~7 min) re-descarga los crudos agregados
make data-m1       # crudos versionados -> data/m1_demanda/processed  (<5 s, sin internet)
make eda-m1        # figuras reports/figures/m1/eda_*.png y tablas reports/tables/m1_*.csv
make test-m1       # pruebas del pipeline de datos
```

Todos los parámetros (rutas, umbrales de limpieza, particiones, calendario) están en `configs/m1_datos.yaml`.

| Script | Función |
|---|---|
| `data/download.py` | Descarga reproducible (agregación SoQL en el servidor) |
| `data/calendar_features.py` | Festivos (Ley 51/1983, Ley 2578/2026), puentes, Semana Santa, temporadas y elecciones |
| `data/clean.py` | Reglas de limpieza R1–R8 |
| `data/build_series.py` | Panel ruta × día + catálogo + mezcla de vehículos |
| `data/split.py` | Particiones temporales fijas y orígenes rodantes |
| `eda.py` | Análisis descriptivo y pruebas de hipótesis |

**Particiones:** entrenamiento 2021-01-01–2023-06-30 · validación 2023-07-01–2023-09-30 · prueba 2023-10-01–2023-12-31 (temporada alta) · post-hueco 2025-01-01–2026-03-30.

## Contrato con la app (`predict.py`, Persona 2 → Persona 5)
```python
from m1_demanda.predict import list_routes, forecast
list_routes()                              # ruta_id, nombre, corredor, sentido, ruta_regreso
df = forecast("50001-11001", quantile=0.9, # None -> planea con la media; 0.9 -> con el P90; otro -> ValueError
              op_params=None,              # {"target_occupancy": 0.8, "workday.max_conduccion_dia_h": 8}; clave desconocida -> ValueError
              origen=None)                 # None -> 2026-03-31 (día siguiente al último dato); o "2023-12-02" para reproducir
```
Columnas: `fecha, ruta_id, horizonte (1..30), pronostico, p90, base_plan, despachos, vehiculos, conductores, real`.
`vehiculos` y `conductores` son del **corredor** (ida y regreso comparten flota); `despachos` es de la ruta pedida.
`df.attrs`: `modelo, origen, ultimo_dato, flota_corredor, plantilla_conductores, advertencias, quantile, parametros_operacion`
(pandas marca `attrs` como experimental; si la app necesita algo más estable se cambia por `(df, resumen)`).
No reentrena, no usa internet y carga solo de `models/m1_demanda/`. Ejemplo para desarrollar la app sin cargar nada:
`tests/fixtures/m1_forecast_ejemplo.csv` (+ `_attrs.json`). Pruebas del contrato: `tests/test_m1_contrato.py`.

## Modelos (Persona 2)
Entorno: Python 3.12 en `.venv` (`uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt`); PyTorch 2.15.0. Dispositivo por defecto: **CPU** (medido: MPS es 3 veces más lento y no reproduce la misma semilla; `reports/tables/m1_benchmark_dispositivo.csv`).

Todos los modelos pronostican 30 días con **el mismo protocolo**: mismas particiones, ventanas sin fuga y métricas (MAE, RMSE, MASE, error por horizonte, cobertura del P90, pérdida cuantílica) con origen rodante. Solo se usan variables conocidas 30 días antes: el historial hasta el día anterior al origen y el calendario (festivos, puentes, temporadas y elecciones se fijan por ley).

| Modelo | Qué es | Parámetros | Presupuesto de búsqueda |
|---|---|---|---|
| `ingenuo` | Repite la última semana observada; P90 empírico calibrado en validación | 0 | 0 configuraciones |
| `glm` | GLM Poisson (enlace log) por ruta con rezagos del mismo día de la semana, calendario y horizonte | `n_parametros` (por ruta) | 1 configuración |
| `mlp` | MLP global con embedding de ruta y dos cabezas (media y P90) | ver `reports/tables/m1_comparacion.csv` | `search.budget` = 30, 1 semilla cada una |
| `rnn` | GRU/LSTM sobre el historial + calendario futuro + embedding de ruta; mismas dos cabezas | ídem | 30, 1 semilla cada una |

**Pérdida:** Poisson (`nn.PoissonNLLLoss(log_input=True)`) sobre `y/ℓ` para la media más pérdida cuantílica (τ = 0,9) para el P90, con `q = μ·exp(softplus(g)) ≥ μ`. Justificación y diagnóstico de sobredispersión: `docs/math_foundations.md` y `reports/tables/m1_sobredispersion.csv`.

```bash
python -m m1_demanda.run --mode evaluate --modelo ingenuo     # o glm; con --particiones val test post_hueco
python -m m1_demanda.run --mode train --modelo mlp            # una semilla, hiperparámetros del YAML
python -m m1_demanda.run --mode search --modelo mlp           # búsqueda aleatoria (--presupuesto N)
python -m m1_demanda.run --mode final                         # 30 semillas × 2 mejores configuraciones × 2 familias
python -m m1_demanda.run --mode demo                          # < 2 min, solo carga artefactos
python -m m1_demanda.valor                                    # valor del pronóstico, temporada alta, sensibilidad
```
Todo se parametriza en `configs/m1_demanda.yaml` (ventana, arquitectura, optimizador, búsqueda), `configs/m1_operacion.yaml` y `configs/global.yaml` (semilla base y 30 semillas), o por la línea de comandos (`--config`, `--device`, `--seed`, `--semillas`, `--presupuesto`).

**Comparación justa:** presupuesto de búsqueda explícito (`reports/tables/m1_busqueda_*.json`), 30 semillas por configuración final (`reports/tables/m1_semillas_*.csv`), número de parámetros entrenables y tiempo (`reports/tables/m1_comparacion.csv`). La red que usa la app se elige con el MASE de **validación** (`models/m1_demanda/modelo_final.json`), nunca con la prueba.

| Archivo | Función |
|---|---|
| `ventanas.py` | Ventanas sin fuga: historial, calendario futuro, nivel local, máscaras y columnas prohibidas |
| `rolling_origin.py` | Orígenes de entrenamiento, validación y prueba (regla de la Persona 1) |
| `evaluate.py` | Métricas por ruta, horizonte y temporada alta/base |
| `baselines.py` | Ingenuo estacional y GLM Poisson |
| `losses.py` | Poisson, pinball, combinada y binomial negativa (ablación) |
| `sobredispersion.py` | Diagnóstico de varianza/media |
| `nn_models.py`, `train.py` | MLP, GRU/LSTM; entrenamiento con parada temprana, guardado y carga |
| `search.py`, `final.py` | Búsqueda con presupuesto; 30 semillas y tabla de comparación |
| `figuras.py`, `valor.py` | Figuras; valor del pronóstico y sensibilidad |
| `predict.py`, `run.py` | Contrato con la app; línea de comandos |

## Operación (Persona 2)
A partir del pronóstico (media o P90) se dimensionan, por corredor, los despachos, los vehículos, la flota del horizonte y la plantilla de conductores. Fórmulas y justificación en `docs/math_foundations.md` (secciones 3 y 4). Decisiones clave:

* **Ida y regreso:** los ciclos del corredor son el **máximo** de los despachos de los dos sentidos, no la suma (un vehículo que va debe volver). Los despachos del sentido liviano que exceden lo requerido se reportan como subutilizados y se cobran.
* **Parámetros** en `configs/m1_operacion.yaml`, cada uno marcado como **[F]** fuente (con cita y fecha), **[S]** supuesto o **[D]** derivado de los datos. Un valor [F] vacío se rechaza al cargar. **Casi todo es [S]** (capacidades, tiempos de viaje, ocupación, costos): son supuestos a reemplazar con datos de la empresa; la sensibilidad de ±10 % (`reports/tables/m1_sensibilidad.csv`, `reports/figures/m1/sensibilidad_tornado.png`) muestra cuáles importan.
* **Oferta limitada:** los pasajeros de origen no miden la ocupación real ni la demanda no atendida por falta de sillas (`reports/tables/m1_censura_indicios.csv`); se planea con un cuantil alto en lugar de modelar la censura.
* **Cómo leer la flota del horizonte de producción:** la flota `F_c` es el **máximo diario** del horizonte, así que un solo día de pico la fija para todo el mes. El origen de producción (2026-03-31) cae en la **Semana Santa de 2026** (Jueves Santo = 2 de abril): para Bogotá → Girardot el P90 del 1 de abril es ~5.600 pasajeros frente a una mediana de ~900, y exige 134 vehículos ese día frente a una mediana de 30. Por eso el corredor Girardot ↔ Bogotá aparece con una flota mucho mayor que Pereira ↔ Armenia con demanda mensual parecida. No es un error: es el efecto de temporada alta (`reports/tables/m1_flota_conductores_dia.csv`), y es la base de la pregunta de flota propia, arrendada o convenios.
* **Valor del pronóstico y temporada alta:** `reports/tables/m1_valor_pronostico.csv`, `m1_temporada_alta.csv` y `reports/figures/m1/m1_valor_pronostico.png` comparan planes con ingenuo, GLM y redes, puntual y P90, frente al oráculo (demanda real). Con los costos supuestos, el cuantil de decisión óptimo es ≈ 0,72, no 0,90 (`docs/math_foundations.md`, sección 4).
