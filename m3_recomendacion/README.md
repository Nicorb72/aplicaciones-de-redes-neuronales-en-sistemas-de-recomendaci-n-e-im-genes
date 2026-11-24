# Módulo 3 — Sistema de recomendación de destinos turísticos

**Proyecto:** Redes neuronales para series de tiempo, imágenes y recomendación   
**Destino del proyecto:** recomendaciones turísticas en India  
**Métrica principal:** NDCG@3  
**Estado:** entrenamiento, evaluación y servicio de inferencia implementados; la interfaz web integrada y la publicación corresponden a la aplicación general del equipo.

## 1. Objetivo y caso de uso

Desarrollar y comparar un sistema que recomiende destinos turísticos según el historial de interacciones y las preferencias del usuario. Se implementan tres familias de modelos para usuarios conocidos (popularidad, factorización colaborativa y red neuronal) y dos alternativas para usuarios nuevos (popularidad global y popularidad por preferencias).

**Escenario:** en el canal web de turismo de la empresa, el cliente elige su perfil o se identifica como usuario existente y recibe hasta **K = 3** sugerencias, con nombre del destino, ubicación, tipo de turismo y época recomendada. El valor de K es configurable de **1 a 5**, sujeto al número de destinos aún disponibles para cada usuario. En los usuarios existentes se muestra el historial registrado y se excluyen del ranking los destinos con interacciones observadas en entrenamiento.

**Alcance y límites:** los destinos son un catálogo académico de cinco opciones consolidadas; los resultados sirven para comparar métodos bajo este conjunto de datos y **no** equivalen a tasas de compra ni a una validación comercial real.

## 2. Datos y fuente

**Fuente:** Aman Mehra, *Travel Recommendation Dataset (India Travel Recommender System Dataset)*, disponible en [Kaggle](https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset).

Los datos originales están en `data/m3_recomendacion/raw/`:

| Archivo | Contenido |
|---|---|
| `Expanded_Destinations.csv` | Catálogo inicial de destinos y atributos turísticos |
| `Final_Updated_Expanded_Users.csv` | Identificadores y preferencias de usuarios |
| `Final_Updated_Expanded_UserHistory.csv` | Historial de visitas y calificaciones |
| `Final_Updated_Expanded_Reviews.csv` | Reseñas y calificaciones |

La preparación consolida registros del catálogo con igual nombre, estado, tipo y temporada; normaliza el historial y las reseñas; y agrega las calificaciones de cada par usuario–destino antes de generar las particiones. La escala de calificación es **1 a 5**. Se define como interacción positiva una calificación consolidada **mayor o igual a 4**. Los negativos utilizados para la pérdida son interacciones **observadas** con calificación inferior a 4: no se consideran negativas automáticamente las visitas ausentes.

**Tamaño de trabajo:** 999 usuarios registrados, **5 destinos únicos** después de consolidar el catálogo y 1.644 pares usuario–destino consolidados. Esto equivale a una densidad de interacciones observadas aproximada de `1.644 / (999 × 5) = 32,91 %` y una dispersión de **67,09 %**, usando todos los usuarios registrados. La definición y las hipótesis de estas cifras deben explicarse en el informe técnico. Las tablas originales de destinos tienen múltiples identificadores correspondientes a las mismas cinco opciones, por lo que no se interpretan esos identificadores como destinos turísticos distintos.

**Perfiles de preferencias disponibles:**

- `Beaches, Historical`
- `City, Historical`
- `Nature, Adventure`

**Precaución de interpretación:** la naturaleza real o sintética de los registros debe verificarse y citarse con la documentación original de Kaggle. El repositorio por sí solo no acredita observaciones de clientes reales; no presentar estas métricas como resultados de producción.

### Particiones y prevención de fugas de información

| Partición | Pares observados | Usuarios | Uso |
|---|---:|---:|---|
| `train.csv` | 1.440 | 813 | Ajustar los tres modelos y las frecuencias de popularidad |
| `val.csv` | 51 | 51 | Selección de hiperparámetros y parada temprana |
| `test.csv` | 52 | 52 | Evaluación final de usuarios existentes |
| `cold_val.csv` | 17 | 9 | Comparar estrategias para nuevos usuarios durante validación |
| `cold_test.csv` | 84 | 36 | Evaluación final para usuarios nuevos |

Los 51 y 52 usuarios de validación/test **son usuarios presentes también en entrenamiento**: se retiene una interacción positiva por usuario elegible. Para `cold_val.csv` y `cold_test.csv`, se reservan los usuarios completos, sin historial en `train.csv`. Por tanto, los conjuntos `warm` y `cold` se interpretan de manera diferente. La semilla para la partición es **42**.

**Regla de evaluación:** `test.csv` y `cold_test.csv` no intervienen en entrenamiento, búsqueda ni selección de configuraciones. Los resultados de test se informan una vez, sin reajustar el modelo en función de ellos.

> Las particiones y los archivos procesados están incluidos en el repositorio; para reproducir las evaluaciones no es necesario volver a descargarlos ni regenerarlos. `data.prepare` y `data.split` vuelven a escribir esos archivos y solo deben ejecutarse cuando se requiera regenerar el preprocesamiento.

## 3. Estructura del módulo

Todas las rutas siguientes son relativas a **la raíz del repositorio**:

```text
m3_recomendacion/
├── __init__.py
├── README.md
├── COMANDOS.md
├── predict.py                 # API de inferencia para la Persona 5
├── data/
│   ├── explore.py
│   ├── prepare.py
│   └── split.py
├── models/
│   ├── popularity.py
│   ├── collaborative.py
│   ├── neural_recommender.py
│   ├── random_baseline.py
│   ├── cold_start.py
│   └── model_io.py           # Exportación/carga del modelo neuronal
├── experiments/
│   ├── train.py
│   ├── search.py
│   ├── search_neural.py
│   ├── stability.py
│   ├── stability_neural.py
│   ├── ablation_neural.py
│   ├── ablation_stability.py
│   └── negative_sampling.py
├── evaluation/
│   ├── evaluate.py
│   ├── metrics_by_k.py
│   ├── evaluate_user_groups.py
│   ├── compare_models.py
│   ├── cold_diagnostics.py
│   ├── test_final.py
│   └── cold_test_final.py
└── analysis/
    ├── eda.py
    ├── embedding_visualization.py
    ├── recommendation_examples.py
    ├── recommendation_concentration.py
    ├── first_position_analysis.py
    └── campaign_simulation.py

data/m3_recomendacion/
├── raw/                        # Cuatro CSV originales
└── processed/                  # Particiones, métricas y tablas derivadas

artifacts/m3/
├── neural_c5.pt                # Pesos, mapeos e historial del modelo C5
└── cold_start.json            # Frecuencias para usuarios nuevos

reports/figures/m3/           # Gráficas del módulo
tests/test_m3_predict.py       # Pruebas unitarias del recomendador
```

Los directorios `__pycache__/`, `.pytest_cache/` y los entornos virtuales no forman parte de la entrega. Conservar los archivos `__init__.py`.

## 4. Entorno y dependencias

Ejecutar **siempre desde la raíz del repositorio**. Se verificó localmente la ejecución con **Python 3.14.7**, **PyTorch 2.15.0+cpu** y **pytest 9.2.0**. El archivo `requirements.txt` de la raíz es compartido por los tres módulos y debe coordinarse con el equipo: **no sustituirlo por el README del M3**.

Dependencias usadas en el M3: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` y `pytest` (para pruebas). La reproducibilidad de la instalación requiere versiones fijadas en el archivo general `requirements.txt`; la compatibilidad entre plataformas debe comprobarse en el equipo de destino.

```powershell
# Instalación con el archivo compartido, cuando esté consolidado:
python -m pip install -r requirements.txt

# Verificación de dependencias:
python -c "import torch, pandas, numpy, matplotlib, sklearn, pytest; print('Dependencias M3 correctas')"
```

No es obligatorio utilizar un entorno virtual, aunque se recomienda para separar dependencias. Un entorno local no debe versionarse en Git.

## 5. Modelos y metodología

### 5.1 Popularidad global

Ordena los destinos por el número de interacciones **positivas observadas en `train.csv`**. Sirve como línea base que no necesita identificadores de usuario.

### 5.2 Filtrado colaborativo clásico

Modelo de factorización con **8 factores latentes**, `learning_rate = 0.01`, `weight_decay = 0.001`, **100 épocas** y semilla 42 en la comparación de referencia. Sus representaciones capturan regularidades de interacción usuario–destino.

### 5.3 Recomendador neuronal C5

El modelo combina embeddings de **usuarios** y **destinos** con variables de contenido codificadas: preferencia turística del usuario y tipo del destino. La red produce un *logit* para ordenar candidatos; se transforma con una función sigmoide cuando se consulta el servicio. **Ese puntaje no es una probabilidad comercial calibrada**.

| Parámetro | Valor |
|---|---:|
| Dimensión de embeddings | 8 |
| Dimensión de capa oculta | 16 |
| Dropout | 0,1 |
| Optimizador | Adam |
| Tasa de aprendizaje | 0,001 |
| Función de pérdida | `BCEWithLogitsLoss` |
| Épocas máximas | 100 |
| Paciencia de parada temprana | 15 |
| Semilla de referencia | 42 |
| K principal | 3 |
| Parámetros entrenables | **7.089** |

Para `train.csv` se registraron **426 positivos** y **1.014 negativos observados**. La configuración final utiliza **todos** esos registros (`negative_ratio=None`); el experimento `negative_sampling.py` estudia una proporción reducida de negativos. El entrenamiento selecciona los mejores pesos mediante **NDCG@3 de validación** (mejor época 5, parada en época 20 en la corrida con semilla 42).

**Justificación de complejidad:** se exploraron dimensiones y probabilidades de dropout; el tamaño final es moderado frente a la cantidad de usuarios y a los cinco destinos disponibles. La elección final se obtuvo en validación, no en prueba.

### 5.4 Usuarios nuevos (*cold-start*)

- **Global:** ranking por frecuencia total de positivos observados en entrenamiento.
- **Por preferencias:** ranking por frecuencia de positivos en usuarios del perfil seleccionado, con popularidad global como desempate.

El servicio de inferencia usa por defecto `preferences` para mostrar una recomendación personalizada cuando el perfil está disponible; el método `global` sigue disponible como alternativa. La validación con solo nueve usuarios muestra diferencias pequeñas y métricas mixtas: **la selección predeterminada no debe justificarse exclusivamente por el resultado de `cold_test.csv`**.

## 6. Métricas

Para un usuario con conjunto de destinos relevantes `R_u` y lista ordenada de recomendaciones `L_u(K)`:

- **Precision@K:** `|L_u(K) ∩ R_u| / K`.
- **Recall@K:** `|L_u(K) ∩ R_u| / |R_u|`.
- **DCG@K:** suma de `1 / log2(posición + 1)` para las posiciones relevantes del ranking; **NDCG@K** divide esta suma entre el DCG ideal.
- **Cobertura del catálogo:** proporción de destinos diferentes que aparecen en las listas, respecto de todos los destinos del catálogo.

Se reportan los promedios sobre los usuarios evaluables. En el conjunto con cinco destinos, Recall@3 puede alcanzar cifras elevadas simplemente porque K cubre una fracción grande del catálogo; interpretar conjuntamente NDCG, Precision y cobertura.

## 7. Resultados reproducidos

### 7.1 Validación de usuarios existentes (51 usuarios, K = 3)

| Modelo | Precision@3 | Recall@3 | NDCG@3 | Cobertura |
|---|---:|---:|---:|---:|
| Popularidad | 0,2810 | 0,8431 | 0,6438 | 1,0000 |
| Colaborativo | **0,3203** | **0,9608** | **0,7474** | 1,0000 |
| Neuronal C5 | 0,3137 | 0,9412 | 0,7423 | 1,0000 |

Resultados de la comparación con semilla de referencia 42; `data/m3_recomendacion/processed/metrics_by_k.csv` conserva los cinco valores de K (1 a 5).

### 7.2 Estabilidad de los modelos entrenables: 30 semillas en validación

| Modelo | NDCG@3 media | Desv. estándar | Peor valor | Mejor valor |
|---|---:|---:|---:|---:|
| Colaborativo | 0,6800 | 0,0377 | 0,5906 | 0,7836 |
| Neuronal C5 | **0,7059** | 0,0289 | 0,6293 | 0,7631 |

Los resultados completos por semilla, con Precision, Recall, cobertura y tiempos, están en `stability_collaborative.csv` y `stability_neural.csv`. Los tiempos acumulados registrados en esos experimentos fueron ~20,65 s y ~58,81 s respectivamente, **en el entorno donde se generaron los archivos**, por lo que no representan garantías de tiempo en cualquier computador.

### 7.3 Evaluación FINAL de usuarios existentes (52 usuarios, `test.csv`, K = 3)

| Modelo | Precision@3 | Recall@3 | NDCG@3 | Cobertura |
|---|---:|---:|---:|---:|
| **Popularidad** | **0,3205** | **0,9615** | **0,7640** | 1,0000 |
| Neuronal C5 | 0,3077 | 0,9231 | 0,6774 | 1,0000 |
| Colaborativo | 0,2949 | 0,8846 | 0,6440 | 1,0000 |

**Interpretación:** la popularidad obtuvo el mayor NDCG@3 en prueba, a pesar de que el colaborativo lideró en la corrida de validación con semilla 42. El modelo neuronal se mantiene como modelo de la demostración de usuarios existentes por ser la arquitectura entrenada que integra contenido; **no afirmar que superó a las líneas base en test**. Los resultados de test no se utilizaron para reajustar configuraciones.

### 7.4 Usuarios nuevos — validación (9 usuarios)

| Estrategia | Precision@3 | Recall@3 | NDCG@3 |
|---|---:|---:|---:|
| Global | 0,2593 | 0,6667 | 0,4091 |
| Por preferencias | 0,2222 | 0,5000 | 0,4216 |

### 7.5 Usuarios nuevos — prueba FINAL (36 usuarios, `cold_test.csv`)

| Estrategia | Precision@3 | Recall@3 | NDCG@3 | Cobertura |
|---|---:|---:|---:|---:|
| Global | 0,2315 | 0,5880 | 0,4499 | 0,6000 |
| **Por preferencias** | **0,2685** | **0,6713** | **0,5257** | **1,0000** |

Las mejoras en *cold test* se documentan como resultados de evaluación, no como motivo para nuevos ajustes posteriores al test.

## 8. Ejecución: demostración rápida sin reentrenar

**Requisito previo:** existen `artifacts/m3/neural_c5.pt`, `artifacts/m3/cold_start.json` y `data/m3_recomendacion/processed/catalogo_destinos.csv`.

```powershell
# Usuario registrado en el entrenamiento
python -m m3_recomendacion.predict --usuario 1 --k 3

# Usuario nuevo: preferencias (opción predeterminada)
python -m m3_recomendacion.predict --preferencias "Nature, Adventure" --k 3

# Usuario nuevo: popularidad global
python -m m3_recomendacion.predict --preferencias "Nature, Adventure" --estrategia global --k 3
```

Cada comando imprime un JSON con `tipo_usuario`, `user_id`, `preferencias`, `estrategia`, `k_solicitado`, `historial`, `recomendaciones` y `nota_puntaje`. La inferencia carga los pesos guardados y **no reentrena**. Este es el flujo corto propuesto para la demostración del M3; la duración menor a dos minutos debe comprobarse en el equipo de entrega.

**Ejemplo comprobado, usuario 1:** historial Goa Beaches y Taj Mahal; recomendaciones Jaipur City, Leh Ladakh y Kerala Backwaters, en ese orden. El historial utilizado para exclusión incluye todas las interacciones observadas de entrenamiento, tanto positivas como no positivas.

### 8.1 Interfaz para la Persona 5

```python
from m3_recomendacion.predict import recomendar, obtener_preferencias

opciones = obtener_preferencias()  # Preferencias válidas para un selector

existente = recomendar(user_id=1, k=3)
nuevo = recomendar(preferencias="Nature, Adventure", k=3)
alternativo = recomendar(
    preferencias="Nature, Adventure", estrategia="global", k=3
)

print(existente["historial"])
print(existente["recomendaciones"])
print(nuevo["recomendaciones"])
```

La función `recomendar()` **solo genera datos de predicción**. La construcción, publicación e integración visual de la aplicación web general corresponden a la Persona 5. El módulo M3 entrega el contrato de integración; no publica la aplicación por sí solo.

**Nota funcional:** un usuario cuyo `UserID` no se encuentre en los embeddings se trata como nuevo y debe proporcionar `preferencias`. Si un usuario existente tiene menos de K destinos no vistos, la lista devuelta puede contener menos de K elementos.

## 9. Exportación de artefactos (solo si faltan)

La exportación se realiza antes del despliegue. No es necesaria para cada consulta:

```powershell
# Entrena C5 con train, selecciona época con val, guarda y prueba la carga
python -m m3_recomendacion.models.model_io

# Genera frecuencias cold-start usando solo train
python -m m3_recomendacion.predict --preparar-cold-start
```

| Archivo | Uso |
|---|---|
| `artifacts/m3/neural_c5.pt` | Pesos del modelo, índices de usuarios/destinos, variables y registros observados |
| `artifacts/m3/cold_start.json` | Conteos de popularidad global y por perfil |

Ambos archivos, más el catálogo procesado, deben quedar disponibles al ejecutar la aplicación. **No incluirlos dentro de `.gitignore` si se espera que el repositorio funcione sin entrenamiento**. Evitar cargar checkpoints provenientes de fuentes no confiables.

## 10. Entrenamiento y evaluación experimental

Estos comandos **sí** pueden reentrenar modelos y regenerar resultados. Ejecutar desde la raíz.

```powershell
# Diagnóstico, preparación y particiones (opcionales si ya hay CSV procesados)
python -m m3_recomendacion.data.explore
python -m m3_recomendacion.data.prepare
python -m m3_recomendacion.data.split

# Entrenamiento de referencia C5
python -m m3_recomendacion.experiments.train --embedding-dim 8 --hidden-dim 16 --dropout 0.1 --lr 0.001 --epochs 100 --seed 42 --patience 15 --k 3

# Comparación de modelos sobre validación, K=1..5
python -m m3_recomendacion.evaluation.metrics_by_k

# Comparación por nivel de historial
python -m m3_recomendacion.evaluation.evaluate_user_groups

# Cold-start en validación
python -m m3_recomendacion.models.cold_start

# Pruebas finales, conservando las configuraciones ya seleccionadas
python -m m3_recomendacion.evaluation.test_final
python -m m3_recomendacion.evaluation.cold_test_final
```

**Resultados generados:**

- `processed/metrics_by_k.csv`: comparación por K.
- `processed/test_final_results.csv` y `test_final_users.csv`: test de usuarios existentes.
- `processed/cold_test_final_results.csv` y `cold_test_final_users.csv`: test de usuarios nuevos.
- `processed/neural_training_history_original.csv`: evolución de la pérdida de entrenamiento y NDCG en validación.
- `reports/figures/m3/`: figuras del módulo.

Los archivos de prueba se utilizan solo para reportar las métricas finales, no para entrenar ni elegir nuevos hiperparámetros.

### Experimentos complementarios

```powershell
# Búsqueda de hiperparámetros
python -m m3_recomendacion.experiments.search
python -m m3_recomendacion.experiments.search_neural

# Estabilidad en validación
python -m m3_recomendacion.experiments.stability
python -m m3_recomendacion.experiments.stability_neural

# Ablación y muestreo de negativos
python -m m3_recomendacion.experiments.ablation_neural
python -m m3_recomendacion.experiments.ablation_stability
python -m m3_recomendacion.experiments.negative_sampling

# Análisis y gráficas
python -m m3_recomendacion.analysis.eda
python -m m3_recomendacion.analysis.embedding_visualization
python -m m3_recomendacion.analysis.recommendation_examples
python -m m3_recomendacion.analysis.recommendation_concentration
python -m m3_recomendacion.analysis.first_position_analysis
python -m m3_recomendacion.analysis.campaign_simulation
```

La búsqueda registrada incluye **8 configuraciones neuronales** y **8 colaborativas** en los CSV `search_neural.csv` y `search_collaborative.csv`. Es importante declarar ese presupuesto y sus tiempos en el informe, además de explicar las condiciones no idénticas de optimización entre familias. La variabilidad neuronal de 30 semillas sí se ha documentado.

## 11. Pruebas automatizadas

```powershell
# Las nueve pruebas actuales del recomendador
python -m pytest tests/test_m3_predict.py -v
```

**Resultado comprobado en Windows con Python 3.14.7:** `9 passed`. Estas pruebas verifican la salida y las reglas principales de los modelos; **no sustituyen** las pruebas de la interfaz web, la carga en otro entorno ni las evaluaciones de test.

## 12. Figuras, análisis de negocio y discusión

Las figuras se generan desde scripts en `analysis/`, `experiments/` y `evaluation/`. Los principales productos son:

| Tema | Archivos / ejecución |
|---|---|
| Distribución de interacciones | `fig_m3_01`, `fig_m3_02` y análisis descriptivo |
| Preferencias y calificaciones | `fig_m3_03`, `fig_m3_04` |
| Curvas del neuronal | `fig_m3_05_*`, `fig_m3_06_*` |
| Estabilidad y sensibilidad | `fig_m3_07_*` y CSV de 30 semillas |
| Concentración de recomendaciones | `fig_m3_08_*` y `recommendation_concentration.csv` |
| Precision/Recall/NDCG por K | `fig_m3_09`, `fig_m3_10`, `fig_m3_11` |
| Embeddings de usuarios y destinos | `fig_m3_12`, `fig_m3_13` |
| Simulación de campaña | `campaign_simulation.csv` |

**Concentración:** la cobertura puede ser 100 % aunque el primer destino recomendado se concentre en unas pocas alternativas. Por ello también se estudia la frecuencia de exposición en la primera posición y un índice de concentración. Dado el catálogo de cinco destinos, las conclusiones sobre congestión o capacidad operativa son **escenarios hipotéticos**, no pronósticos de ocupación real.

**Campaña:** `campaign_simulation.py` simula el efecto de promocionar Kerala Backwaters a una fracción de usuarios del perfil `Nature, Adventure` bajo tasas de conversión explícitamente supuestas. Las conversiones calculadas **no son conversiones medidas**, y la red neuronal no está calibrada para estimarlas directamente.

**Lectura crítica:** la red neuronal obtiene una media de NDCG@3 mayor que el colaborativo en las 30 semillas de validación, pero el modelo de popularidad gana en el test final. El tamaño del catálogo, la información limitada por usuario, la búsqueda de configuraciones, el posible carácter sintético de los datos y la ausencia de validación en producción impiden generalizar los resultados a un negocio turístico real.

## 13. Fundamentación matemática y reproducibilidad

Para el informe técnico deben desarrollarse explícitamente, con ecuaciones y citas apropiadas:

1. Factorización de matrices, embeddings, variables de contenido y dimensiones de la red C5.
2. Función sigmoide y deducción de `BCEWithLogitsLoss` a partir de un modelo Bernoulli de interacción positiva.
3. Retropropagación, actualizaciones de Adam y regularización (dropout, penalización del modelo colaborativo).
4. Precision@K, Recall@K, NDCG@K, cobertura y el criterio de selección por validación.
5. Justificación de las semillas (partición 42, referencia 42 y 30 corridas de estabilidad), rangos de hiperparámetros y presupuesto computacional.

**Estado de figuras solicitadas por el profesor:** hay curvas de **pérdida de entrenamiento** y **NDCG de validación**; esa segunda curva no equivale a una curva de **pérdida de validación**. El diagrama formal de arquitectura con capas y dimensiones y la curva de pérdida de validación se deben completar o justificar como pendientes en el informe final si todavía no existen. Igualmente, el informe debe incluir las ecuaciones, tablas con fuente, conclusiones de negocio, bibliografía APA 7, discusión ética, registro del uso de IA y “Cacería de la alucinación”. El README documenta el código, pero **no sustituye el reporte técnico**.

## 14. Ética y límites de uso

- **Privacidad:** utilizar identificadores y perfiles solo para la finalidad prevista; no mostrar historiales de terceros sin autorización en una aplicación real.
- **Sesgo y representatividad:** con pocos destinos y perfiles discretos, pueden reforzarse preferencias dominantes y excluir alternativas menos frecuentes.
- **Exploración:** usuarios nuevos y perfiles no observados requieren controles explícitos; un usuario con perfil desconocido recibe un error claro y no una recomendación fabricada.
- **Interpretación:** NDCG y popularidad no miden satisfacción, ventas o conversiones reales.
- **Trazabilidad:** conservar versiones de datos, semillas, métricas de validación y test, y las hipótesis de negocio.


## 16. Referencia del conjunto de datos

Mehra, A. (s. f.). *Travel Recommendation Dataset (India Travel Recommender System Dataset)* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset

> La fecha de publicación y la descripción de generación de datos deben confirmarse en la ficha original antes de fijar esta referencia en el reporte APA 7.
