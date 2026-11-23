# Plan de implementación — Persona 2, Módulo 1 (modelos y operación)

**Responsable:** Persona 2. **Alcance:** solo Módulo 1. Los baselines de M2 y M3 quedan fuera hasta que la Persona 1 entregue esos datos (el de M2 deberá usar el mismo backbone y las mismas transformaciones que la Persona 3).

## 1. Problema

Pronosticar los pasajeros diarios de 10 rutas (5 corredores × 2 sentidos) a 30 días, con pronóstico puntual y cuantil P90. Con ese pronóstico se dimensionan vehículos y conductores por corredor. Se compara de forma justa contra baselines y se expone a la app mediante `m1_demanda/predict.py`, sin reentrenar y sin internet.

## 2. Requisitos acordados

1. Baselines M1: ingenuo estacional (mismo día de la semana anterior) y regresión lineal (GLM Poisson) con rezagos y calendario.
2. Redes: MLP con ventanas de rezagos y un codificador recurrente (LSTM o GRU).
3. Pérdida para conteos justificada, cuantil P90, búsqueda de hiperparámetros con presupuesto explícito, 30 semillas por configuración final y número de parámetros entrenables.
4. Evaluación con origen rodante a 30 días: MAE, RMSE, MASE, error por horizonte, cobertura del P90 y pérdida cuantílica.
5. Operación: flota y conductores, sensibilidad ±10 %, valor del pronóstico (puntual vs P90), temporada alta, `configs/m1_operacion.yaml` con fuentes citadas.
6. Restricciones: no modificar particiones ni preprocesamiento de la Persona 1; ninguna variable puede depender de información desconocida 30 días antes; todo parametrizable por YAML o CLI; la demo corre en menos de 2 minutos; todo debe poder explicarse en vivo.
7. Stack: PyTorch, estilo del profesor (clases `nn.Module` con `nn.Sequential`, `Dataset`/`DataLoader`, funciones de entrenamiento y validación explícitas, semilla con `torch.random.manual_seed`, conteo de parámetros con `sum(p.numel() ...)`). CPU por defecto y `--device {cpu,mps,auto}`; la decisión final se toma con una medición (Tarea 9).
8. `common/` (semillas, multisemilla, presupuesto, carga de configuración): versiones mínimas que la Persona 5 revisa.
9. Documentación: aportes a `docs/math_foundations.md`, `docs/hallucination_hunt.md` (entradas H-M1-5 en adelante) y `docs/ai_usage.md`.

## 3. Contexto del repo

- Datos de la Persona 1 en `data/m1_demanda/processed/`: `series_diarias.csv.gz`, `pasajeros_ancho.csv`, `particiones.json` (train 2021-01-01 a 2023-06-30, val 2023-07-01 a 2023-09-30, test 2023-10-01 a 2023-12-31, post_hueco 2025-01-01 a 2026-03-30, horizonte 30, 10 orígenes rodantes en prueba), `rutas.csv`, `mezcla_clase_vehiculo.csv` (solo entrenamiento) y `validez_sistema.csv`. Calendario en `data/m1_demanda/external/calendario_colombia.csv`.
- Notas de la sección 6 de `docs/m1_datos_y_eda.md`: objetivo `pasajeros_imp`; no usar `segmento == 0`; descartar ventanas que crucen los huecos largos (Villavicencio→Bogotá, marzo 2022; Duitama→Sogamoso, enero 2026); `evento_no_previsible` no es predictor; el patrón semanal y el efecto de los puentes dependen del sentido; ventanas de al menos 28 días; flota con el sentido más cargado del corredor.
- Entorno: `.venv` con Python 3.12 (`uv venv --python 3.12 .venv`), dependencias de `requirements.txt` más torch.

## 4. Fases y dependencias

| Fase | Tareas | Necesito | Entrego |
|---|---|---|---|
| 0. Preparación | 1–3 | Enunciado en texto; que la Persona 1 confirme que las particiones no cambian | `common/` a la Persona 5 (revisión) y a las Personas 3 y 4 (uso) |
| 1. Baselines y contrato | 4–7 | — | Contrato y `predict.py` v0 a la Persona 5, con números reales del ingenuo |
| 2. Redes | 8–12 | — | Tabla de comparación justa y modelo en `models/m1_demanda/` |
| 3. Operación | 13–14 | — | Valor del pronóstico, temporada alta, sensibilidad |
| 4. Integración | 15 | Que la Persona 5 valide `predict.py` v1 | Demo, README y documentación |

## 5. Decisiones de diseño (D1–D10, opciones recomendadas aprobadas)

- **D1. Pérdida de conteo: Poisson** (`nn.PoissonNLLLoss(log_input=True)`) para la media, con una cabeza separada de pérdida cuantílica para el P90. La sobredispersión se documenta con la razón varianza/media por ruta × día de semana × mes en entrenamiento. La binomial negativa queda solo como ablación opcional. Alternativas descartadas: MSE sobre log1p (sesgo al volver a la escala original) y binomial negativa como pérdida principal (cabeza extra para α, parametrización de `torch.distributions` confusa).
- **D2. P90:** tronco compartido con dos cabezas. μ_h = ℓ·exp(f_h); q_h = μ_h·exp(softplus(g_h)), lo que garantiza q ≥ μ. Pérdida L = L_Poisson(μ) + λ·L_pinball,0.9(q), con L_τ(y,q) = max(τ(y−q), (τ−1)(y−q)). En los baselines el P90 es el cuantil empírico 0,9 del cociente y/ŷ por horizonte, estimado en orígenes de validación.
- **D3. Modelo global con embedding de ruta** (dimensión 2 a 4) para las redes; baselines por ruta.
- **D4. Estrategia directa multisalida (MIMO, 30 salidas).** Entrada: log(y/ℓ) de los últimos L días antes del origen, calendario de esos días, calendario de los 30 días objetivo (conocido por ley) y embedding de ruta. Ningún dato posterior a o−1 entra al modelo.
- **D5. Escala:** nivel local ℓ = media de y en los 28 días previos al origen, como offset: log μ = log ℓ + f(x). No se usa el rezago 364.
- **D6. Máscaras:** entradas con `pasajeros_imp`; pérdida y métricas solo sobre días objetivo no imputados con `pasajeros` no nulo; se descartan ventanas que crucen segmentos o NaN largos; peso 0 a objetivos dentro del Paro Nacional (configurable excluir también ventanas cuya entrada lo toque). Columnas prohibidas como entrada, validadas por código: `evento_no_previsible`, `particion`, `imputado`, `segmento`, `pasajeros_crudo`, `despachos*`, `pasajeros_transito`, `nombre_festivo`.
- **D7. Evaluación:** pesos entrenados una vez (train, parada temprana en val). En cada origen solo se actualiza la ventana de entrada con datos revelados. Selección de hiperparámetros con orígenes semanales dentro de validación; nunca con prueba.
- **D8. Búsqueda aleatoria** (Bergstra y Bengio, 2012): 30 configuraciones por familia (MLP y RNN), 1 semilla cada una, parada temprana, selección por MASE medio en validación con desempate por pérdida cuantílica. Registro con `common/search_budget`. Las 2 mejores configuraciones se corren con 30 semillas. Los baselines declaran su presupuesto (ingenuo: 0 hiperparámetros; GLM: 1 configuración).
- **D9. Oferta limitada:** se documenta como limitación y se cuantifican indicios (días en que pasajeros/despacho se acercan a `max_pax_por_despacho`). Se planea con P90 y sensibilidad de demanda +10 %. No se modela la censura.
- **D10. Sentido de regreso:** viaje redondo; ciclos requeridos = max(N_ida, N_regreso) por corredor, no la suma. El modelo de flujo en red queda como extensión documentada. Los despachos subutilizados del sentido liviano se reportan como costo.

## 6. Fórmulas de operación

Notación: r ruta, r′ su regreso, c corredor, t día; D_{r,t} = μ (puntual) o q (P90).

```
Capacidad media            C_r     = Σ_k w_{r,k}·cap_k            (w = participacion_despachos, solo entrenamiento)
Despachos                  N_{r,t} = ⌈ D_{r,t} / (φ·C_r) ⌉         (φ = ocupación objetivo)
Ciclos del corredor        R_{c,t} = max(N_{r,t}, N_{r′,t})
Ciclos por vehículo y día  v_c     = ⌊ H_op / T_c ⌋,  T_c = t_ida + t_regreso + 2·t_alistamiento
Vehículos por día          V_{c,t} = ⌈ R_{c,t} / v_c ⌉
Flota del horizonte        F_c     = ⌈ max_t V_{c,t}·(1 + ρ_m) ⌉
Horas de conducción        H_{c,t} = R_{c,t}·(t_ida + t_regreso)
Conductores-día            CD_{c,t}= ⌈ H_{c,t} / h_dia_max ⌉
Plantilla del mes          K_c     = ⌈ max( max_t CD_{c,t}, Σ_t H_{c,t} / (h_semana·30/7) ) / (1 − a) ⌉
No atendidos               U_{r,t} = max(0, y_{r,t} − N_{r,t}·C_r)
Costo                      Costo   = c_veh·ΣV + c_desp·ΣN + c_cond·K + c_u·ΣU
```

Justificación del P90 (vendedor de periódicos): el cuantil óptimo es τ* = c_u/(c_u + c_o); va en `docs/math_foundations.md`.

### Parámetros de `configs/m1_operacion.yaml`

F = fuente primaria obligatoria, S = supuesto declarado cubierto por la sensibilidad, D = derivado de los datos.

| Clave | Contenido | Tipo |
|---|---|---|
| `capacity_by_class` | Sillas por clase (BUS, BUSETA, MICROBUS, CAMIONETA, CAMPERO, AUTOMOVIL) | F (normativa del Ministerio de Transporte o fichas técnicas); si no se encuentra pasa a S y se avisa. Los `max_pax_por_despacho` de `m1_datos.yaml` no son normativos. |
| `class_mix` | Ruta a `mezcla_clase_vehiculo.csv` | D |
| `target_occupancy` (φ) | Ocupación objetivo | S |
| `cycle_time` | Por corredor: `viaje_h` (F, fuente pública citada) y `alistamiento_h` (S) | F / S |
| `operating_hours` (H_op), `technical_reserve` (ρ_m), `absenteeism` (a) | — | S |
| `workday.jornada_semanal_h` | Por vigencia, con fecha de inicio. Ley 2101 de 2021 (reducción gradual de 48 a 42 h); escalones y fechas verificados en el texto oficial | F |
| `workday.max_conduccion_dia_h` | Norma de jornada para conductores de transporte público, por identificar y verificar | F |
| `costs` | c_veh, c_desp, c_u, c_cond. Si c_cond se ancla al SMMLV, ese valor es F (decreto anual) | S |
| `sensitivity.delta` | 0.10 | S |

Ningún número normativo se escribe de memoria: cada valor F lleva cita y fecha de consulta en el comentario del YAML. Un valor F sin verificar queda `null` con el comentario "PENDIENTE: fuente" y el código lo rechaza con un mensaje claro.

## 7. Contrato de `predict.py`

```python
list_routes() -> pd.DataFrame
    # columnas: ruta_id, nombre, corredor, sentido, ruta_regreso

forecast(route_id: str, quantile: float | None = 0.9,
         op_params: dict | None = None, *, origen: str | None = None) -> pd.DataFrame
    # quantile: None -> plan con la media; 0.9 -> plan con P90; otro valor -> ValueError
    # op_params: sobrescribe claves de m1_operacion.yaml; clave desconocida -> ValueError
    # origen: None -> día siguiente al último dato (2026-03-31); o una fecha de prueba para reproducir
    # columnas: fecha, ruta_id, horizonte (1..30), pronostico, p90, base_plan,
    #           despachos, vehiculos (corredor), conductores (corredor, conductores-día), real (observado o NaN)
    # df.attrs: modelo, origen, ultimo_dato, flota_corredor, plantilla_conductores, advertencias
```

- Carga desde `models/m1_demanda/`, sin reentrenar ni internet, con caché (`lru_cache`); objetivo < 2 s por llamada.
- Si no hay artefactos o la ventana tiene NaN (p. ej. enero de 2026 en Duitama→Sogamoso), respaldo al ingenuo estacional con `attrs["modelo"] = "ingenuo_estacional"` y una advertencia.
- `predict.py` v0 (Tarea 6) implementa el contrato completo con el ingenuo y la operación real; se genera además `tests/fixtures/m1_forecast_ejemplo.csv` para la Persona 5. Pendiente de acordar con la Persona 5: si `df.attrs` le sirve (pandas lo marca como experimental) o si prefiere `(df, resumen)`.

## 8. Riesgos

- Enunciado no disponible como texto: tabla de requisitos y sección "Por confirmar" (sección 11).
- Cómputo: modelos pequeños, parada temprana, medir en la Tarea 9 antes de fijar el presupuesto.
- MPS puede no reproducir resultados con la misma semilla: CPU por defecto, se verifica en la Tarea 9.
- NaN en la ventana de producción: respaldo al ingenuo con advertencia y prueba.
- Fuga de información (escaladores, columnas contemporáneas, rezagos menores que h, selección con prueba): columnas prohibidas validadas por código, escaladores solo con entrenamiento, pruebas de fechas, selección solo con validación.
- La prueba es temporada alta y hay pocos fines de año para entrenar: métricas separadas en temporada alta y base.
- Parámetros normativos inventados: marcas F/S y nada sin cita.
- Demo > 2 min: `demo` solo carga artefactos; una prueba mide el tiempo.

## 9. Checklist de verificación y puntos típicos de alucinación

1. Ingenuo estacional: para horizonte h ≥ 7, usar el mismo día de la semana más reciente antes del origen (rezago 7·⌈h/7⌉ con h = 1..30; la primera versión de este plan decía 7·⌈(h+1)/7⌉, que es incorrecta: para h = 7 daría 14 en vez de 7. Ver H-M1-5). Usar y_{t−7} con t > o+6 es fuga.
2. MASE: denominador = MAE del ingenuo con m = 7 en entrenamiento, nunca en prueba; agregar entre rutas con MASE.
3. Pérdida cuantílica con signo invertido da un P10; la prueba sintética debe dar cobertura ≈ 0,90.
4. `PoissonNLLLoss(log_input=True)` espera log μ.
5. `torch.distributions.NegativeBinomial`: verificar qué representa `probs`.
6. `nn.LSTM` usa `batch_first=False` por defecto.
7. Variables prohibidas: `despachos`, `pasajeros_transito` del día objetivo y `evento_no_previsible`.
8. La "mejor semilla" para la app se elige por validación, no por prueba.
9. Métricas sobre valores no imputados.
10. Parámetros entrenables: solo `requires_grad`, incluido el embedding.
11. Operación: ciclo = ida + vuelta + alistamiento; vehículos con el máximo por sentido, no la suma.
12. Normativa (Ley 2101 de 2021, máximo de conducción diaria, SMMLV, sillas por clase) requiere cita verificada.
13. Fijar la versión de torch realmente instalada en Python 3.12 (instalada: 2.15.0), no de memoria.
14. Los 10 orígenes se solapan; no son observaciones independientes en pruebas estadísticas.
15. Makefile: `python -m m1_demanda.run`.

## 10. Tareas

| # | Tarea | Demo / criterio de terminado |
|---|---|---|
| 1 | Escribir este plan y contrastarlo con el enunciado | Documento en el repo con la tabla de requisitos |
| 2 | Dependencias, `common/` mínimo, `global.yaml`, Makefile | 30 semillas de una función de juguete con resumen; `tests/test_common.py` |
| 3 | `m1_demanda/ventanas.py` (ventanas sin fuga) | `python -m m1_demanda.ventanas` imprime ventanas y dimensiones; `tests/test_m1_ventanas.py` |
| 4 | `rolling_origin.py`, `evaluate.py`, ingenuo, `run.py --mode evaluate` | `reports/tables/m1_metricas_ingenuo.csv` |
| 5 | `m1_operacion.yaml`, `operacion/fleet.py`, `operacion/drivers.py` | Tabla del corredor 1 con el ingenuo; `tests/test_m1_operacion.py` |
| 6 | `predict.py` v0 y entrega a la Persona 5 | `forecast("50001-11001", 0.9)`; `tests/test_m1_contrato.py`; fixture CSV |
| 7 | Baseline GLM Poisson por ruta | Tabla ingenuo vs GLM |
| 8 | `losses.py` y diagnóstico de sobredispersión | `reports/tables/m1_sobredispersion.csv`; pruebas de pérdidas |
| 9 | MLP, `train.py`, medición CPU/MPS | Tabla ingenuo / GLM / MLP; dispositivo por defecto con evidencia |
| 10 | Codificador recurrente (LSTM/GRU) | Fila RNN en la tabla |
| 11 | `search.py` (búsqueda aleatoria con presupuesto) | `--mode search --presupuesto 3` en minutos |
| 12 | 30 semillas y tabla de comparación justa | `reports/tables/m1_comparacion.csv` y figuras |
| 13 | `forecast_value.py` y `peak_season.py` | `reports/tables/m1_valor_pronostico.csv` |
| 14 | `sensitivity.py` (±10 %) | `reports/figures/m1/sensibilidad_tornado.png` |
| 15 | `predict.py` v1, demo < 2 min y documentación | `make demo-m1` < 120 s; `python -m pytest -q` completo |

## 11. Contraste con el enunciado (`docs/enunciado_trabajo3.txt`)

El texto del enunciado se extrajo del PDF el 2026-11-22 y está en el repositorio. Esta tabla cruza **solo los requisitos del Módulo 1 y los transversales que tocan a la Persona 2** con lo implementado. Estado: ✅ hecho y verificado, 🟡 parcial, ⏳ pendiente de la corrida final o de un tercero.

### Tareas del módulo
| # | Requisito del enunciado | Estado | Dónde |
|---|---|---|---|
| 5 | Baselines: ingenuo estacional y modelo estadístico/lineal | ✅ | `baselines.py` (ingenuo; GLM Poisson con rezagos y calendario) |
| 6 | MLP con ventanas + arquitectura secuencial; optimizar ventana, capas, unidades, regularización, optimizador, lr; justificar la búsqueda y modelo global vs por ruta | ✅ | `nn_models.py`, `search.py`; decisiones D3, D8 |
| 7 | Pérdida justificada por ser conteo + al menos un cuantil alto | ✅ | `losses.py`; `docs/math_foundations.md` §1 y §6 |
| 8 | Evaluación con origen rodante: MAE, RMSE, MASE por ruta y agregado | ✅ | `evaluate.py` |
| 8 | **Error por horizonte en tramos 1–7, 8–14, 15–30** | ✅ (agregado tras leer el enunciado) | `evaluate.por_tramos`, `reports/tables/m1_error_tramos.csv` |
| 8 | Cobertura y pérdida cuantílica | ✅ | `evaluate.py` |
| 8 | Gráficas de pronóstico vs real **por ruta**, con temporada alta en prueba | ✅ (10 rutas, 2 orígenes) | `reporte.py`, `m1_pronostico_rutas_*.png` |

### Preguntas de negocio
| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 1 | Despachos y vehículos por ruta y día; fórmula justificada; efecto de ±10 % de demanda y parámetro más influyente | ✅ | `fleet.py`, `sensitivity.py`, `m1_sensibilidad.csv`, `m1_sensibilidad_flota.png` |
| 2 | Conductores con jornada máxima, **descansos** y ausentismo; normativa desde fuente oficial | 🟡 descansos agregados (CST art. 172, 6 de 7 días); Ley 2101 verificada en fuentes secundarias, **falta contrastar con el texto oficial** | `drivers.py`, `configs/m1_operacion.yaml` |
| 3 | Valor del pronóstico: pasajeros sin atender y **vehículos-día sobrantes** con la mejor red y el mejor baseline; decisión según costo relativo; ¿justifica la red su complejidad? | 🟡 tablas y figuras listas con las redes finales (`m1_valor_pronostico.csv`); **falta redactar la discusión** | `forecast_value.py`, `valor.py` |
| 4 | Temporada alta: vehículos **y conductores** adicionales; recomendación propia/arrendada/convenio | 🟡 vehículos y conductores adicionales calculados (`m1_temporada_alta.csv`); **falta la recomendación propia/arrendada/convenio** (decisión de negocio a redactar) | `peak_season.py` |
| — | Limitaciones: origen vs vía, sentido de regreso, demanda censurada | ✅ en el código y los docs; ⏳ redacción final en el reporte | `docs/math_foundations.md` §3, `m1_censura_indicios.csv` |

### Fundamentación matemática y figuras mínimas
| Requisito | Estado |
|---|---|
| Ecuaciones de la arquitectura secuencial | ✅ `math_foundations.md` §5 (verificadas contra `nn.GRU` y `nn.LSTM`) |
| Derivación de Poisson y pinball a partir de un modelo probabilístico | ✅ §6 |
| Definición de MASE, cobertura y pérdida cuantílica | ✅ §2 |
| Formulación de flota y conductores | ✅ §3 y §4 |
| Figura: series de las rutas; descomposición MSTL | ✅ de la Persona 1 (`eda_02`, `eda_03`) |
| Figura: pronóstico vs real por ruta con banda del cuantil | ✅ |
| Figura: error por horizonte (días 1 a 30) para cada modelo | ✅ `m1_error_por_horizonte.png` (se regenera con `--mode final`) |
| Figura: flota y conductores por día en los 30 días | ✅ `m1_flota_conductores_dia.png` |
| Figura: sensibilidad frente a la demanda y al parámetro más influyente | ✅ `m1_sensibilidad_flota.png` |

### Regla de comparación justa y entregables transversales
| Requisito | Estado |
|---|---|
| Mismos datos y particiones para todos los modelos | ✅ |
| Presupuesto de búsqueda explícito: configuraciones y tiempo de cómputo | ✅ `m1_busqueda_*.json` (MLP 30 en 157 s; RNN 30 en 921 s) |
| 30 semillas por configuración final: media, desviación, mejor y peor | ✅ MLP y GRU, 2 configuraciones cada una (`m1_semillas_*.csv`, `m1_comparacion.csv`) |
| Número de parámetros entrenables | ✅ `m1_comparacion.csv` |
| Figuras comunes: arquitectura, curvas de aprendizaje, resultados de la búsqueda, cajas de las 30 semillas | ✅ MLP y GRU |
| Discusión: ¿qué aportó la red frente al modelo de referencia? (métricas, variabilidad, costo, interpretabilidad, negocio) | ⏳ se escribe con los resultados finales |
| Cacería de la alucinación: ≥ 5 hallazgos con prompt literal, ≥ 1 por módulo, ≥ 3 categorías | 🟡 8 hallazgos del Módulo 1 + 1 verificación; el prompt se resumió en lugar de ser literal (ver nota); **falta M2 y M3** (otras personas) |
| Sección de uso de IA | ✅ `docs/ai_usage.md` (parte de la Persona 2) |
| Todo corre con un solo comando; parámetros por YAML o CLI; demo < 2 minutos | ✅ `make demo-m1` (3 s) |
| Probar en un clon limpio, en otra máquina | ⏳ pendiente |
| Historial de commits que refleje el trabajo | ⏳ **no se ha hecho ningún commit** de este trabajo |

> **Nota sobre "prompt literal".** El enunciado pide el prompt **literal** de cada hallazgo. En `docs/hallucination_hunt.md` los prompts de la Persona 2 están resumidos con fidelidad, no copiados: la Persona 2 debe pegar el texto exacto de sus mensajes antes de entregar.

### Desviaciones respecto al plan original (y por qué)
1. **Poisson sobre `y/ℓ`** (peso 1/ℓ por ventana) en lugar de sobre `y`: para que las 10 rutas pesen parecido en el entrenamiento; el GLM usa la misma pérdida (comparación justa).
2. **Corrección de una fórmula del plan**: el rezago del ingenuo es `7·⌈h/7⌉`, no `7·⌈(h+1)/7⌉` (H-M1-5).
3. **El oráculo no es el plan más barato** con φ < 1 (H-M1-10); la propiedad probada es "cero pasajeros sin atender".
4. **Con los costos supuestos, planear con P90 no compensa** (cuantil óptimo ≈ 0,72, no 0,90): conclusión que depende de supuestos [S] y se presenta con su barrido de costos.
5. **Parámetros normativos**: solo la jornada semanal (Ley 2101), el máximo de conducción (Convenio 153, **sin confirmar su ratificación**), el SMMLV 2026 y el descanso semanal (CST art. 172) llevan fuente; capacidades, tiempos de viaje, ocupación y costos son supuestos declarados.
6. **Se agregaron** tramos de horizonte, descansos, vehículos-día sobrantes, tabla de arquitectura, curvas de aprendizaje y cajas de semillas, que el plan original no incluía y el enunciado exige.
