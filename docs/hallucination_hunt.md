# Búsqueda de alucinaciones

## Módulo 1 — Datos (Persona 1)

Herramienta: Claude Code (modelo Claude Opus 5.5), sesión del 2026-11-21.
Prompt general de la sesión (literal): *"Ayudame a hacer este trabajo [enunciado del Trabajo 3] teniendo en cuenta que yo soy la persona 1 en la reparticion, basandote en las clases … y todo va a quedar en este repo"*.

### H-M1-1. Códigos DIVIPOLA convertidos a octal (categoría: Código / Datos del mundo real)
1. **Prompt:** el prompt general de la sesión; la IA generó `configs/m1_datos.yaml`.
2. **Respuesta de la IA:** `corredores: - [05001, 05615]  # Medellín – Rionegro`, sin comillas.
3. **Cómo se sospechó:** los códigos DIVIPOLA empiezan con 0 y YAML interpreta los números con cero inicial.
4. **Evidencia (experimento):** `yaml.safe_load('a: [05001, 05615, 50001]')` → `{'a': [2561, 2957, 50001]}`: 05615 se convierte en el entero octal 2957 y se pierde el código de Rionegro. El código lo prueba `tests/test_m1_datos.py::test_yaml_sin_comillas_es_octal`.
5. **Corrección:** códigos entre comillas, más una validación en `m1_demanda/data/utils.py::cargar_config` que rechaza códigos que no sean texto de 5 dígitos.
6. **Lección:** los identificadores con ceros a la izquierda (DIVIPOLA, códigos postales) siempre se tratan como texto, y se valida el tipo al cargar la configuración.

### H-M1-2. Lista de festivos desactualizada (categoría: Datos del mundo real – normativa)
1. **Prompt:** el prompt general de la sesión; la IA implementó `festivos_colombia()` según la Ley 51 de 1983.
2. **Respuesta de la IA:** 18 festivos por año (6 fijos, 2 de Semana Santa y 10 trasladables), sin el 9 de julio.
3. **Cómo se sospechó:** al comparar contra la librería `holidays` 0.109 aparecieron dos fechas que la IA no tenía: 2026-07-13 y 2027-07-12.
4. **Evidencia (fuente primaria):** Ley 2578 del 1 de junio de 2026, art. 6, publicada en el Diario Oficial No. 53.510 del 2 de junio de 2026: declara festivo nacional el Día de Nuestra Señora del Rosario de Chiquinquirá (9 de julio), con traslado según la Ley 51 de 1983.
5. **Corrección:** festivo agregado desde 2026, citado en el código; prueba `test_festivo_ley_2578_de_2026`.
6. **Lección:** las reglas normativas cambian; hay que contrastar contra una fuente independiente y actualizada y luego confirmar en la fuente oficial. No basta con "la ley que todo el mundo conoce".

### H-M1-3. Alias que sombrea una columna en SoQL (categoría: Código)
1. **Prompt:** el prompt general de la sesión; la IA escribió la consulta de agregación de `download.py`.
2. **Respuesta de la IA:** `SELECT ..., sum(despachos) AS despachos, ..., sum(case(pasajeros > 60 * despachos, 1, true, 0)) ...`
3. **Cómo se sospechó:** la API respondió HTTP 400 en todos los reintentos.
4. **Evidencia:** mensaje del servidor: `query.soql.aggregate-in-ungrouped-context; Aggregate function 'sum' used in ungrouped context`. Dentro de `case(...)`, `despachos` se refería al alias (un agregado) y no a la columna.
5. **Corrección:** alias `total_despachos` y `total_pasajeros`, renombrados después en pandas. Además, el código ya no reintenta ante un 400: muestra el mensaje del servidor.
6. **Lección:** nunca reutilizar como alias el nombre de una columna en SQL/SoQL, y leer el cuerpo del error en lugar de reintentar a ciegas.

### H-M1-4. Regla de atípicos que deja pasar fallas de reporte (categoría: Código – preprocesamiento)
1. **Prompt:** el prompt general de la sesión; la IA diseñó la regla R6 de `clean.py`.
2. **Respuesta de la IA:** "si |z| > 5 y el día NO está cerca de un evento de calendario se marca como atípico; los picos de festivos se conservan".
3. **Cómo se sospechó:** al revisar los faltantes que quedaban, Duitama→Sogamoso tenía 19 pasajeros el 2026-01-10 (z = −40) marcado como válido, frente a ~1 000 en un día normal.
4. **Evidencia:** el 10 de enero cae en la ventana de `temporada_fin_anio`, así que la exención por "evento cercano" se aplicaba también a derrumbes que ningún evento explica. La versión corregida produjo el efecto contrario: marcaba como error la caída real del 1-ene en una ruta pendular (81 pasajeros, también real en 2021, 2022 y 2025).
5. **Corrección:** se añadió una regla de "caída extrema" (< 15 % de la mediana del mismo día de la semana) que se aplica incluso cerca de eventos, excepto en los propios festivos y el 24/31-dic.
6. **Lección:** las reglas de limpieza se validan mirando los casos que marcan **y** los que dejan pasar, ruta por ruta, en vez de confiar en una regla que "suena razonable".

## Módulo 1 — Modelos y operación (Persona 2)

Herramienta: Kiro (agente de IA del IDE), modelos Claude Opus 5.5 (planificación) y Claude Sonnet 5.5 (implementación), sesión del 2026-11-22.
Prompts generales de la sesión (resumidos con fidelidad): (1) *"Actúa como planificador técnico… soy la Persona 2… NO escribas código todavía: genera solo un plan"* con el alcance de baselines, redes, operación e interfaces del Módulo 1; (2) *"Implement this plan"* con el plan aprobado (`docs/plan_persona2.md`).

### H-M1-5. Fórmula del rezago del ingenuo estacional con un desfase (categoría: Matemática / Código)
1. **Prompt:** el de planificación; la IA redactó el punto 1 de la lista de verificación del plan.
2. **Respuesta de la IA:** "para horizonte h ≥ 7, usar el mismo día de la semana más reciente antes del origen (**rezago 7·⌈(h+1)/7⌉**)".
3. **Cómo se sospechó:** la IA al implementar `baselines.py::rezago_ingenuo` y comprobar el caso h = 7.
4. **Evidencia:** con h = 1, …, 30 y el primer día pronosticado como h = 1, la fórmula da 14 para h = 7, es decir un rezago de dos semanas; el día de la semana más reciente antes del origen está a 7 días. La fórmula correcta es 7·⌈h/7⌉ (equivale a `7·(⌊k/7⌋+1)` con k = h−1). Lo verifica `tests/test_m1_evaluacion.py::test_rezago_ingenuo` (rezagos 7, 7, 14, 14, 21, 35 para k = 0, 6, 7, 13, 14, 29) y la prueba de que corromper todo dato desde el origen no cambia el pronóstico.
5. **Corrección:** `docs/plan_persona2.md` (lista de verificación, punto 1) y la implementación usan 7·⌈h/7⌉.
6. **Lección:** una fórmula con índices se comprueba con los casos frontera (aquí h = 7 y h = 8) antes de escribirla en un documento; un desfase de uno en un rezago mezcla "mismo día de la semana" con "dos semanas atrás".

### H-M1-6. Norma de conducción atribuida a Colombia sin verificar (categoría: Normativa)
1. **Prompt:** el de planificación; la IA propuso los parámetros de `m1_operacion.yaml` con fuente.
2. **Respuesta de la IA:** que la capacidad por clase y el máximo de conducción diaria saldrían de normativa colombiana ("Decreto 1079 de 2015 para el transporte" y "norma de jornada para conductores de transporte público, por identificar").
3. **Cómo se sospechó:** la propia IA al buscar las fuentes antes de escribir el YAML.
4. **Evidencia:** (a) la búsqueda no halló ningún artículo verificable que fije las sillas por clase de vehículo; (b) la Circular 87 de 2026 (Ministerio de Transporte y Ministerio del Trabajo, 2-mar-2026, https://jurinfo.jep.gov.co/Normograma/compilacion/docs/circular_mintransporte_0087_2026.htm) **no fija ningún número**: remite al Código Sustantivo del Trabajo y al Convenio 153 de la OIT; (c) el art. 6.1 del Convenio 153 (https://compilacionjuridica.antioquia.gov.co/compilacion/compilacion/docs/cv_oit_0153_79.htm) fija 9 h de conducción por día, pero en las fuentes consultadas no se pudo confirmar que Colombia lo haya ratificado.
5. **Corrección:** `capacity_by_class` y `cycle_time.viaje_h` quedan como supuestos [S] declarados; `max_conduccion_dia_h = 9` se cita como estándar internacional de referencia con la salvedad de la ratificación (`configs/m1_operacion.yaml`). Los escalones de la Ley 2101 de 2021 se confirmaron en 6 fuentes secundarias coherentes, pero el texto oficial no se pudo abrir (error de certificado y tiempo de espera): queda marcado "PENDIENTE: contrastar el art. 3 con el texto oficial".
6. **Lección:** "hay una norma que lo dice" no es una fuente; se cita el artículo y se distingue la norma nacional del estándar internacional de referencia. Si no se puede abrir la fuente primaria, se declara en el mismo archivo.

### H-M1-7. Conversión a float64 en un dispositivo que no la soporta (categoría: Código)
1. **Prompt:** el de implementación (Tarea 9, medición CPU frente a MPS).
2. **Respuesta de la IA:** en `m1_demanda/train.py::PronosticadorRed.predecir`: `mus.append(mu.double().cpu().numpy())`.
3. **Cómo se sospechó:** falló el primer `python -m m1_demanda.benchmark_dispositivo` en MPS.
4. **Evidencia:** `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.` Con CPU funcionaba, por eso las pruebas anteriores no lo mostraron.
5. **Corrección:** `mu.cpu().double().numpy()`: primero a CPU y luego a doble precisión.
6. **Lección:** el código que "funciona en CPU" no está probado en GPU; el orden `.cpu()` antes de cambiar de tipo evita las restricciones de cada backend.

### H-M1-8. Prueba de sobreajuste que suponía una pérdida mínima inexistente (categoría: Matemática / Código)
1. **Prompt:** el de implementación (Tarea 9).
2. **Respuesta de la IA:** `tests/test_m1_mlp.py::test_sobreajusta_un_lote_pequeno` exigía `pérdida_final < 0,6 × pérdida_inicial`.
3. **Cómo se sospechó:** la prueba falló (0,925 frente a 0,6 × 1,070 = 0,642) aunque la red sí aprendía.
4. **Evidencia:** la pérdida de Poisson sin el término constante `log y!` no llega a 0; su mínimo con f = log(y/ℓ) es mean(r·(1 − log r)), r = y/ℓ, que para el lote vale 0,9192. Tras 1.500 pasos la red llega a 0,9194: cierra el 99,7 % de la brecha hasta el mínimo teórico y el pinball baja de 0,0704 a 0,0029.
5. **Corrección:** la prueba compara contra el piso analítico (cierra más del 90 % de la brecha y nunca baja del mínimo); misma lógica en `tests/test_m1_rnn.py`.
6. **Lección:** al probar el sobreajuste, calcular el mínimo alcanzable de la pérdida concreta; "debe bajar a cero" solo vale para pérdidas que tienen el cero como mínimo.

### H-M1-9. `main()` perdida al editar un archivo y no detectada por un `import` (categoría: Código)
1. **Prompt:** el de implementación (Tarea 12, modo `final` de `run.py`).
2. **Respuesta de la IA:** una edición de `m1_demanda/run.py` reemplazó la cabecera `def main(...)` y la creación del `ArgumentParser` por otras funciones.
3. **Cómo se sospechó:** la búsqueda en segundo plano de la RNN terminó de inmediato.
4. **Evidencia:** `NameError: name 'main' is not defined` en `run.py`, línea 264. La comprobación `python -c "import m1_demanda.run"` había dado "ok" porque la parte rota solo falla al ejecutarse como script.
5. **Corrección:** se repuso `main()`; se agregó `tests/test_m1_cli.py`, que ejecuta `python -m m1_demanda.run --help` y otros casos como proceso.
6. **Lección:** un `import` no prueba un punto de entrada; hay que ejecutarlo. Tras reemplazar texto en un archivo, releer el resultado completo y no solo la parte editada.

### H-M1-10. "El oráculo es el plan más barato" (categoría: Matemática / diseño)
1. **Prompt:** el de planificación; la IA redactó la prueba de la Tarea 13.
2. **Respuesta de la IA:** "el oráculo (demanda real) tiene costo ≤ el de los demás planes".
3. **Cómo se sospechó:** la IA al diseñar `forecast_value.py` (con ocupación objetivo φ < 1 el oráculo planea con holgura) y confirmado con los números.
4. **Evidencia:** en `reports/tables/m1_valor_pronostico.csv`, `ingenuo_puntual` cuesta 1,1545·10¹⁰ frente a 1,1610·10¹⁰ del oráculo (−0,56 %) y deja 15.585 pasajeros sin atender frente a 0 del oráculo.
5. **Corrección:** la invariante probada es "el oráculo deja cero pasajeros sin atender" (`tests/test_m1_valor.py::test_el_oraculo_no_deja_pasajeros_sin_atender`); el docstring de `forecast_value.py` explica que un plan con menos capacidad puede costar menos.
6. **Lección:** con holguras (φ < 1) la información perfecta no minimiza el costo total; se prueba la propiedad que sí se cumple y se mide la otra.

### H-M1-11. Edición fuera del alcance en el Makefile (categoría: Código / alcance)
1. **Prompt:** el de implementación (Tarea 2: corregir `demo-m1`/`train-m1`).
2. **Respuesta de la IA:** un `sed` con expresión regular cambió también los objetivos de M2 y M3 (`python -m m2_conduccion.run`, `python -m m3_recomendacion.run`).
3. **Cómo se sospechó:** la IA al leer la salida del `grep` posterior.
4. **Evidencia:** `grep -n run Makefile` mostraba cuatro líneas modificadas de módulos de otras personas.
5. **Corrección:** se revirtieron M2 y M3; `git diff Makefile` muestra solo `demo-m1` y `train-m1`.
6. **Lección:** un cambio con patrón general se acota al alcance pedido (aquí, M1) y se verifica con `git diff` antes de seguir.

### V-M1-1. Verificación (NO es un error de la IA): ecuación de la GRU, puerta de reinicio antes o después de la matriz recurrente (categoría: Conceptos / Matemática)
> Esta entrada documenta una **verificación** que evitó un posible error; la IA no llegó a escribir la forma incorrecta. Se incluye porque el enunciado pide documentar las verificaciones hechas, y no se cuenta como hallazgo de alucinación.

1. **Prompt:** el de implementación (fundamentación matemática de la arquitectura secuencial, pedida por el enunciado).
2. **Qué se consideró:** al preparar las ecuaciones de la GRU había dos formas habituales del candidato: `n_t = tanh(W_in x_t + b_in + W_hn (r_t ⊙ h_{t−1}) + b_hn)` (reinicio *antes* de la matriz recurrente, como en la presentación original) y `n_t = tanh(W_in x_t + b_in + r_t ⊙ (W_hn h_{t−1} + b_hn))` (reinicio *después*).
3. **Por qué se verificó:** las presentaciones de la GRU no coinciden entre sí y la página dinámica de PyTorch no se pudo leer (la herramienta devolvió vacío), así que no se podía confiar en recordar cuál usa la librería.
4. **Evidencia (experimento):** un paso de `nn.GRU` con pesos aleatorios, comparado contra ambas fórmulas con los pesos de la propia capa: la forma "reinicio después" reproduce la salida con tolerancia 10⁻⁶ (`True`); la forma "reinicio antes" no la reproduce (`False`). Mismo ejercicio con `nn.LSTM` (orden de puertas `i, f, g, o`): coincide. También se verificó el conteo de parámetros `3(d·h + h² + 2h)` para la GRU (5.952 con d = 28, h = 32) y `4(·)` para la LSTM (7.936) contra `sum(p.numel())`.
5. **Resultado:** `docs/math_foundations.md` (sección 5) escribe la ecuación de PyTorch y aclara que la variante alternativa no coincide con la librería.
6. **Lección:** una ecuación de una arquitectura se verifica contra la implementación que se usa, no contra otra presentación de la misma idea; basta con un paso con pesos aleatorios.

### H-M1-12. Evaluar una red con ventana de 56 días fallaba por la línea de comandos (categoría: Código)
1. **Prompt:** el de implementación (Tareas 4, 9 y 12: protocolo de evaluación común a todos los modelos).
2. **Respuesta de la IA:** `evaluate.py::evaluar_modelo` construía siempre las ventanas con `cfg_m1["window"]` (28 días), y `run.py --mode evaluate --modelo mlp` cargaba una red cualquiera sin comprobar su ventana.
3. **Cómo se sospechó:** al cambiar `run.py` para evaluar la red **final** (la que se defiende) en lugar de la corrida suelta de una semilla.
4. **Evidencia:** `python -m m1_demanda.run --mode evaluate --modelo mlp` → `ValueError: El modelo espera ventanas de 56 días, no 28.` La red final (la mejor de la búsqueda) usa ventana de 56; las pruebas anteriores solo evaluaban redes de ventana 28 o pasaban las ventanas a mano, así que el defecto no aparecía. Tras la corrección, la evaluación de la semilla 2026 por CLI da MASE de validación 0,824, igual que en `reports/tables/m1_semillas_mlp.csv`.
5. **Corrección:** `evaluar_modelo` usa `modelo.hp["window"]` cuando el modelo lo declara y difiere de la configuración; prueba `tests/test_m1_mlp.py::test_evaluar_modelo_usa_la_ventana_del_modelo`.
6. **Lección:** un protocolo de evaluación "común" debe probarse con el modelo final real, no solo con modelos de juguete con la configuración por defecto; una función que recibe un modelo y una configuración debe comprobar que son compatibles.
