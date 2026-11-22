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
