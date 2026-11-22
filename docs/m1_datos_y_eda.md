# Módulo 1 — Datos y análisis descriptivo de la demanda

**Responsable:** Persona 1 (series de tiempo: datos y análisis).
**Entregable:** datos procesados + análisis exploratorio, insumo de la Persona 2 (modelos y operación).

Todo lo que aparece aquí se regenera con:

```bash
make data-m1   # series, calendario y particiones desde los crudos versionados (sin internet, <5 s)
make eda-m1    # figuras y tablas de este documento (sin internet, <1 min)
make test-m1   # 18 pruebas del pipeline
```

Las figuras están en `reports/figures/m1/` y las tablas en `reports/tables/m1_*.csv`.
La numeración (Figura M1-x, Tabla M1-x) es provisional; se ajusta al consolidar el reporte.

---

## 1. Caso de uso (Tarea 1)

| Pregunta | Respuesta |
|---|---|
| **¿Quién usa el pronóstico?** | El área de programación de la operación (jefe de despachos / planeador de flota) de la empresa de transporte intermunicipal. |
| **¿Para qué decisión?** | Cuántos despachos programar por ruta y día, cuántos vehículos asignar (propios, arrendados o por convenio) y cuántos conductores turnar. |
| **¿Con cuánta anticipación?** | 30 días: es el plazo en que se publica la programación mensual de turnos de conductores y en que se contratan vehículos adicionales para temporadas. |
| **¿Qué información hay disponible en ese momento?** | (a) La historia de pasajeros por ruta y día hasta el día anterior al origen del pronóstico; (b) el calendario: festivos (fijados por ley), puentes, Semana Santa, temporadas escolares y jornadas electorales, todos conocidos con meses de anticipación. |
| **¿Qué NO está disponible?** | Eventos imprevisibles (paros, cierres de vías, clima extremo), la demanda de los próximos 30 días y —si se usa solo el dato público— el último mes: la fuente se actualiza mensualmente con ~1 mes de rezago (la versión descargada el 2026-11-21 llega a abril de 2026, con abril incompleto). |

**Restricción que se respeta en todo el pipeline:** ninguna variable exógena depende de información no conocida 30 días antes. Por eso el Paro Nacional de 2021 se marca (`evento_no_previsible`) para el análisis y para enmascarar el entrenamiento, pero **no** se entrega como predictor.

> **Nota para la operación real.** Con los datos públicos, el rezago de publicación hace que el horizonte efectivo sea de 31–60 días. Una empresa real usaría sus propios registros de despacho (diarios) y el rezago desaparecería. En el trabajo se simula la situación ideal: el origen del pronóstico tiene datos hasta el día anterior.

---

## 2. Fuentes y variables

### 2.1 Conjunto principal

Superintendencia de Transporte (2026), *Operación de pasajeros y despacho de vehículos en la modalidad de transporte de pasajeros por carretera*, datos.gov.co, identificador `eh75-8ah6`. Según la ficha, corresponde a "los datos de movilidad de pasajeros y despachos de vehículos desde las terminales terrestres de pasajeros habilitadas y/o homologadas del país", con "datos disponibles desde 1 de enero de 2021" y actualización mensual.

**Tabla M1-1. Tamaño del conjunto** (fuente: `data/m1_demanda/raw/resumen_anual.csv`, consultado el 2026-11-21).

| Concepto | Valor |
|---|---|
| Registros | **49 202 812** (39 124 098 de tipo ORIGEN y 10 078 714 de TRÁNSITO) |
| Variables | **10** |
| Periodo declarado | desde 2021-01-01 |
| Periodo con reporte completo | 2021-01-01 a 2023-12-31; 2025-01-01 a 2025-11-29; 2026-01-01 a 2026-03-30 |
| Rutas origen–destino con despachos en origen | 1 983 |
| Unidad de registro | terminal × fecha × hora × clase de vehículo × nivel de servicio × ruta × tipo de despacho |

**Tabla M1-2. Diccionario de variables** (descripción textual del diccionario publicado por la fuente, `reports/tables/m1_diccionario_variables.csv`).

| Variable | Descripción según el diccionario | Valores observados |
|---|---|---|
| `terminal` | "Corresponde a la sede de la terminal desde donde se realizan y reportan los despachos" | p. ej. "T.T. DE CALI" |
| `clase_vehiculo` | "Identificador numérico dado por el RUNT" | **texto**: BUS, MICROBUS, BUSETA, CAMIONETA, AUTOMOVIL, CAMPERO |
| `nivel_servicio` | "1 BASICO - CORRIENTE 2 LUJO 3 CORRIENTE DIRECTO" | **texto**: BASICO, LUJO, PREFERENCIAL DE LUJO |
| `municipio_origen_ruta` | "Código Divipola del municipio de origen de la ruta" | código de 5 dígitos |
| `municipio_destino_ruta` | "Código Divipola del **departamento** de destino de la ruta" | código de **municipio** de 5 dígitos |
| `fecha_despacho` | "Fecha de realización del despacho. date (DD/MM/YYYY)" | ISO 8601 (`2022-06-14T00:00:00.000`) |
| `hora_despacho` | "Hora 24H (hh)" | "00"–"23" |
| `tipo_despacho` | "1 = Si el despacho de la ruta es en origen, y 2 = Si el despacho de la ruta es en tránsito" | **texto**: ORIGEN, TRANSITO |
| `despachos` | "Número vehículos de transporte de pasajeros por carretera que salen desde la terminal" | entero ≥ 0 |
| `pasajeros` | "Número de pasajeros que abordaron viaje desde la terminal de transporte. Las terminales en tránsito sólo deben reportar los pasajeros que abordaron el vehículo en dicha terminal." | entero ≥ 0 |

### 2.2 Inconsistencias entre el diccionario y los datos

| # | Inconsistencia | Evidencia | Tratamiento |
|---|---|---|---|
| I1 | `clase_vehiculo` se describe como "identificador numérico del RUNT", pero trae nombres en texto. | 6 valores de texto; ninguno numérico. | Se usa el texto. |
| I2 | `nivel_servicio` se describe con 3 códigos numéricos (BÁSICO, LUJO, CORRIENTE DIRECTO), pero trae texto y una categoría no documentada: "PREFERENCIAL DE LUJO" (5 478 registros); "CORRIENTE DIRECTO" no aparece. | Conteo por categoría. | No se usa en la serie objetivo (se suma sobre niveles). |
| I3 | `tipo_despacho` documentado como 1/2, publicado como ORIGEN/TRANSITO. | — | Se usa el texto. |
| I4 | `municipio_destino_ruta` dice "código Divipola del **departamento**", pero son códigos de **municipio** de 5 dígitos (p. ej. 52356 = Ipiales). | Cruce con la DIVIPOLA del DANE. | Se interpreta como municipio. |
| I5 | La ficha dice "datos desde el 1 de enero de 2021", pero hay 14 342 registros de 2019–2020. | `resumen_anual.csv` | Se descartan (fuera del periodo declarado y muy incompletos). |
| I6 | Fechas imposibles: años 2202, 2222 y 6202 (1 registro cada uno). | `resumen_anual.csv` | Se descartan. |
| I7 | `fecha_despacho` se documenta como DD/MM/YYYY; la API la entrega en ISO 8601. | — | Se parsea ISO. |
| I8 | **Huecos de reporte**: 2024 tiene 106 498 registros frente a ~12 millones de los demás años; dic-2025 tiene ~0,5 % de lo normal; abril de 2026 está a medio cargar. La ficha no lo advierte. | Figura M1-1. | Días inválidos → faltantes, nunca ceros (Sección 3). |
| I9 | Despachos con 0 pasajeros: ~15 % de los registros de origen en 2021 y ~8 % en 2025. | `resumen_anual.csv` | Se conservan y se cuentan aparte. |
| I10 | Saltos de nivel no explicados en algunas rutas (p. ej. Bogotá→Sibaté pasa de ~1 000 a ~5 000 pasajeros/día en 2025). | Tabla de candidatas. | Esa ruta no se eligió. |

### 2.3 Codificación DIVIPOLA

Los nombres de municipios se toman de DANE (2025), *DIVIPOLA – Códigos municipios*, datos.gov.co, identificador `gdxc-w37w` (actualización con corte al 30 de diciembre de 2024; 1 122 municipios), publicado por el propio DANE. Copia local: `data/m1_demanda/raw/divipola.csv`.

> ⚠️ Los códigos DIVIPOLA deben tratarse **siempre como texto**: el de Medellín es `05001`. Si se leen como número se pierde el cero inicial, y en YAML sin comillas `05615` se convierte en el entero octal 2957 (ver `tests/test_m1_datos.py::test_yaml_sin_comillas_es_octal`).

### 2.4 Variables exógenas (Tarea 4)

Archivo: `data/m1_demanda/external/calendario_colombia.csv` (2020–2027), generado por `m1_demanda/data/calendar_features.py`.

| Variable | Fuente primaria | ¿Se conoce 30 días antes? |
|---|---|---|
| `festivo`, `nombre_festivo` | Ley 51 de 1983 (traslado al lunes) y Ley 2578 de 2026, art. 6 (nuevo festivo del 9 de julio desde 2026). Implementación propia, verificada contra la librería `holidays` 0.109 para 2020–2027 (0 diferencias). | Sí (ley) |
| `puente`, `inicio_puente`, `ultimo_dia_puente`, `vispera_puente`, `dia_despues_puente` | Derivadas: bloque de ≥ 3 días no laborables consecutivos con al menos un festivo. | Sí |
| `semana_santa` | Domingo de Ramos a Domingo de Pascua (cómputo gregoriano, Meeus, 1991). | Sí |
| `receso_octubre` | Decreto 1373 de 2007 (compilado en el Decreto 1075 de 2015): semana anterior al festivo del 12 de octubre. | Sí |
| `vacaciones_mitad_anio`, `vacaciones_fin_anio` | Decreto 1075 de 2015 (40 semanas lectivas). Las fechas exactas las fija cada Secretaría de Educación: **se usan ventanas aproximadas configurables**. | Sí (resoluciones publicadas antes del año escolar) |
| `temporada_fin_anio` | 15-dic a 12-ene (ventana configurable). | Sí |
| `eleccion` | Calendario electoral de la Registraduría Nacional del Estado Civil (7 jornadas, 2022–2026). | Sí (meses antes) |
| `dow_sin/cos`, `doy_sin/cos` | Codificación cíclica del día de la semana y del año. | Sí |
| `evento_no_previsible` | Paro Nacional 2021: inicio el 28-abr-2021; fin aproximado el 15-jun-2021, definido con los datos (el sistema vuelve a ≥ 80 % de su mediana). | **No** → no usar como predictor |

---

## 3. Construcción de las series (Tarea 2)

### 3.1 Selección de rutas

Se descargaron, agregadas por ruta × día × clase de vehículo × tipo de despacho, las 40 rutas con más pasajeros en origen y sus sentidos de regreso (68 rutas candidatas). Se eligieron **5 corredores (10 series)** con estos criterios: alto volumen, cobertura > 99 % de los días válidos, **ambos sentidos reportados** (para incorporar el regreso en la flota) y diversidad de tipo de viaje.

**Tabla M1-3. Rutas analizadas** (`data/m1_demanda/processed/rutas.csv`).

| Corredor | Ida | Regreso | Perfil |
|---|---|---|---|
| 1 | Villavicencio (50001) → Bogotá (11001) | Bogotá → Villavicencio | Larga distancia, turismo de puente |
| 2 | Girardot (25307) → Bogotá | Bogotá → Girardot | Turismo de fin de semana (clima cálido) |
| 3 | Pereira (66001) → Armenia (63001) | Armenia → Pereira | Interurbano corto, Eje Cafetero |
| 4 | Pasto (52001) → Ipiales (52356) | Ipiales → Pasto | Corredor fronterizo |
| 5 | Duitama (15238) → Sogamoso (15759) | Sogamoso → Duitama | Pendular, intradepartamental |

Medellín→Rionegro, la ruta de mayor volumen, se descartó porque su sentido de regreso no aparece reportado.

### 3.2 Variable objetivo y agregación

Para la ruta *r* = (origen *o*, destino *d*) y el día *t*:

$$y_{r,t} = \sum_{i \,:\, \text{tipo}_i = \text{ORIGEN},\ o_i = o,\ d_i = d,\ \text{fecha}_i = t} \text{pasajeros}_i$$

sumando todas las terminales del municipio de origen, horas, clases de vehículo y niveles de servicio. Solo se usan los despachos de tipo **ORIGEN** para no contar dos veces el mismo viaje; los pasajeros de terminales de tránsito se guardan aparte (`pasajeros_transito`). Por el tamaño del conjunto, la agregación se hace en el servidor con consultas SoQL (`m1_demanda/data/download.py`).

### 3.3 Reglas de limpieza

Implementadas en `m1_demanda/data/clean.py`; los parámetros están en `configs/m1_datos.yaml`.

| Regla | Qué hace | Por qué |
|---|---|---|
| R1 Rango | Descarta fechas antes de 2021-01-01 o después del corte (2026-03-30). | I5, I6, I8 |
| R2 Días sin reporte del sistema | Si los registros nacionales del día son < 10 % de la mediana (32 209/día), el día es **faltante** en todas las rutas. 399 días: casi todo 2024, dic-2025 y algunos días sueltos. | Un día sin reporte no es un día sin pasajeros. El umbral 0,10 separa los huecos (< 8 %) del Paro Nacional (caída real hasta ~16 %) (Figura M1-1). |
| R3 Objetivo | Solo ORIGEN. | Evita doble conteo. |
| R4 Despachos con 0 pasajeros | Se conservan; `despachos_efectivos = despachos − despachos_cero_pax`. | Son vehículos que salieron; afectan la flota, no la demanda. |
| R5 Día de ruta sin registro o con 0 pasajeros, con el sistema reportando | Faltante. | Con > 1 000 pasajeros/día de media, un cero real es inverosímil: es falta de reporte de la terminal. |
| R6 Atípicos | (a) **Caída extrema:** < 15 % de la mediana del mismo día de la semana en ±8 semanas, salvo en festivos y 24/31-dic. (b) **z robusto** del log(1+y) frente a esa mediana con escala MAD, \|z\| > 5, salvo cerca (±2 días) de un evento de calendario. | Los picos de festivos son demanda real y se conservan; los derrumbes en días normales son fallas de reporte. |
| R7 Imputación | Huecos ≤ 14 días dentro de un segmento: promedio del mismo día de la semana anterior y siguiente → mediana del mismo día en ±4 semanas → interpolación lineal. Bandera `imputado`. | Conserva el patrón semanal. Los huecos largos no se imputan. |
| R8 Segmentos | La serie se divide en segmentos continuos: 1 = 2021-01-01…2023-12-31; 2 = 2025-01-01…2025-11-29; 3 = 2026-01-01…2026-03-30. | Las ventanas de rezagos no deben cruzar el hueco de 2024. |

**Tabla M1-4. Bitácora de limpieza por ruta** (`reports/tables/m1_calidad_datos.csv`): días faltantes por ruta, atípicos, imputados, % de despachos con 0 pasajeros y % de pasajeros en tránsito. Resumen: entre 1 y 24 días sin reporte por ruta, entre 1 y 27 atípicos y 0 faltantes finales en validación y prueba. Quedan **dos huecos largos sin imputar** en entrenamiento/post-hueco: Villavicencio→Bogotá (14–29 de marzo de 2022, falla de reporte de la terminal) y Duitama→Sogamoso (enero de 2026, 16 días).

### 3.4 Archivos que entrega la Persona 1

| Archivo | Contenido |
|---|---|
| `data/m1_demanda/processed/series_diarias.csv.gz` | Panel largo ruta × día (19 150 filas): `pasajeros` (objetivo limpio con NaN), `pasajeros_imp` (con imputación de huecos cortos), `pasajeros_crudo`, despachos, banderas de calidad, `segmento`, `particion` y las variables de calendario. |
| `data/m1_demanda/processed/pasajeros_ancho.csv` | Matriz fecha × ruta con `pasajeros_imp`. |
| `data/m1_demanda/processed/rutas.csv` | Catálogo: nombres DIVIPOLA, corredor, sentido y ruta de regreso. |
| `data/m1_demanda/processed/mezcla_clase_vehiculo.csv` | Participación de cada clase de vehículo en los despachos y pasajeros por despacho, **solo con datos de entrenamiento**. Insumo para la flota. |
| `data/m1_demanda/processed/particiones.json` | Fechas de las particiones y orígenes rodantes de prueba. |
| `data/m1_demanda/processed/validez_sistema.csv` | Validez de cada día y segmentos. |
| `data/m1_demanda/external/calendario_colombia.csv` | Variables exógenas. |

### 3.5 Particiones temporales

`m1_demanda/data/split.py`, configurables en `configs/m1_datos.yaml`:

| Partición | Fechas | Para qué |
|---|---|---|
| Entrenamiento | 2021-01-01 – 2023-06-30 | Ajuste. Incluye 2 Semanas Santas, 2 fines de año y la recuperación pospandemia. |
| Validación | 2023-07-01 – 2023-09-30 | Hiperparámetros y parada temprana. |
| Prueba | 2023-10-01 – 2023-12-31 | **Temporada alta**: receso de octubre, 3 puentes festivos, Navidad y fin de año. |
| Post-hueco | 2025-01-01 – 2026-03-30 | Evaluación adicional fuera de tiempo y pronóstico "en producción" de la app. No se usa para elegir modelos. |

Orígenes rodantes de prueba (horizonte de 30 días): cada 7 días desde el 2023-10-01, más el último posible (2023-12-02) para cubrir hasta el 31 de diciembre: **10 orígenes**.

**¿Por qué no se prueba en 2025–2026?** Diciembre de 2025 no tiene datos y el segmento posterior al hueco no tiene un año completo de historia previa (rezago de 365 días) sin cruzar el hueco de 2024. El bloque 2021–2023 es el único continuo con estacionalidad anual observable.

---

## 4. Análisis descriptivo (Tarea 3)

### 4.1 Figuras

| Figura | Archivo | Qué muestra |
|---|---|---|
| M1-1 | `eda_01_cobertura_sistema.png` | Registros diarios de todo el sistema / mediana; huecos de 2024 y dic-2025. |
| M1-2 | `eda_02_series_rutas.png` | **Series de demanda de las 10 rutas** (figura mínima). |
| M1-3 | `eda_03_descomposicion_50001-11001.png` | **Descomposición MSTL** (semanal y anual) de Villavicencio→Bogotá (figura mínima). |
| M1-4 | `eda_04_perfil_dia_semana.png` | Demanda relativa por día de la semana. |
| M1-5 | `eda_05_estacionalidad_mensual.png` | Demanda relativa por mes (2022–2023). |
| M1-6 | `eda_06_efecto_festivos_puentes.png` | Índice relativo en vísperas, puentes, festivos y elecciones. |
| M1-7 | `eda_07_semana_santa_fin_anio.png` | Perfil diario alrededor de Semana Santa y de Navidad/Año Nuevo. |
| M1-8 | `eda_08_recuperacion_pospandemia.png` | Nivel mensual relativo a 2023 y Paro Nacional 2021. |
| M1-9 | `eda_09_ida_vs_regreso.png` | Ida vs. regreso semanal por corredor. |
| M1-10 | `eda_10_calidad_despachos_transito.png` | % de despachos sin pasajeros y pasajeros en tránsito. |
| M1-11 | `eda_11_acf_50001-11001.png` | Autocorrelación (rezagos 7, 14, 28, 364). |

### 4.2 Método para medir los efectos de calendario

Para cada día se calcula una **línea base** $b_{r,t}$ = mediana de $y_{r,t+7k}$, $k=\pm1,\dots,\pm4$, usando solo días **sin** evento. El **índice relativo** $I_{r,t}=y_{r,t}/b_{r,t}$ controla la estacionalidad semanal y la tendencia local: $I=1{,}30$ significa "30 % más que un día normal del mismo día de la semana". Para cada hipótesis se prueba $H_0:\ \text{mediana}(\log I)=0$ con la prueba de rangos con signo de Wilcoxon (unilateral, nivel 1 %), sin suponer normalidad. Para el día de la semana se usa Kruskal–Wallis por ruta.

### 4.3 Hipótesis y resultados

**Tabla M1-5. Hipótesis sobre los determinantes de la demanda** (`reports/tables/m1_hipotesis.csv`). "Índice" = mediana del índice relativo de todos los días-ruta del evento; "Rutas" = rutas cuyo índice mediano va en la dirección esperada (de 10).

| ID | Hipótesis | Índice | Rutas | p-valor | Veredicto |
|---|---|---|---|---|---|
| H1 | El día de la semana determina la demanda, con un patrón que depende del sentido | máx/mín = 1,36 | 10 | < 1e-12 (KW, la peor ruta) | ✅ Confirmada |
| H2a | La **víspera de un puente** aumenta la demanda | 1,12 | 10 | 2,5e-31 | ✅ Confirmada |
| H2b | El **último día de un puente** (retorno) aumenta la demanda | 1,07 | 6 | 3,0e-10 | ✅ Confirmada, **solo en el sentido de retorno** |
| H2c | Los festivos que no son día de retorno **reducen** la demanda | 0,82 | 9 | 2,6e-17 | ✅ Confirmada |
| H3 | La **Semana Santa** aumenta la demanda | 1,16 | 8 | 9,3e-15 | ✅ Confirmada |
| H4 | La **temporada de fin de año** aumenta la demanda | 1,18 | 10 | 1,6e-80 | ✅ Confirmada |
| H5a | Las **vacaciones escolares de mitad de año** aumentan la demanda | 1,08 | 8 | 7,0e-34 | ✅ Confirmada (efecto pequeño) |
| H5b | El **receso escolar de octubre** aumenta la demanda | 1,16 | 10 | 1,0e-36 | ✅ Confirmada |
| H6 | **Recuperación pospandemia**: la demanda crece de 2021 a 2023 y luego se estabiliza | 2021 = 0,64 × 2023 | 10 | 9,8e-4 | ✅ Confirmada |
| H7 | Las **jornadas electorales** reducen la demanda (hipótesis surgida del análisis de atípicos) | 0,68 | 10 | 6,2e-14 | ✅ Confirmada |

**Lectura de los resultados:**

- **H1 – Día de la semana (Figura M1-4).** El patrón depende del **sentido**, no solo del día. Bogotá→Girardot: viernes ×1,46 y sábado ×1,34 (salida de fin de semana); Girardot→Bogotá: domingo ×1,42 y lunes ×1,12 (retorno). Duitama↔Sogamoso es **pendular**: el domingo cae a ×0,74. Conclusión para el modelo: el día de la semana debe interactuar con la ruta, ya sea con un modelo por ruta o con un *embedding* de ruta en un modelo global.
- **H2 – Festivos y puentes (Figura M1-6).** El efecto más fuerte de todo el análisis: Bogotá→Girardot **×3,03** el primer día del puente y Girardot→Bogotá **×2,52** el último día. Las rutas pendulares de Boyacá **caen** en festivos (×0,48–0,54). Un festivo aislado no implica más demanda: implica **salida antes y retorno después**.
- **H3 – Semana Santa (Figura M1-7).** En Bogotá→Villavicencio la salida pico el Miércoles Santo (×3,4) y el retorno el Domingo de Pascua y el lunes (×2,1). El Viernes Santo es un valle (×0,57).
- **H4 – Fin de año (Figuras M1-5 y M1-7).** Diciembre es el mes más alto en 8 de 10 rutas (Bogotá→Girardot ×1,50 y Bogotá→Villavicencio ×1,41). Salidas de Bogotá con pico el 29–30 de dic (×3); retornos a Bogotá con pico del 3 al 12 de enero (×2,3). El 25 de diciembre y el 1 de enero son valles.
- **H5 – Temporadas escolares.** El receso de octubre (×1,16) pesa más que las vacaciones de mitad de año (×1,08). Las fechas escolares son aproximadas (Sección 2.4), así que el efecto estimado es un límite inferior.
- **H6 – Pospandemia (Figuras M1-2 y M1-8).** En 2021 la demanda estuvo en 0,64 veces el nivel de 2023 (mediana de las rutas). 2022 alcanzó 0,96 y 2025 está en 0,95: **la recuperación terminó en 2022–2023**. El Paro Nacional (28-abr a ~15-jun-2021) produjo la caída más profunda de la serie (hasta ~0,2) y no es previsible.
- **H7 – Elecciones.** Todas las rutas caen el día de elecciones (×0,68; Villavicencio→Bogotá ×0,44). Se detectó al revisar los atípicos: el 13-mar-2022, el 29-oct-2023 y el 8-mar-2026 aparecían como "atípicos" en todas las rutas a la vez. Como las fechas se conocen con meses de anticipación, se incorporaron al calendario y dejaron de tratarse como errores.

### 4.4 Tendencia y estacionalidad

**Descomposición MSTL** (Bandara et al., 2021) de log(1+y) para Villavicencio→Bogotá, 2021–2023 (Figura M1-3). Reparto de la varianza: tendencia 30 %, estacionalidad anual 26 %, semanal 6 % y **residuo 38 %**. El residuo grande no es ruido: contiene los picos de festivos y puentes, que **no son periódicos** (cambian de fecha cada año por la Ley 51 de 1983 y la Pascua). Esto justifica usar variables de calendario explícitas en lugar de depender solo de la estacionalidad anual.

**Autocorrelación** (Figura M1-11): picos en los rezagos 7, 14, 21 y 28 y un pico menor alrededor de 364. Sugerencia para la Persona 2: ventanas de al menos 28 días, y el rezago 364 como variable explícita si se usa.

### 4.5 Ida y regreso (limitación del enunciado)

**Tabla M1-6** (`reports/tables/m1_ida_vs_regreso.csv`):

| Corredor | Correlación semanal ida–regreso | Pasajeros regreso / ida |
|---|---|---|
| Villavicencio ↔ Bogotá | 0,81 | **0,43** |
| Girardot ↔ Bogotá | 0,54 | 0,61 |
| Pereira ↔ Armenia | 0,87 | 0,53 |
| Pasto ↔ Ipiales | 0,95 | 0,98 |
| Duitama ↔ Sogamoso | 0,79 | 0,73 |

Los corredores son **muy asimétricos**: desde Bogotá salen registrados menos de la mitad de los pasajeros que llegan desde Villavicencio. Como un vehículo que va debe volver, la flota de un corredor está dada por el **sentido más cargado de cada día** y por el tiempo de ciclo de ida y vuelta, no por la suma de las dos series. La asimetría se explica en parte por las limitaciones de la Sección 5.

### 4.6 Despachos vacíos y pasajeros en tránsito (Figura M1-10)

- **Despachos con 0 pasajeros:** 13–14 % en los corredores que salen o llegan a Bogotá; < 2 % en Pasto–Ipiales y Pereira–Armenia.
- **Pasajeros que suben en terminales de tránsito:** en Bogotá→Girardot son **1,6 veces** los que suben en origen; en Girardot→Bogotá, 0,42 veces. En el resto de corredores son < 3 %.
- **Pasajeros por despacho efectivo** (`mezcla_clase_vehiculo.csv`): los BUS de Bogotá→Girardot salen de la terminal con ~7 pasajeros de media, y los de Villavicencio→Bogotá con ~23. El dato de origen **no mide la ocupación del vehículo**.

---

## 5. Limitaciones de los datos

1. **Solo se observa a quien aborda en una terminal.** Quien sube en la vía no aparece (enunciado). En Bogotá→Girardot el tránsito supera al origen (Sección 4.6), y los buses salen de Bogotá con ~7 pasajeros registrados. El pronóstico de "pasajeros en origen" subestima la carga real del vehículo, así que la ocupación objetivo de la flota debería calibrarse con `pasajeros_transito`.
2. **Demanda censurada por la oferta.** Si no hay sillas, el pasajero no aparece. En los picos de puente (×2–3) es probable que la demanda real sea mayor que la observada, por lo que el pronóstico de temporada alta está sesgado hacia abajo. Esto refuerza dimensionar con un cuantil alto (P90).
3. **Huecos de publicación** (2024, dic-2025): no hay ningún diciembre posterior a 2023 para validar el pronóstico de fin de año.
4. **Sentido de regreso:** ver Sección 4.5.
5. **Agregación municipal:** una ruta suma todas las terminales del municipio de origen (p. ej. las terminales de Bogotá). No se puede separar por terminal sin perder continuidad.
6. **Fechas escolares aproximadas:** el calendario escolar real varía por Secretaría de Educación.

---

## 6. Notas para la Persona 2 (modelos y operación)

- **Objetivo:** `pasajeros_imp` (entero ≥ 0, conteo). Varianza alta y picos multiplicativos: conviene log1p o una pérdida de Poisson/binomial negativa.
- **Máscaras:** no entrenar sobre `segmento == 0`. Hay dos huecos largos con NaN en entrenamiento (Villavicencio→Bogotá, marzo de 2022) y post-hueco (Duitama→Sogamoso, enero de 2026): descartar las ventanas que los contengan. Considerar descartar o ponderar menos `evento_no_previsible == True`.
- **Exógenas válidas:** todas las columnas de calendario de `series_diarias.csv.gz`, excepto `evento_no_previsible`.
- **Modelo global vs. por ruta:** el perfil semanal y el efecto de los puentes **dependen del sentido** (H1, H2). Un modelo global necesita la ruta como entrada (one-hot o *embedding*) e interacciones con el calendario.
- **Ingenuo estacional:** el rezago 7 es fuerte (ACF), pero falla en puentes, que no son periódicos semanalmente. El ingenuo "mismo día del año pasado" (rezago 364) tampoco los captura, porque los festivos se mueven.
- **Flota:** usar `mezcla_clase_vehiculo.csv` para la capacidad media ponderada por ruta y el sentido más cargado del corredor (Sección 4.5).

---

## 7. Hallazgos para la "Cacería de la alucinación" (Módulo 1)

Errores reales de la IA (Claude Code) detectados al construir este pipeline. El detalle completo está en `docs/hallucination_hunt.md`.

1. **Código / Datos – DIVIPOLA en YAML:** la IA escribió los códigos sin comillas (`[05001, 05615]`); PyYAML los lee como enteros octales (05615 → 2957).
2. **Datos del mundo real – normativa desactualizada:** la lista de festivos generada por la IA no incluía el festivo creado por la Ley 2578 de 2026.
3. **Código – consulta SoQL:** la IA usó un alias igual al nombre de la columna (`sum(despachos) AS despachos`), lo que rompe la consulta.
4. **Código – preprocesamiento:** la primera regla de atípicos de la IA eximía cualquier día "cerca de un evento", y dejó pasar un día con 19 pasajeros (frente a ~1 000 normales) como dato válido.

---

## Referencias (APA 7.ª ed.)

- Bandara, K., Hyndman, R. J., & Bergmeir, C. (2021). *MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns* (arXiv:2107.13462). arXiv. https://doi.org/10.48550/arXiv.2107.13462
- Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics, 6*(1), 3–73.
- Congreso de Colombia. (1983, 6 de diciembre). *Ley 51 de 1983, por la cual se traslada el descanso remunerado de algunos días festivos*. https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=4954
- Congreso de Colombia. (2026, 1 de junio). *Ley 2578 de 2026, por medio de la cual se vincula a la Nación en el homenaje al municipio de Chiquinquirá…* Diario Oficial No. 53.510.
- Departamento Administrativo Nacional de Estadística [DANE]. (2025). *DIVIPOLA – Códigos municipios* [Conjunto de datos]. datos.gov.co. https://www.datos.gov.co/d/gdxc-w37w
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association, 47*(260), 583–621. https://doi.org/10.1080/01621459.1952.10483441
- Meeus, J. (1991). *Astronomical algorithms*. Willmann-Bell.
- Ministerio de Educación Nacional. (2007). *Decreto 1373 de 2007, por el cual se establece una semana de receso estudiantil*.
- Ministerio de Educación Nacional. (2015). *Decreto 1075 de 2015, Decreto Único Reglamentario del Sector Educación*.
- Superintendencia de Transporte. (2026). *Operación de pasajeros y despacho de vehículos en la modalidad de transporte de pasajeros por carretera* [Conjunto de datos]. datos.gov.co. https://www.datos.gov.co/d/eh75-8ah6
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin, 1*(6), 80–83. https://doi.org/10.2307/3001968

> **Estado de verificación (2026-11-21):** verificados en línea: los dos conjuntos de datos, la Ley 51 de 1983 (Función Pública), la Ley 2578 de 2026 (Diario Oficial No. 53.510) y los DOI de Bandara et al., Kruskal y Wallis y Wilcoxon (volumen, número y páginas confirmados en Crossref). **Pendiente:** confirmar en SUIN-Juriscol los datos de los Decretos 1373/2007 y 1075/2015 y el artículo exacto sobre el receso; agregar una fuente primaria para las fechas del Paro Nacional 2021 y para el calendario electoral (Registraduría). Cleveland et al. (1990), Hyndman y Athanasopoulos (2021) y Meeus (1991) son referencias clásicas sin DOI que no se comprobaron en línea.
