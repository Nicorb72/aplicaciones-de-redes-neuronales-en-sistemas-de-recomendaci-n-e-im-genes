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

## Modelos
(Persona 2)

## Operación
(Persona 2)
