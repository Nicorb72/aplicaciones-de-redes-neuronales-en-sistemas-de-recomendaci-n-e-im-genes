# Uso de IA

## Módulo 1 — Modelos y operación (Persona 2)

**Herramienta:** Kiro, el agente de IA integrado en el IDE (modelos Claude Opus 5.5 para la planificación y Claude Sonnet 5.5 para la implementación). Sesión del 2026-11-22. El agente lee y escribe archivos del repositorio, ejecuta comandos y consulta la web.

### Qué se le pidió y qué hizo

| Fase | Pedido (resumido) | Qué hizo la IA |
|---|---|---|
| Planificación | Un plan técnico (sin código) para el alcance de la Persona 2 en el Módulo 1 | Leyó el repo y `docs/m1_datos_y_eda.md`, hizo 3 preguntas de aclaración (enunciado, hardware y framework, origen de los parámetros de operación), y propuso 10 decisiones de diseño con alternativas. La Persona 2 aprobó las opciones marcadas como recomendadas. |
| Implementación | Ejecutar el plan por tareas | Escribió el código de `m1_demanda/` (ventanas, evaluación, baselines, pérdidas, redes, búsqueda, operación, `predict.py`, `run.py`), las pruebas de `tests/`, la configuración de `configs/`, y los documentos de `docs/`. |
| Normativa | Parámetros de operación "con fuentes citadas" | Buscó en la web la Ley 2101 de 2021, el máximo de conducción, el salario mínimo 2026 y la capacidad por clase; **no aceptó una cifra sin fuente** (ver H-M1-6). |

### Qué decidió la persona y qué decidió la IA

* **La persona decidió:** el alcance (solo Módulo 1), PyTorch como framework, las opciones de diseño D1 a D10, que `common/` lo escribiera la Persona 2 con revisión de la Persona 5, y la aprobación del plan antes de implementar.
* **La IA propuso y la persona aprobó:** la pérdida de Poisson más una cabeza cuantílica para el P90, el modelo global con embedding de ruta, la búsqueda aleatoria con presupuesto, y el modelo de viaje redondo para la flota.
* **La IA decidió dentro del plan** (y quedó documentado para que la persona lo revise y pueda defenderlo): la Poisson sobre la escala `y/ℓ`, las variables del GLM, el espacio de búsqueda, y que el dispositivo por defecto sea CPU tras medirlo.

### Cómo se verificó el trabajo de la IA

* **Pruebas automáticas** (`python -m pytest -q`): sobre ventanas sin fuga (se corrompen los datos desde el origen y la entrada no cambia), métricas contra ejemplos calculados a mano, pérdidas contra `nn.PoissonNLLLoss` y `scipy`, operación con un ejemplo a mano por corredor y propiedades de monotonía, y el contrato de `predict.py`.
* **Ejecución real**, no solo lectura del código: cada entregable se corrió y se leyeron sus números (p. ej. `reports/tables/m1_metricas_*.csv`, `m1_sobredispersion.csv`).
* **Revisión visual** de cada figura generada; se corrigieron defectos que la IA no había notado al escribir el código (leyenda sobre las barras, series indistinguibles).
* **Fuentes primarias** para lo normativo, y declaración explícita de lo que no se pudo abrir.

### Dónde se equivocó la IA

Ocho errores concretos y verificables, con su evidencia y corrección, en `docs/hallucination_hunt.md` (H-M1-5 a H-M1-12; el último: evaluar una red con ventana de 56 días fallaba por la CLI): un rezago mal calculado en el plan, una norma atribuida sin fuente, una conversión a `float64` que falla en GPU, una prueba con un supuesto matemático falso (pérdida mínima igual a cero), una función `main()` perdida que un `import` no detectó, la afirmación de que el oráculo es el plan más barato, y una edición fuera del alcance en el Makefile. Además se documenta una **verificación** que no cuenta como error (V-M1-1: las ecuaciones de la GRU y la LSTM se comprobaron contra `nn.GRU` y `nn.LSTM`).

### Limitaciones y lo que la persona debe revisar por sí misma

* Los parámetros marcados `[S]` en `configs/m1_operacion.yaml` (capacidades, tiempos de viaje, ocupación, costos) son **supuestos**, no datos; cualquier cifra de costos que se cite depende de ellos.
* El texto oficial de la Ley 2101 de 2021 no se pudo abrir desde el equipo de desarrollo; los escalones (47, 46, 44 y 42 h) coinciden en seis fuentes secundarias y falta contrastarlos con el artículo 3.
* Colombia no figura entre los Estados que ratificaron el Convenio 153 de la OIT en las fuentes consultadas; el máximo de 9 h de conducción se usa como referencia.
* La persona debe poder explicar en vivo cada decisión; el plan (`docs/plan_persona2.md`) y `docs/math_foundations.md` están escritos con ese fin.
