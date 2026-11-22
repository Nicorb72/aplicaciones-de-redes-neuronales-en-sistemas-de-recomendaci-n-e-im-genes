.PHONY: setup data-m1 download-m1 eda-m1 test-m1 demo-m1 demo-m2 demo-m3 train-m1 train-m2 train-m3 app figures

setup:
	@echo "TODO: configurar el entorno"

# --- Módulo 1: datos (Persona 1) ---
# Descarga desde datos.gov.co (solo si faltan los crudos; requiere internet, ~7 min)
download-m1:
	python -m m1_demanda.data.download

# Construye series, calendario y particiones desde los crudos versionados (sin internet, <5 s)
data-m1:
	python -m m1_demanda.data.build_series

# Figuras y tablas del análisis descriptivo (sin internet, <1 min)
eda-m1: data-m1
	python -m m1_demanda.eda

test-m1:
	python -m pytest tests/test_m1_datos.py -q

demo-m1:
	python m1_demanda/run.py --mode demo

demo-m2:
	python m2_conduccion/run.py --mode demo

demo-m3:
	python m3_recomendacion/run.py --mode demo

train-m1:
	python m1_demanda/run.py --mode train

train-m2:
	python m2_conduccion/run.py --mode train

train-m3:
	python m3_recomendacion/run.py --mode train

app:
	streamlit run app/app.py

figures:
	@echo "TODO: generar figuras"
