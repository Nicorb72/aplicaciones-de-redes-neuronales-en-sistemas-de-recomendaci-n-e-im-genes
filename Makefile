.PHONY: setup demo-m1 demo-m2 demo-m3 train-m1 train-m2 train-m3 app figures

setup:
	@echo "TODO: configurar el entorno"

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
