# Módulo 2 · Clasificación de conducción distractiva

El [informe técnico](../docs/m2_informe_tecnico.md) reúne el protocolo, resultados, figuras y política de alertas. Las tablas quedan en [reports/tables](../reports/tables) y las figuras en [reports/figures/m2](../reports/figures/m2).

## Ejecución desde la raíz del repositorio

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-m2.txt
python -m m2_conduccion.run --mode data --download
python -m m2_conduccion.run --mode train --reuse-data --device auto
python -m m2_conduccion.run --mode demo --image ruta/a/imagen.jpg
python -m streamlit run app/app.py
```

El comando de datos descarga 2,56 GB de Mendeley, valida SHA-256 y genera el manifiesto agrupado. Si ya tienes el ZIP, usa `--mode data --archive ruta/al/archivo.zip`. El primer entrenamiento también necesita los pesos ImageNet de ResNet-18. En un entorno desconectado se pueden guardar previamente como `models/m2_conduccion/resnet18-f37072fd.pth`. La demostración usa el checkpoint local `models/m2_conduccion/best.pt` y expone la página **Conducción** de Streamlit.

La búsqueda admite `--epochs`, `--search-trials` y `--threads`. La interfaz `m2_conduccion.predict.classify(image, threshold=None)` recibe una ruta, archivo binario o imagen PIL. Devuelve clase, probabilidades calibradas, riesgo, umbral, alerta y mapa Grad-CAM. En inferencia por lotes de video se puede desactivar Grad-CAM con `with_gradcam=False`.

Los datos preparados se distribuyen en `data/m2_conduccion/processed`. La carpeta contiene el manifiesto de particiones, las coincidencias visuales y la caché de imágenes de 112 × 112 px. Tras clonar el repositorio, ejecuta `git lfs pull` para descargar la caché y usa `--mode train --reuse-data` para entrenar con la partición publicada. Para regenerar los datos, el comando de descarga obtiene el ZIP original (2,56 GB) de [Mendeley Data](https://data.mendeley.com/datasets/mzb4b6dff3/1) y comprueba su SHA-256.
