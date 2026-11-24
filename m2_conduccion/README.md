# Módulo 2 · Clasificación de conducción distractiva

El [informe técnico](../docs/m2_informe_tecnico.md) reúne el protocolo, resultados, figuras y política de alertas. Las tablas quedan en [reports/tables](../reports/tables) y las figuras en [reports/figures/m2](../reports/figures/m2).

## Ejecución desde la raíz del repositorio

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-m2.txt
git lfs pull
python -m m2_conduccion.run --mode demo
python -m streamlit run app/pages/2_Conduccion.py
```

La demostración usa `models/m2_conduccion/best.pt` y clasifica una imagen de prueba. Con `--image ruta/a/imagen.jpg` se puede clasificar una imagen propia. El comando de Streamlit abre la página de M2 con el umbral ajustable y Grad-CAM.

`python -m m2_conduccion.run --mode figures --reuse-data` regenera las figuras a partir de los resultados guardados y el checkpoint.

La búsqueda admite `--epochs`, `--search-trials` y `--threads`. La interfaz `m2_conduccion.predict.classify(image, threshold=None)` recibe una ruta, archivo binario o imagen PIL. Devuelve clase, probabilidades calibradas, riesgo, umbral, alerta y mapa Grad-CAM. En inferencia por lotes de video se puede desactivar Grad-CAM con `with_gradcam=False`.

Los datos preparados se distribuyen en `data/m2_conduccion/processed`. La carpeta contiene el manifiesto de particiones, las coincidencias visuales y la caché de imágenes de 112 × 112 px. Tras clonar el repositorio, ejecuta `git lfs pull` para descargar la caché y usa `--mode train --reuse-data` para entrenar con la partición publicada. Para regenerar los datos, el comando de descarga obtiene el ZIP original (2,56 GB) de [Mendeley Data](https://data.mendeley.com/datasets/mzb4b6dff3/1) y comprueba su SHA-256.

## Reproducción del experimento

```powershell
python -m m2_conduccion.run --mode train --reuse-data --device auto
python -m m2_conduccion.run --mode stability --reuse-data --device auto
python -m m2_conduccion.run --mode audit-crops --reuse-data
python -m m2_conduccion.run --mode alert-review --reuse-data
```

La estabilidad usa las configuraciones seleccionadas y ejecuta 30 semillas por red. Conserva cada corrida terminada para poder reanudarla con la misma configuración. El modo `search-timing` mide el presupuesto de búsqueda y conserva la selección publicada. El primer entrenamiento necesita los pesos ImageNet de ResNet-18. Se pueden ubicar previamente como `models/m2_conduccion/resnet18-f37072fd.pth` para entrenar en un equipo desconectado. Para preparar los datos desde cero, usa `--mode data --download` o `--mode data --archive ruta/al/archivo.zip`.

## Prueba temporal con video

```powershell
python -m pip install opencv-python-headless==4.11.0.86
python -m m2_conduccion.run --mode video --video ruta/a/video.mp4
```

El modo de video mide el tiempo desde la lectura del cuadro de un archivo local hasta la decisión de alertar. Las alertas se registran con el segundo del video. La opción `--sound` habilita el aviso durante la prueba.
