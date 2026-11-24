"""Demostración de clasificación de conducción en Streamlit."""
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import streamlit as st

from app.services.driver_service import alert_threshold, analyze_image

st.title("Conducción distractiva")
try:
    selected = alert_threshold()
except FileNotFoundError as error:
    st.info(str(error))
    st.stop()

image = st.file_uploader("Imagen del puesto del conductor", type=["jpg", "jpeg", "png", "webp"])
threshold = st.number_input("Umbral de alerta", min_value=0.0, max_value=1.0,
                            value=float(selected), step=0.01, format="%.3f")
if image is not None:
    try:
        result = analyze_image(image, threshold)
        image.seek(0)
        with Image.open(image) as source:
            preview = ImageOps.exif_transpose(source).convert("RGB")
        left, right = st.columns(2)
        left.image(preview, caption="Imagen analizada")
        cam = result["gradcam"]
        small = preview.resize((cam.shape[1], cam.shape[0]), Image.Resampling.BILINEAR)
        heat = np.broadcast_to(np.array([255, 75, 20]), (cam.shape[0], cam.shape[1], 3))
        opacity = 0.62 * cam[..., None]
        right.image((np.asarray(small) * (1 - opacity) + heat * opacity).astype(np.uint8),
                    caption="Grad-CAM de la clase predicha")
        st.metric("Riesgo estimado", f"{result['riesgo']:.1%}")
        if result["alerta"]:
            st.warning(f"Alerta · clase predicha: {result['clase']}")
        else:
            st.success(f"Sin alerta · clase predicha: {result['clase']}")
        for label, probability in result["probabilidades"].items():
            st.progress(probability, text=f"{label.capitalize()} · {probability:.1%}")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"No se pudo analizar la imagen: {error}")
