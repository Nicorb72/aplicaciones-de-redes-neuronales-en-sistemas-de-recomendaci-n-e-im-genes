"""Interfaz local entre la página de conducción y el clasificador."""
from m2_conduccion.predict import classify, load_checkpoint


def alert_threshold():
    return load_checkpoint()[1]["threshold"]


def analyze_image(image, threshold):
    return classify(image, threshold)
