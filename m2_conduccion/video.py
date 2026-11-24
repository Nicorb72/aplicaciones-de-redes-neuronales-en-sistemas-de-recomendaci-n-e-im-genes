"""Prueba local del flujo temporal con un video grabado."""
import json
import math
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from m2_conduccion.predict import classify, load_checkpoint
from m2_conduccion.temporal import TemporalAlert


def analyze_video(path, config, threshold=None, sound=False):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Instala opencv-python-headless==4.11.0.86 para procesar video.") from error
    _, metadata = load_checkpoint()
    threshold = metadata["threshold"] if threshold is None else threshold
    if not math.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva y finita.")
    policy = TemporalAlert(threshold, config["window"], config["positives"], config["cooldown"])
    capture = cv2.VideoCapture(str(path))
    durations = []
    alerts = sampled = 0
    next_sample = 0.0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not math.isfinite(fps) or fps < config["fps"]:
            raise ValueError(f"Se requiere video decodificable de al menos {config['fps']} FPS: {path}")
        index = 0
        while True:
            start = perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            stamp = index / fps
            index += 1
            if stamp < next_sample - 1e-9:
                continue
            next_sample = (math.floor((stamp + 1e-9) * config["fps"]) + 1) / config["fps"]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            risk = classify(image, threshold, with_gradcam=False)["riesgo"]
            fired = policy.update(stamp, risk)
            durations.append(1000 * (perf_counter() - start))
            sampled += 1
            if fired:
                alerts += 1
                print(json.dumps({"alerta": True, "segundo": stamp, "riesgo": risk}), flush=True)
                if sound:
                    if sys.platform == "win32":
                        import winsound

                        winsound.MessageBeep()
                    else:
                        print("\a", end="", flush=True)
    finally:
        capture.release()
    if not sampled:
        raise ValueError("El archivo no contiene cuadros decodificables.")
    return {"cuadros_evaluados": sampled, "alertas": alerts, "umbral": threshold,
            "decision_archivo_p95_ms": float(np.quantile(durations, 0.95))}
