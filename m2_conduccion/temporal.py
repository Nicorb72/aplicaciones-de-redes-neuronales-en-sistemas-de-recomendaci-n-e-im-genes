"""Regla de persistencia para un flujo de cuadros muestreados."""
from collections import deque
import math


class TemporalAlert:
    def __init__(self, threshold, window=5, positives=3, cooldown=20.0):
        if not 0 <= threshold <= 1 or not 1 <= positives <= window:
            raise ValueError("El umbral y la ventana deben definir una regla válida.")
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError("El período de silencio debe ser finito y no negativo.")
        self.threshold = threshold
        self.positives = positives
        self.cooldown = cooldown
        self.history = deque(maxlen=window)
        self.last_sample = -math.inf
        self.last_alert = -math.inf

    def update(self, timestamp, risk, moving=True):
        if not math.isfinite(timestamp) or timestamp <= self.last_sample or not 0 <= risk <= 1:
            raise ValueError("Se requiere un tiempo creciente y un riesgo entre 0 y 1.")
        self.last_sample = timestamp
        if not moving:
            self.history.clear()
            return False
        self.history.append(risk >= self.threshold)
        if (len(self.history) == self.history.maxlen and sum(self.history) >= self.positives
                and timestamp - self.last_alert >= self.cooldown):
            self.last_alert = timestamp
            self.history.clear()
            return True
        return False
