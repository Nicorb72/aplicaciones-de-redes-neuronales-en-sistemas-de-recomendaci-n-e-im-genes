"""Interfaz de inferencia del Módulo 3 para la aplicación del equipo.

No entrena modelos en la inferencia. Prepara las estadísticas cold-start UNA vez:
    python -m m3_recomendacion.predict --preparar-cold-start

Demostración:
    python -m m3_recomendacion.predict --usuario 1
    python -m m3_recomendacion.predict --preferencias "Nature, Adventure"

Desde Python:
    from m3_recomendacion.predict import recomendar, obtener_preferencias
    recomendar(user_id=1, k=3)
    recomendar(preferencias="Nature, Adventure", k=3)

El identificador de usuario es el UserID del dataset. Los resultados no son
probabilidades calibradas de compra ni estimaciones de conversión.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from m3_recomendacion.models.model_io import cargar_modelo

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "m3_recomendacion" / "processed"
CHECKPOINT_NEURAL = ROOT / "artifacts" / "m3" / "neural_c5.pt"
CHECKPOINT_COLD = ROOT / "artifacts" / "m3" / "cold_start.json"


def preparar_cold_start(
    train_path: Path = PROCESSED / "train.csv",
    model_path: Path = CHECKPOINT_NEURAL,
    output_path: Path = CHECKPOINT_COLD,
) -> Path:
    """Exporta popularidad por perfil aprendida SOLO con entrenamiento.

    Esta operación se hace una vez, antes del despliegue. No lee validación ni test.
    """
    sistema = cargar_modelo(model_path)
    train = pd.read_csv(train_path)
    requeridas = {"UserID", "ItemID", "Positiva"}
    if not requeridas.issubset(train.columns):
        raise ValueError(f"Faltan columnas en train.csv: {sorted(requeridas - set(train.columns))}")

    train_users = set(train["UserID"].astype(int))
    if train_users != set(sistema.user_map):
        raise ValueError("Los usuarios de train.csv no coinciden con el modelo guardado.")

    catalogo = sorted(int(i) for i in sistema.item_ids)
    if not set(train["ItemID"].astype(int)).issubset(catalogo):
        raise ValueError("Hay destinos en entrenamiento que no aparecen en el modelo.")

    positivos = train.loc[train["Positiva"].astype(int).eq(1)]
    global_counts = {str(item): 0 for item in catalogo}
    profile_counts = {
        pref: {str(item): 0 for item in catalogo}
        for pref in sistema.pref_map
    }
    for user_id, item_id in positivos[["UserID", "ItemID"]].itertuples(index=False, name=None):
        user_id, item_id = int(user_id), int(item_id)
        pref = sistema.user_preferences.get(user_id)
        if pref not in profile_counts:
            raise ValueError(f"No se encuentra la preferencia del usuario {user_id}.")
        global_counts[str(item_id)] += 1
        profile_counts[pref][str(item_id)] += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "source": "train.csv: interacciones positivas observadas",
        "catalogo_item_ids": catalogo,
        "global_counts": global_counts,
        "profile_counts": profile_counts,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


class ServicioRecomendacion:
    """Servicio liviano para inferencia; carga pesos y conteos ya exportados."""

    def __init__(
        self,
        neural_path: Path = CHECKPOINT_NEURAL,
        cold_path: Path = CHECKPOINT_COLD,
        catalogo_path: Path = PROCESSED / "catalogo_destinos.csv",
    ) -> None:
        self.neural = cargar_modelo(neural_path)
        cold_path = Path(cold_path)
        if not cold_path.is_file():
            raise FileNotFoundError(
                f"Falta {cold_path}. Ejecute primero: "
                "python -m m3_recomendacion.predict --preparar-cold-start"
            )
        cold = json.loads(cold_path.read_text(encoding="utf-8"))
        if cold.get("format_version") != 1:
            raise ValueError("Versión de estadísticas cold-start incompatible.")

        tabla = pd.read_csv(catalogo_path)
        columnas = {"ItemID", "Name", "Type"}
        if not columnas.issubset(tabla.columns):
            raise ValueError(f"Faltan columnas en catálogo: {sorted(columnas - set(tabla.columns))}")
        if tabla["ItemID"].duplicated().any():
            raise ValueError("El catálogo contiene ItemID duplicados.")

        self.catalogo = {
            int(fila["ItemID"]): {c: self._limpiar(fila[c]) for c in tabla.columns if c != "ItemID"}
            for _, fila in tabla.iterrows()
        }
        ids = set(self.catalogo)
        if ids != set(self.neural.item_ids) or ids != set(cold["catalogo_item_ids"]):
            raise ValueError("El catálogo no coincide con los modelos exportados.")

        self.preferencias = tuple(sorted(self.neural.pref_map))
        self.global_counts = {int(k): int(v) for k, v in cold["global_counts"].items()}
        self.profile_counts = {
            perfil: {int(k): int(v) for k, v in conteos.items()}
            for perfil, conteos in cold["profile_counts"].items()
        }
        if set(self.profile_counts) != set(self.preferencias):
            raise ValueError("Las preferencias de cold-start y del modelo difieren.")
        if ids != set(self.global_counts) or any(ids != set(v) for v in self.profile_counts.values()):
            raise ValueError("Los conteos cold-start no coinciden con el catálogo.")

    @staticmethod
    def _limpiar(valor: Any) -> Any:
        """Convierte valores de pandas a tipos sencillos para la interfaz."""
        if pd.isna(valor):
            return None
        if hasattr(valor, "item"):
            return valor.item()
        return valor

    def _preferencia_valida(self, preferencias: str) -> str:
        if not isinstance(preferencias, str) or not preferencias.strip():
            raise ValueError("Seleccione una preferencia válida.")
        buscada = " ".join(preferencias.split()).casefold()
        opciones = {
            " ".join(p.split()).casefold(): p for p in self.preferencias
        }
        if buscada not in opciones:
            raise ValueError(
                "Preferencia desconocida. Opciones: " + "; ".join(self.preferencias)
            )
        return opciones[buscada]

    def _destino(self, item_id: int) -> dict[str, Any]:
        return {"item_id": int(item_id), **self.catalogo[int(item_id)]}

    def recomendar(
        self,
        user_id: int | None = None,
        preferencias: str | None = None,
        k: int = 3,
        estrategia: str = "preferences",
    ) -> dict[str, Any]:
        """Devuelve información serializable para la interfaz web.

        Usuario del entrenamiento: C5, se excluyen TODAS las interacciones
        conocidas del train, positivas y negativas.
        Usuario nuevo: popularidad global o por perfil, sin historial registrado.
        """
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(self.catalogo):
            raise ValueError(f"k debe ser un entero entre 1 y {len(self.catalogo)}.")
        if estrategia not in {"global", "preferences"}:
            raise ValueError("Estrategia inválida: use 'global' o 'preferences'.")

        conocido = False
        if user_id is not None:
            if isinstance(user_id, bool):
                raise ValueError("user_id debe ser un entero.")
            try:
                user_id = int(user_id)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("user_id debe ser un entero.") from exc
            conocido = user_id in self.neural.user_map

        if conocido:
            perfil = self.neural.user_preferences[user_id]
            if preferencias is not None and self._preferencia_valida(preferencias) != perfil:
                raise ValueError(
                    "El perfil de un usuario existente está fijado en el modelo; "
                    "para elegir otro perfil, cree un usuario nuevo."
                )
            ranking = self.neural.recommend(user_id=user_id, k=k)
            vistos = sorted(int(i) for i in self.neural.historial.get(user_id, set()))
            recomendaciones = [
                {**self._destino(item_id), "posicion": pos, "puntaje_modelo": float(score)}
                for pos, (item_id, score) in enumerate(ranking, start=1)
            ]
            origen = "neuronal_c5"
            tipo_usuario = "existente"
        else:
            if preferencias is None:
                raise ValueError(
                    "Para un usuario nuevo debe indicar 'preferencias'. "
                    "Opciones: " + "; ".join(self.preferencias)
                )
            perfil = self._preferencia_valida(preferencias)
            ranking = sorted(
                self.catalogo,
                key=lambda item: (
                    -(
                        self.profile_counts[perfil][item] if estrategia == "preferences" else self.global_counts[item]
                    ),
                    -(self.global_counts[item] if estrategia == "preferences" else 0),
                    item,
                ),
            )[:k]
            recomendaciones = [
                {**self._destino(item_id), "posicion": pos, "puntaje_modelo": None}
                for pos, item_id in enumerate(ranking, start=1)
            ]
            vistos = []
            tipo_usuario = "nuevo"
            origen = f"cold_start_{estrategia}"

        return {
            "tipo_usuario": tipo_usuario,
            "user_id": user_id,
            "preferencias": perfil,
            "estrategia": origen,
            "k_solicitado": k,
            "historial": [self._destino(i) for i in vistos],
            "recomendaciones": recomendaciones,
            "nota_puntaje": (
                "Los puntajes neuronales no son probabilidades calibradas de compra."
                if conocido else "Orden basado en frecuencias de entrenamiento; no se asigna probabilidad."
            ),
        }


@lru_cache(maxsize=1)
def obtener_servicio() -> ServicioRecomendacion:
    """Carga los archivos UNA sola vez por proceso de aplicación."""
    return ServicioRecomendacion()


def obtener_preferencias() -> list[str]:
    """Valores válidos para el selector de preferencias de la aplicación."""
    return list(obtener_servicio().preferencias)


def recomendar(
    user_id: int | None = None,
    preferencias: str | None = None,
    k: int = 3,
    estrategia: str = "preferences",
) -> dict[str, Any]:
    """Punto de entrada recomendado para Persona 5."""
    return obtener_servicio().recomendar(
        user_id=user_id, preferencias=preferencias, k=k, estrategia=estrategia
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia del recomendador M3 sin reentrenamiento")
    parser.add_argument("--preparar-cold-start", action="store_true")
    parser.add_argument("--usuario", type=int, default=None)
    parser.add_argument("--preferencias", type=str, default=None)
    parser.add_argument("--estrategia", choices=("global", "preferences"), default="preferences")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    if args.preparar_cold_start:
        print("Estadísticas cold-start guardadas en:", preparar_cold_start())
        return
    try:
        resultado = recomendar(
            user_id=args.usuario,
            preferencias=args.preferencias,
            k=args.k,
            estrategia=args.estrategia,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
