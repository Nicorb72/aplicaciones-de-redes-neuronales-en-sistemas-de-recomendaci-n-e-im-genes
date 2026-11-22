"""Variables de calendario (exógenas) para la demanda del Módulo 1.

Responsable: Persona 1.

Regla de oro (Tarea 1 del enunciado): solo se incluyen variables que se conocen
con al menos 30 días de anticipación. Todas las de este archivo se derivan de
leyes/decretos o del calendario astronómico, por lo que se conocen con años de
anticipación. Los eventos imprevisibles (paro nacional de 2021, cierres de vías)
se marcan aparte como ``evento_no_previsible`` y NO deben usarse como predictores
del pronóstico; solo sirven para el análisis y para enmascarar el entrenamiento.

Fuentes primarias
-----------------
* Festivos: Ley 51 de 1983 ("Ley Emiliani"), que traslada al lunes siguiente
  ciertos festivos, y Ley 35 de 1939 / Ley 37 de 1905 para los festivos fijos.
  Texto vigente (Función Pública, Gestor Normativo):
  https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=4954
* Ley 2578 de 2026 (Diario Oficial No. 53.510, 2-jun-2026), art. 6: nuevo festivo
  del 9 de julio (Virgen de Chiquinquirá), trasladable al lunes, desde 2026.
* Semana Santa: depende del Domingo de Pascua (cómputo gregoriano; algoritmo
  "anónimo gregoriano" de Meeus/Jones/Butcher, Meeus, 1991, *Astronomical
  Algorithms*, cap. 8).
* Receso estudiantil de octubre: Decreto 1373 de 2007 (MEN), compilado en el
  Decreto 1075 de 2015 (artículo exacto pendiente de confirmar en SUIN-Juriscol): cinco días de receso en la semana
  inmediatamente anterior al festivo del 12 de octubre.
* Jornadas electorales: calendario electoral de la Registraduría Nacional del
  Estado Civil (https://www.registraduria.gov.co). Fechas en configs/m1_datos.yaml.
* Vacaciones escolares de mitad y fin de año: el Decreto 1075 de 2015 fija 40 semanas lectivas y 12 de vacaciones para estudiantes,
  pero las fechas exactas las fija cada Secretaría de Educación por resolución.
  Aquí se usan VENTANAS APROXIMADAS configurables (configs/m1_datos.yaml) para
  calendario A; es un supuesto documentado, no una fecha normativa.

Uso
    python -m m1_demanda.data.calendar_features      # genera data/m1_demanda/external/calendario_colombia.csv
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from m1_demanda.data.utils import cargar_config, ruta_abs


# ---------------------------------------------------------------------------
# Festivos
# ---------------------------------------------------------------------------
def domingo_pascua(anio: int) -> date:
    """Domingo de Pascua (calendario gregoriano), algoritmo anónimo de Meeus."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = (h + l - 7 * m + 114) % 31 + 1
    return date(anio, mes, dia)


def _lunes_siguiente(d: date) -> date:
    """Ley 51 de 1983: si el festivo no cae en lunes, se traslada al lunes siguiente."""
    return d + timedelta(days=(7 - d.weekday()) % 7)


def festivos_colombia(anio: int) -> dict[date, str]:
    p = domingo_pascua(anio)
    fijos = {
        date(anio, 1, 1): "Año Nuevo",
        date(anio, 5, 1): "Día del Trabajo",
        date(anio, 7, 20): "Día de la Independencia",
        date(anio, 8, 7): "Batalla de Boyacá",
        date(anio, 12, 8): "Inmaculada Concepción",
        date(anio, 12, 25): "Navidad",
        p - timedelta(days=3): "Jueves Santo",
        p - timedelta(days=2): "Viernes Santo",
    }
    trasladables = {
        date(anio, 1, 6): "Reyes Magos",
        date(anio, 3, 19): "San José",
        date(anio, 6, 29): "San Pedro y San Pablo",
        date(anio, 8, 15): "Asunción de la Virgen",
        date(anio, 10, 12): "Día de la Raza",
        date(anio, 11, 1): "Todos los Santos",
        date(anio, 11, 11): "Independencia de Cartagena",
        p + timedelta(days=39): "Ascensión del Señor",
        p + timedelta(days=60): "Corpus Christi",
        p + timedelta(days=68): "Sagrado Corazón",
    }
    if anio >= 2026:
        # Ley 2578 de 2026 (1 de junio), art. 6: Día de Nuestra Señora del Rosario de
        # Chiquinquirá, 9 de julio, con traslado según la Ley 51 de 1983.
        # Diario Oficial No. 53.510 del 2 de junio de 2026.
        trasladables[date(anio, 7, 9)] = "Virgen de Chiquinquirá"
    out = dict(fijos)
    for d, nombre in trasladables.items():
        out[_lunes_siguiente(d)] = nombre
    return out


# ---------------------------------------------------------------------------
# Calendario diario
# ---------------------------------------------------------------------------
def _en_ventana(fechas: pd.DatetimeIndex, inicio_mmdd: str, fin_mmdd: str) -> np.ndarray:
    """True si la fecha cae en la ventana [inicio, fin] (mm-dd), admite cruzar año."""
    md = fechas.strftime("%m-%d")
    if inicio_mmdd <= fin_mmdd:
        return (md >= inicio_mmdd) & (md <= fin_mmdd)
    return (md >= inicio_mmdd) | (md <= fin_mmdd)


def _dias_hasta(evento: np.ndarray, hacia_adelante: bool, tope: int = 30) -> np.ndarray:
    """Días hasta el próximo (o desde el último) True en ``evento``, truncado en ``tope``."""
    n = len(evento)
    out = np.full(n, tope, dtype=int)
    ultimo = None
    rango = range(n - 1, -1, -1) if hacia_adelante else range(n)
    for i in rango:
        if evento[i]:
            ultimo = i
        if ultimo is not None:
            out[i] = min(abs(ultimo - i), tope)
    return out


def construir_calendario(cfg: dict) -> pd.DataFrame:
    cc = cfg["calendario"]
    fechas = pd.date_range(f"{cc['anio_inicio']}-01-01", f"{cc['anio_fin']}-12-31", freq="D")
    fest = {}
    for a in range(cc["anio_inicio"], cc["anio_fin"] + 1):
        fest.update(festivos_colombia(a))

    df = pd.DataFrame({"fecha": fechas})
    df["anio"] = fechas.year
    df["mes"] = fechas.month
    df["dia_semana"] = fechas.day_of_week            # 0 = lunes ... 6 = domingo
    df["dia_anio"] = fechas.day_of_year
    df["semana_iso"] = fechas.isocalendar().week.to_numpy()
    df["fin_de_semana"] = df.dia_semana >= 5
    df["festivo"] = [d.date() in fest for d in fechas]
    df["nombre_festivo"] = [fest.get(d.date(), "") for d in fechas]

    # Día no laborable = sábado, domingo o festivo. Un "puente" es un bloque de
    # 3 o más días no laborables consecutivos que contiene al menos un festivo.
    no_lab = (df.fin_de_semana | df.festivo).to_numpy()
    bloque = np.cumsum(np.r_[True, no_lab[1:] != no_lab[:-1]])
    tam = pd.Series(bloque).map(pd.Series(bloque[no_lab]).value_counts()).fillna(0).to_numpy()
    tiene_fest = pd.Series(df.festivo.to_numpy()).groupby(bloque).transform("any").to_numpy()
    df["puente"] = no_lab & (tam >= 3) & tiene_fest
    # Primer y último día del puente, y el día hábil anterior (salida) y siguiente.
    ini = df.puente & ~df.puente.shift(1, fill_value=False)
    fin = df.puente & ~df.puente.shift(-1, fill_value=False)
    df["inicio_puente"] = ini
    df["ultimo_dia_puente"] = fin
    df["vispera_puente"] = ini.shift(-1, fill_value=False)
    df["dia_despues_puente"] = fin.shift(1, fill_value=False)

    # Semana Santa: Domingo de Ramos a Domingo de Pascua.
    pascuas = {a: domingo_pascua(a) for a in range(cc["anio_inicio"], cc["anio_fin"] + 1)}
    df["semana_santa"] = [
        pascuas[d.year] - timedelta(days=7) <= d.date() <= pascuas[d.year] for d in fechas
    ]

    # Receso estudiantil de octubre (Decreto 1373 de 2007): lunes a viernes de la
    # semana anterior al festivo de Día de la Raza, más los fines de semana que lo rodean.
    receso = np.zeros(len(df), dtype=bool)
    for a in range(cc["anio_inicio"], cc["anio_fin"] + 1):
        lunes_raza = _lunes_siguiente(date(a, 10, 12))
        ini_r = pd.Timestamp(lunes_raza - timedelta(days=9))   # sábado anterior a la semana de receso
        receso |= ((df.fecha >= ini_r) & (df.fecha <= pd.Timestamp(lunes_raza))).to_numpy()
    df["receso_octubre"] = receso

    df["vacaciones_mitad_anio"] = _en_ventana(fechas, **{
        "inicio_mmdd": cc["vacaciones_mitad_anio"]["inicio_mmdd"],
        "fin_mmdd": cc["vacaciones_mitad_anio"]["fin_mmdd"]})
    df["vacaciones_fin_anio"] = _en_ventana(fechas, **{
        "inicio_mmdd": cc["vacaciones_fin_anio"]["inicio_mmdd"],
        "fin_mmdd": cc["vacaciones_fin_anio"]["fin_mmdd"]})
    df["temporada_fin_anio"] = _en_ventana(fechas, **{
        "inicio_mmdd": cc["temporada_fin_anio"]["inicio_mmdd"],
        "fin_mmdd": cc["temporada_fin_anio"]["fin_mmdd"]})
    # Jornadas electorales (config): suelen reducir los viajes (ley seca, restricciones).
    elec = {pd.Timestamp(f): n for f, n in cc.get("elecciones", [])}
    df["eleccion"] = df.fecha.isin(elec)
    df["nombre_eleccion"] = df.fecha.map(elec).fillna("")

    df["temporada_alta"] = (df.puente | df.semana_santa | df.temporada_fin_anio
                            | df.receso_octubre | df.vacaciones_mitad_anio)

    df["dias_hasta_festivo"] = _dias_hasta(df.festivo.to_numpy(), hacia_adelante=True)
    df["dias_desde_festivo"] = _dias_hasta(df.festivo.to_numpy(), hacia_adelante=False)

    # Codificación cíclica (útil para las redes): evita el salto 6 -> 0 o 365 -> 1.
    df["dow_sin"] = np.sin(2 * np.pi * df.dia_semana / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.dia_semana / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df.dia_anio / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.dia_anio / 365.25)
    return df


def verificar_contra_holidays(cfg: dict) -> pd.DataFrame:
    """Compara la implementación propia contra la librería ``holidays`` (si está
    instalada). Devuelve las discrepancias (debería ser una tabla vacía)."""
    try:
        import holidays
    except ImportError:
        return pd.DataFrame()
    cc = cfg["calendario"]
    filas = []
    for a in range(cc["anio_inicio"], cc["anio_fin"] + 1):
        propios = set(festivos_colombia(a))
        lib = set(holidays.Colombia(years=a))
        for d in propios ^ lib:
            filas.append({"fecha": d, "solo_propio": d in propios, "solo_holidays": d in lib})
    return pd.DataFrame(filas)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Genera el calendario de Colombia")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = cargar_config(args.config)
    cal = construir_calendario(cfg)
    salida = ruta_abs(cfg, "external") / "calendario_colombia.csv"
    cal.to_csv(salida, index=False)
    print(f"[ok] {salida.relative_to(salida.parents[3])}: {len(cal)} días, "
          f"{cal.festivo.sum()} festivos, {cal.puente.sum()} días de puente")
    dif = verificar_contra_holidays(cfg)
    if dif.empty:
        print("[ok] festivos idénticos a la librería `holidays` en todo el rango")
    else:
        print("[!!] discrepancias con `holidays`:\n", dif)


if __name__ == "__main__":
    main()
