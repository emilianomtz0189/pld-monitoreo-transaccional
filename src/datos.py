"""Descarga, carga optimizada en memoria, validación y respaldo del dataset SAML-D."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARCHIVO_CSV,
    ARCHIVO_PARQUET,
    DATASET_KAGGLE,
    DIR_RESULTADOS,
    RAIZ,
)

# Nombres originales de SAML-D -> nombres en español (snake_case).
# Date y Time se combinan en una sola columna `fecha_hora`.
RENOMBRAR = {
    "Sender_account": "cuenta_origen",
    "Receiver_account": "cuenta_destino",
    "Amount": "monto",
    "Payment_currency": "moneda_pago",
    "Received_currency": "moneda_recibida",
    "Sender_bank_location": "pais_banco_origen",
    "Receiver_bank_location": "pais_banco_destino",
    "Payment_type": "tipo_pago",
    "Is_laundering": "es_sospechosa",
    "Laundering_type": "tipologia",
}
COLUMNAS_ORIGINALES = ["Time", "Date", *RENOMBRAR]

# Tipos de dato al leer el CSV. Las columnas de texto con pocos valores
# distintos se guardan como `category` (un código entero + un diccionario).
TIPOS = {
    "Time": "category", "Date": "category",
    "Sender_account": "int64", "Receiver_account": "int64",
    "Amount": "float32",
    "Payment_currency": "category", "Received_currency": "category",
    "Sender_bank_location": "category", "Receiver_bank_location": "category",
    "Payment_type": "category", "Is_laundering": "int8", "Laundering_type": "category",
}


# --- Obtención del archivo ---------------------------------------------------
def _credenciales_colab() -> None:
    """Toma KAGGLE_USERNAME y KAGGLE_KEY de los Secretos de Colab, si existen."""
    try:
        from google.colab import userdata
        for clave in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            os.environ.setdefault(clave, userdata.get(clave))
    except Exception:
        pass


def obtener_ruta_csv(descargar: bool = True) -> Path | None:
    """Devuelve la ruta del CSV: primero busca en data/raw/, luego en la caché
    de kagglehub y, si `descargar=True`, lo descarga de Kaggle."""
    if ARCHIVO_CSV.exists():
        return ARCHIVO_CSV

    cache = Path.home() / ".cache" / "kagglehub" / "datasets" / DATASET_KAGGLE.split("/")[0]
    en_cache = sorted(cache.rglob("*.csv"), key=lambda p: p.stat().st_size) if cache.exists() else []
    if en_cache:
        return en_cache[-1]
    if not descargar:
        return None

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("Falta kagglehub. Instálalo con: pip install kagglehub")
    _credenciales_colab()
    carpeta = Path(kagglehub.dataset_download(DATASET_KAGGLE))
    csvs = sorted(carpeta.rglob("*.csv"), key=lambda p: p.stat().st_size)
    if not csvs:
        raise FileNotFoundError(f"La descarga no trajo ningún CSV en {carpeta}")
    print(f"Descargado de Kaggle: {csvs[-1].name} ({csvs[-1].stat().st_size / 1e6:,.0f} MB)")
    return csvs[-1]


def validar_columnas(ruta_csv: Path) -> None:
    """Lee 5 filas y confirma que estén las 12 columnas esperadas."""
    columnas = pd.read_csv(ruta_csv, nrows=5).columns
    faltantes = [c for c in COLUMNAS_ORIGINALES if c not in columnas]
    if faltantes:
        extra = [c for c in columnas if c not in COLUMNAS_ORIGINALES]
        raise ValueError(f"Faltan columnas esperadas: {faltantes}. Columnas extra: {extra}")


# --- Lectura optimizada ------------------------------------------------------
def leer_csv(ruta_csv: Path, por_partes: bool = False, tam_parte: int = 1_000_000) -> pd.DataFrame:
    """Lee el CSV con tipos optimizados.

    `por_partes=True` lee de un millón en un millón y une al final: es más
    lento, pero evita que Colab se quede sin memoria durante la lectura.
    """
    if not por_partes:
        return pd.read_csv(ruta_csv, dtype=TIPOS, usecols=COLUMNAS_ORIGINALES)

    from pandas.api.types import union_categoricals
    partes = []
    for i, parte in enumerate(pd.read_csv(ruta_csv, dtype=TIPOS, usecols=COLUMNAS_ORIGINALES,
                                          chunksize=tam_parte)):
        partes.append(parte)
        print(f"  parte {i + 1}: {len(parte):,} filas")
    # Cada parte trae sus propias categorías: se unifican antes de concatenar
    for col in [c for c, t in TIPOS.items() if t == "category"]:
        categorias = union_categoricals([p[col] for p in partes]).categories
        for p in partes:
            p[col] = p[col].cat.set_categories(categorias)
    return pd.concat(partes, ignore_index=True)


def construir_fecha_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Combina Date y Time en `fecha_hora` sin crear millones de textos.

    Convierte solo los valores únicos de fecha y hora (cientos o miles) y los
    reparte a cada fila con los códigos de la categoría.
    """
    fechas = pd.to_datetime(pd.Series(df["Date"].cat.categories.astype(str)))
    horas = pd.to_timedelta(pd.Series(df["Time"].cat.categories.astype(str)))
    if fechas.isna().any() or horas.isna().any():
        raise ValueError("Hay fechas u horas con formato no reconocido.")
    cod_fecha = df["Date"].cat.codes.to_numpy()
    cod_hora = df["Time"].cat.codes.to_numpy()
    if (cod_fecha < 0).any() or (cod_hora < 0).any():
        raise ValueError("Hay fechas u horas vacías.")
    df.insert(0, "fecha_hora", fechas.to_numpy()[cod_fecha] + horas.to_numpy()[cod_hora])
    return df.drop(columns=["Date", "Time"])


def reducir_cuentas(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa las cuentas a int32 solo si todos los números caben (máx. 2,147,483,647)."""
    for col in ("cuenta_origen", "cuenta_destino"):
        info = np.iinfo(np.int32)
        if df[col].min() >= info.min and df[col].max() <= info.max:
            df[col] = df[col].astype("int32")
    return df


def procesar_csv(ruta_csv: Path, por_partes: bool = False) -> pd.DataFrame:
    """CSV de Kaggle -> DataFrame limpio, ordenado en el tiempo y con nombres en español."""
    validar_columnas(ruta_csv)
    df = leer_csv(ruta_csv, por_partes=por_partes)
    df = construir_fecha_hora(df).rename(columns=RENOMBRAR)
    df = reducir_cuentas(df)
    return df.sort_values("fecha_hora", kind="stable").reset_index(drop=True)


def preparar_datos(carpeta_drive: Path | None = None, por_partes: bool = False) -> pd.DataFrame:
    """Devuelve las transacciones, usando el camino más rápido disponible.

    1. Parquet en data/processed/ (sesión actual).
    2. Parquet respaldado en Google Drive (sesiones anteriores).
    3. CSV de Kaggle: se procesa, se guarda en Parquet y se respalda en Drive.
    """
    if not ARCHIVO_PARQUET.exists() and carpeta_drive is not None:
        respaldo = Path(carpeta_drive) / ARCHIVO_PARQUET.name
        if respaldo.exists():
            ARCHIVO_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(respaldo, ARCHIVO_PARQUET)
            print(f"Copiado desde Drive: {respaldo}")

    if ARCHIVO_PARQUET.exists():
        return pd.read_parquet(ARCHIVO_PARQUET)

    df = procesar_csv(obtener_ruta_csv(), por_partes=por_partes)
    ARCHIVO_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARCHIVO_PARQUET, index=False)
    print(f"Guardado: {ARCHIVO_PARQUET.relative_to(RAIZ)} ({ARCHIVO_PARQUET.stat().st_size / 1e6:,.0f} MB)")
    if carpeta_drive is not None:
        Path(carpeta_drive).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ARCHIVO_PARQUET, Path(carpeta_drive) / ARCHIVO_PARQUET.name)
        print(f"Respaldo en Drive: {Path(carpeta_drive) / ARCHIVO_PARQUET.name}")
    return df


# --- Diagnóstico -------------------------------------------------------------
def resumen_estructura(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla por variable: tipo, memoria, nulos y valores únicos."""
    memoria = df.memory_usage(deep=True).drop("Index") / 1e6
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "memoria_mb": memoria.round(1),
        "nulos": df.isna().sum(),
        "valores_unicos": df.nunique(),
    })


def medir_ahorro_memoria(n_filas: int = 500_000) -> dict | None:
    """Mide la memoria de las primeras `n_filas` leídas sin tipos vs. con tipos optimizados.

    Es una medición real sobre una muestra, no una estimación del total.
    """
    ruta = obtener_ruta_csv(descargar=False)
    if ruta is None:
        return None
    bruto = float(pd.read_csv(ruta, nrows=n_filas).memory_usage(deep=True).sum())
    optimizado = float(pd.read_csv(ruta, nrows=n_filas, dtype=TIPOS).memory_usage(deep=True).sum())
    return {
        "filas_medidas": n_filas,
        "mb_sin_optimizar": round(bruto / 1e6, 1),
        "mb_optimizado": round(optimizado / 1e6, 1),
        "reduccion_pct": round(100 * (1 - optimizado / bruto), 1),
    }


def respaldar_resultados(carpeta_drive: Path) -> None:
    """Copia resultados/ (gráficas y métricas) a Google Drive para no perderlos
    cuando se cierre la sesión de Colab."""
    destino = Path(carpeta_drive) / DIR_RESULTADOS.name
    shutil.copytree(DIR_RESULTADOS, destino, dirs_exist_ok=True)
    print(f"Resultados respaldados en {destino}")
