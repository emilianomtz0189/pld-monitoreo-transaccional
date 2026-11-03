"""Configuración central del proyecto: rutas, semilla y parámetros globales.

Todas las rutas se construyen a partir de la raíz del repositorio, así el
código funciona igual en Colab, en una computadora local o en la de otra
persona, sin depender de rutas absolutas.
"""

import random
from pathlib import Path

import numpy as np

# --- Rutas -------------------------------------------------------------------
# Raíz del repositorio: la carpeta que contiene a src/
RAIZ = Path(__file__).resolve().parents[1]

DIR_DATOS = RAIZ / "data"
DIR_CRUDOS = DIR_DATOS / "raw"            # CSV tal como viene de Kaggle (no se sube)
DIR_PROCESADOS = DIR_DATOS / "processed"  # Parquet optimizado (no se sube)

ARCHIVO_CSV = DIR_CRUDOS / "SAML-D.csv"
ARCHIVO_PARQUET = DIR_PROCESADOS / "transacciones.parquet"
ARCHIVO_ALERTAS = DIR_PROCESADOS / "alertas.parquet"      # Salida de la etapa 2

DIR_RESULTADOS = RAIZ / "resultados"      # Sí se sube: gráficas y métricas del README
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_METRICAS = DIR_RESULTADOS / "metricas"

# --- Fuente de datos ---------------------------------------------------------
DATASET_KAGGLE = "berkanoztas/synthetic-transaction-monitoring-dataset-aml"

# --- Parámetros globales -----------------------------------------------------
SEMILLA = 42
COLUMNA_OBJETIVO = "es_sospechosa"   # 1 = transacción marcada como lavado en SAML-D

# IMPORTANTE: la tipología revela la etiqueta (cada tipología es 100 % normal
# o 100 % sospechosa). Se usa solo para analizar resultados, NUNCA como
# variable de una regla o de un modelo.
COLUMNA_TIPOLOGIA = "tipologia"


def fijar_semilla(semilla: int = SEMILLA) -> None:
    """Fija las semillas de los generadores aleatorios de Python y NumPy."""
    random.seed(semilla)
    np.random.seed(semilla)


def crear_carpetas() -> None:
    """Crea las carpetas de datos y resultados si no existen."""
    for carpeta in (DIR_CRUDOS, DIR_PROCESADOS, DIR_FIGURAS, DIR_METRICAS):
        carpeta.mkdir(parents=True, exist_ok=True)
