"""Segmentación de cuentas por riesgo (enfoque basado en riesgo).

Cada cuenta recibe un puntaje de 0 a 100 con factores de riesgo observables
(sin usar la etiqueta) y se clasifica en nivel bajo, medio o alto.
"""

import numpy as np
import pandas as pd

from src.config import COLUMNA_OBJETIVO
from src.reglas import EFECTIVO, regla_alto_riesgo

# Pesos por criterio experto, alineados a los grupos de factores de riesgo de
# una evaluación de riesgos de PLD: canales/productos, geografía, clientes y
# transacciones. No se ajustan con la etiqueta.
PESOS = {
    "pct_efectivo": 0.25,        # Canal: efectivo
    "pct_internacional": 0.15,   # Geografía: operaciones con otros países
    "pct_alto_riesgo": 0.15,     # Geografía: países en listas del GAFI
    "contrapartes": 0.15,        # Transacciones: con cuántas cuentas distintas opera
    "monto_total": 0.10,         # Transacciones: volumen operado
    "semanas_alertadas": 0.20,   # Historial de alertas del monitoreo (etapa 2)
}

# Cortes de nivel por percentil del puntaje
CORTE_ALTO = 0.95    # 5 % de cuentas con mayor puntaje
CORTE_MEDIO = 0.80   # siguiente 15 %


def perfil_cuentas(df: pd.DataFrame, alertas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por cuenta con sus factores de riesgo (como emisora y como receptora)."""
    monto = df["monto"].to_numpy(np.float64)
    efectivo = df["tipo_pago"].isin(EFECTIVO).to_numpy()
    internacional = df["es_internacional"].to_numpy() == 1
    alto_riesgo = np.zeros(len(df), dtype=bool)
    alto_riesgo[regla_alto_riesgo(df)["idx"].to_numpy()] = True

    partes = []
    for columna in ("cuenta_origen", "cuenta_destino"):
        partes.append(pd.DataFrame({
            "cuenta": df[columna].to_numpy(), "monto": monto,
            "efectivo": efectivo, "internacional": internacional, "alto_riesgo": alto_riesgo,
        }).groupby("cuenta").agg(transacciones=("monto", "size"), monto_total=("monto", "sum"),
                                 efectivo=("efectivo", "sum"), internacional=("internacional", "sum"),
                                 alto_riesgo=("alto_riesgo", "sum")))
    perfil = partes[0].add(partes[1], fill_value=0)
    del partes

    # Contrapartes distintas, sumando ambos roles
    pares = pd.concat([
        pd.DataFrame({"cuenta": df["cuenta_origen"].to_numpy(), "contraparte": df["cuenta_destino"].to_numpy()}),
        pd.DataFrame({"cuenta": df["cuenta_destino"].to_numpy(), "contraparte": df["cuenta_origen"].to_numpy()}),
    ], ignore_index=True).drop_duplicates()
    perfil["contrapartes"] = pares.groupby("cuenta").size()
    del pares

    perfil["pct_efectivo"] = perfil["efectivo"] / perfil["transacciones"]
    perfil["pct_internacional"] = perfil["internacional"] / perfil["transacciones"]
    perfil["pct_alto_riesgo"] = perfil["alto_riesgo"] / perfil["transacciones"]

    casos = alertas.drop_duplicates(["cuenta", "semana"]).groupby("cuenta").size()
    reglas_distintas = alertas.groupby("cuenta")["regla"].nunique()
    perfil["semanas_alertadas"] = casos.reindex(perfil.index, fill_value=0)
    perfil["reglas_distintas"] = reglas_distintas.reindex(perfil.index, fill_value=0)
    perfil.index.name = "cuenta"
    return perfil.drop(columns=["efectivo", "internacional", "alto_riesgo"])


def puntaje_riesgo(perfil: pd.DataFrame, pesos: dict = PESOS) -> pd.DataFrame:
    """Puntaje de 0 a 100 y nivel de riesgo.

    Cada factor se convierte a su percentil entre todas las cuentas (0 a 1). Una
    cuenta con valor 0 en un factor recibe 0 en ese factor. El puntaje es la
    suma ponderada de percentiles por 100.
    """
    resultado = perfil.copy()
    puntaje = np.zeros(len(perfil))
    for factor, peso in pesos.items():
        valor = perfil[factor]
        percentil = valor.rank(pct=True).where(valor > 0, 0.0)
        resultado[f"pts_{factor}"] = 100 * peso * percentil
        puntaje += resultado[f"pts_{factor}"].to_numpy()
    resultado["puntaje"] = puntaje
    corte_alto, corte_medio = np.quantile(puntaje, [CORTE_ALTO, CORTE_MEDIO])
    resultado["nivel"] = pd.Categorical(
        np.where(puntaje >= corte_alto, "alto", np.where(puntaje >= corte_medio, "medio", "bajo")),
        categories=["bajo", "medio", "alto"], ordered=True)
    resultado.attrs["cortes"] = {"medio": float(corte_medio), "alto": float(corte_alto)}
    return resultado


def cuentas_sospechosas(df: pd.DataFrame) -> np.ndarray:
    """Cuentas que participan (enviando o recibiendo) en alguna transacción sospechosa."""
    y = df[COLUMNA_OBJETIVO].to_numpy() == 1
    return np.union1d(df["cuenta_origen"].to_numpy()[y], df["cuenta_destino"].to_numpy()[y])


def evaluar_niveles(segmentos: pd.DataFrame, sospechosas: np.ndarray) -> pd.DataFrame:
    """Por nivel: cuentas, cuántas tienen actividad sospechosa y qué % del total capturan."""
    es_sosp = segmentos.index.isin(sospechosas)
    tabla = pd.DataFrame({"nivel": segmentos["nivel"], "sospechosa": es_sosp}).groupby("nivel", observed=False).agg(
        cuentas=("sospechosa", "size"), cuentas_sospechosas=("sospechosa", "sum"))
    tabla["pct_cuentas"] = 100 * tabla["cuentas"] / tabla["cuentas"].sum()
    tabla["tasa_sospechosa_pct"] = 100 * tabla["cuentas_sospechosas"] / tabla["cuentas"]
    tabla["pct_de_las_sospechosas"] = 100 * tabla["cuentas_sospechosas"] / tabla["cuentas_sospechosas"].sum()
    tabla["veces_el_promedio"] = tabla["tasa_sospechosa_pct"] / (100 * es_sosp.mean())
    return tabla.iloc[::-1]
