"""Modelo de priorización de alertas.

Unidad de análisis: un CASO = cuenta + semana con al menos una alerta de las
reglas (etapa 2). El modelo estima qué tan probable es que el caso tenga
actividad sospechosa, para que el equipo revise primero los más riesgosos.

Todas las variables usan información de esa semana o de semanas ANTERIORES,
nunca posteriores, y ninguna usa la etiqueta ni la tipología.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import COLUMNA_OBJETIVO
from src.reglas import EFECTIVO


def _clave(cuenta, semana) -> np.ndarray:
    return np.asarray(cuenta, dtype=np.int64) * 1000 + np.asarray(semana, dtype=np.int64)


def construir_casos(alertas: pd.DataFrame) -> pd.DataFrame:
    """Un caso por cuenta + semana, con qué reglas lo alertaron (1/0) y cuántas fueron."""
    a = alertas.copy()
    a["codigo"] = "regla_" + a["regla"].astype(str).str.split().str[0]
    reglas = (a.assign(v=1).pivot_table(index=["cuenta", "semana"], columns="codigo", values="v",
                                        aggfunc="max", fill_value=0))
    reglas.columns.name = None
    casos = reglas.astype("int8")
    casos["num_reglas"] = casos.sum(axis=1)
    casos["transacciones_alertadas"] = a.groupby(["cuenta", "semana"])["transacciones"].sum()
    return casos.reset_index()


def _agregado_semanal(df: pd.DataFrame, columna: str, contraparte: str, prefijo: str) -> pd.DataFrame:
    tabla = pd.DataFrame({
        "cuenta": df[columna].to_numpy(), "semana": df["semana"].to_numpy(),
        "monto": df["monto"].to_numpy(np.float64),
        "efectivo": df["tipo_pago"].isin(EFECTIVO).to_numpy(),
        "internacional": df["es_internacional"].to_numpy() == 1,
        "contraparte": df[contraparte].to_numpy(),
    })
    g = tabla.groupby(["cuenta", "semana"]).agg(tx=("monto", "size"), monto=("monto", "sum"), max=("monto", "max"),
                                                pct_efectivo=("efectivo", "mean"),
                                                pct_internacional=("internacional", "mean"))
    g["contrapartes"] = tabla[["cuenta", "semana", "contraparte"]].drop_duplicates().groupby(["cuenta", "semana"]).size()
    return g.add_prefix(prefijo)


def _historia(semanal: pd.DataFrame, columnas: list[str], prefijo: str) -> pd.DataFrame:
    """Promedio por semana activa de las semanas ANTERIORES (sin incluir la actual)."""
    s = semanal[columnas].sort_index()
    grupo = s.groupby(level="cuenta")
    previas = grupo.cumcount()
    acumulado = grupo.cumsum() - s
    promedio = acumulado.div(previas.replace(0, np.nan), axis=0)
    salida = promedio.add_prefix("hist_prom_")
    salida[f"{prefijo}semanas_previas"] = previas
    return salida


def construir_variables(df: pd.DataFrame, casos: pd.DataFrame) -> pd.DataFrame:
    """Agrega a cada caso su actividad de la semana y su historia previa.

    Para ahorrar memoria, solo se procesan las transacciones de cuentas con algún caso.
    """
    cuentas = np.unique(casos["cuenta"].to_numpy())
    en_casos = np.isin(df["cuenta_origen"].to_numpy(), cuentas) | np.isin(df["cuenta_destino"].to_numpy(), cuentas)
    sub = df.loc[en_casos, ["cuenta_origen", "cuenta_destino", "semana", "monto", "tipo_pago", "es_internacional"]]

    envia = _agregado_semanal(sub, "cuenta_origen", "cuenta_destino", "env_")
    recibe = _agregado_semanal(sub, "cuenta_destino", "cuenta_origen", "rec_")
    del sub
    envia = envia[envia.index.get_level_values("cuenta").isin(cuentas)]
    recibe = recibe[recibe.index.get_level_values("cuenta").isin(cuentas)]
    envia = envia.join(_historia(envia, ["env_tx", "env_monto"], "env_"))
    recibe = recibe.join(_historia(recibe, ["rec_tx", "rec_monto"], "rec_"))

    base = casos.set_index(["cuenta", "semana"])
    base = base.join(envia, how="left").join(recibe, how="left")

    # Cambio contra su propia historia (más de 1 = semana más intensa que su promedio)
    with np.errstate(divide="ignore", invalid="ignore"):
        base["env_ratio_monto"] = base["env_monto"] / base["hist_prom_env_monto"]
        base["env_ratio_tx"] = base["env_tx"] / base["hist_prom_env_tx"]
        base["rec_ratio_monto"] = base["rec_monto"] / base["hist_prom_rec_monto"]
    base["neto_semana"] = base["rec_monto"].fillna(0) - base["env_monto"].fillna(0)

    # Casos previos de la misma cuenta (semanas anteriores con alerta)
    base = base.sort_index()
    base["casos_previos"] = base.groupby(level="cuenta").cumcount()

    conteos = [c for c in base.columns if c.endswith(("_tx", "contrapartes", "semanas_previas"))]
    base[conteos] = base[conteos].fillna(0)
    return base.replace([np.inf, -np.inf], np.nan).reset_index()


def etiquetar_casos(df: pd.DataFrame, casos: pd.DataFrame) -> np.ndarray:
    """1 si la cuenta envió o recibió alguna transacción sospechosa en esa semana."""
    y = df[COLUMNA_OBJETIVO].to_numpy() == 1
    semana = df["semana"].to_numpy()[y]
    claves = np.union1d(_clave(df["cuenta_origen"].to_numpy()[y], semana),
                        _clave(df["cuenta_destino"].to_numpy()[y], semana))
    return np.isin(_clave(casos["cuenta"], casos["semana"]), claves).astype(np.int8)


# --- Métricas de priorización -------------------------------------------------
def metricas_priorizacion(y: np.ndarray, puntaje: np.ndarray, recall_objetivo: float = 0.90,
                          fracciones=(0.10, 0.20, 0.30)) -> dict:
    """PR-AUC, recall en el top-K y cuántas alertas se necesitan revisar para el recall objetivo."""
    y = np.asarray(y)
    orden = np.argsort(-np.asarray(puntaje), kind="stable")
    acumulado = np.cumsum(y[orden])
    total = y.sum()
    n = len(y)
    necesarias = int(np.searchsorted(acumulado, np.ceil(recall_objetivo * total)) + 1)
    salida = {
        "casos": int(n), "casos_sospechosos": int(total),
        "tasa_base_pct": round(100 * total / n, 2),
        "pr_auc": round(float(average_precision_score(y, puntaje)), 4),
    }
    for f in fracciones:
        k = int(round(f * n))
        salida[f"recall_top_{int(f * 100)}pct"] = round(100 * acumulado[k - 1] / total, 2)
        salida[f"precision_top_{int(f * 100)}pct"] = round(100 * acumulado[k - 1] / k, 2)
    salida[f"alertas_para_{int(recall_objetivo * 100)}pct_recall"] = necesarias
    salida[f"pct_alertas_para_{int(recall_objetivo * 100)}pct_recall"] = round(100 * necesarias / n, 2)
    salida["reduccion_de_alertas_pct"] = round(100 * (1 - necesarias / n), 2)
    return salida


def curva_ganancia(y: np.ndarray, puntaje: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """% de alertas revisadas (en orden del puntaje) vs. % de casos sospechosos capturados."""
    orden = np.argsort(-np.asarray(puntaje), kind="stable")
    acumulado = np.cumsum(np.asarray(y)[orden])
    x = np.arange(1, len(y) + 1) / len(y) * 100
    return np.r_[0, x], np.r_[0, acumulado / acumulado[-1] * 100]


def curva_pr(y: np.ndarray, puntaje: np.ndarray):
    precision, recall, _ = precision_recall_curve(y, puntaje)
    return recall * 100, precision * 100
