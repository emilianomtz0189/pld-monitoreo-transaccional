"""Reglas de monitoreo transaccional y su evaluación.

Cada regla devuelve las transacciones que marca y la cuenta a la que se le
genera la alerta (la "cuenta foco"). Una ALERTA es una combinación única de
regla + cuenta + semana: así trabaja un analista, que revisa el caso de un
cliente en un periodo, no transacciones sueltas.

Las reglas se calibran SIN usar la etiqueta: con referencias regulatorias,
listas oficiales y percentiles del comportamiento. La etiqueta solo se usa
para evaluarlas.
"""

import numpy as np
import pandas as pd

from src.config import COLUMNA_OBJETIVO, COLUMNA_TIPOLOGIA

SEGUNDOS_DIA = 86_400
EFECTIVO = ["Cash Deposit", "Cash Withdrawal"]

# --- Supuestos ilustrativos (los datos están en libras y no son de México) ---
UMBRAL_USD_RELEVANTE = 7_500   # Referencia: umbral de "Operación Relevante" en efectivo (verificar en el DOF)
USD_POR_GBP = 1.24             # Supuesto de tipo de cambio aproximado para el periodo de los datos
UMBRAL_EFECTIVO_GBP = round(UMBRAL_USD_RELEVANTE / USD_POR_GBP)

# Países del dataset que estuvieron en las listas del GAFI durante el periodo
# de los datos (oct-2022 a ago-2023). Vigencia: (desde, hasta); None = abierto.
# Fechas de las plenarias del GAFI: 21-oct-2022 y 24-feb-2023 (verificar en fatf-gafi.org).
LISTA_GAFI = {
    "Albania": (None, None),
    "Turkey": (None, None),
    "UAE": (None, None),
    "Nigeria": ("2023-02-24", None),     # Agregado en la plenaria de febrero 2023
    "Morocco": (None, "2023-02-24"),     # Retirado en la plenaria de febrero 2023
    "Pakistan": (None, "2022-10-21"),    # Retirado en la plenaria de octubre 2022
}


# --- Preparación ---------------------------------------------------------------
def agregar_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega `segundo`, `dia` y `semana` contados desde el primer día de los datos."""
    inicio = df["fecha_hora"].min().normalize()
    transcurrido = (df["fecha_hora"].to_numpy() - inicio.to_datetime64()).astype("timedelta64[s]").astype(np.int64)
    df["segundo"] = transcurrido.astype(np.int32)
    df["dia"] = (transcurrido // SEGUNDOS_DIA).astype(np.int16)
    df["semana"] = (df["dia"] // 7).astype(np.int16)
    return df


def _marcas(idx: np.ndarray, cuenta: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"idx": idx.astype(np.int64), "cuenta": cuenta.astype(np.int64)}).drop_duplicates()


def _ventana_hacia_atras(cuenta: np.ndarray, segundo: np.ndarray, valor: np.ndarray, dias: int):
    """Para cada transacción: cuántas transacciones y qué monto acumula la misma
    cuenta en los `dias` previos (incluida la actual).

    Vectorizado: ordena por cuenta y tiempo y usa búsqueda binaria, sin ciclos.
    Devuelve el orden usado y, en ese orden: inicio de la ventana, conteo y suma.
    """
    orden = np.lexsort((segundo, cuenta))
    _, rango = np.unique(cuenta[orden], return_inverse=True)
    clave = rango.astype(np.int64) * 1_000_000_000 + segundo[orden].astype(np.int64)
    inicio = np.searchsorted(clave, clave - dias * SEGUNDOS_DIA, side="left")
    pos = np.arange(len(clave))
    acumulado = np.concatenate([[0.0], np.cumsum(valor[orden].astype(np.float64))])
    return orden, inicio, pos - inicio + 1, acumulado[pos + 1] - acumulado[inicio]


def _marcar_rangos(inicio: np.ndarray, fin: np.ndarray, n: int) -> np.ndarray:
    """Marca todas las posiciones dentro de los rangos [inicio, fin]."""
    cambios = np.bincount(inicio, minlength=n + 1) - np.bincount(fin + 1, minlength=n + 1)
    return np.cumsum(cambios)[:n] > 0


# --- R1. Efectivo de monto alto ------------------------------------------------
def regla_efectivo_alto(df: pd.DataFrame, umbral: float = UMBRAL_EFECTIVO_GBP) -> pd.DataFrame:
    """Depósito o retiro en efectivo igual o mayor al umbral (análogo a "operación relevante")."""
    marcado = df["tipo_pago"].isin(EFECTIVO).to_numpy() & (df["monto"].to_numpy() >= umbral)
    idx = np.flatnonzero(marcado)
    return _marcas(idx, df["cuenta_origen"].to_numpy()[idx])


# --- R2. Fraccionamiento -------------------------------------------------------
def regla_fraccionamiento(df: pd.DataFrame, umbral: float = UMBRAL_EFECTIVO_GBP, dias: int = 7,
                          minimo_operaciones: int = 3, piso_pct: float = 0.0) -> pd.DataFrame:
    """Varios depósitos en efectivo debajo del umbral que, juntos, lo rebasan en una ventana corta.

    - `piso_pct = 0.8`: solo cuentan depósitos "justo debajo" (80 % a 100 % del umbral).
    - `piso_pct = 0`: cuenta cualquier depósito debajo del umbral (versión acumulada).
    Se revisa desde quien deposita y desde quien recibe (varias personas a una cuenta).
    """
    monto = df["monto"].to_numpy()
    base = (df["tipo_pago"] == "Cash Deposit").to_numpy() & (monto < umbral) & (monto >= piso_pct * umbral)
    sub = np.flatnonzero(base)
    piezas = []
    for columna in ("cuenta_origen", "cuenta_destino"):
        cuentas = df[columna].to_numpy()
        orden, inicio, conteo, suma = _ventana_hacia_atras(
            cuentas[sub], df["segundo"].to_numpy()[sub], monto[sub], dias)
        dispara = np.flatnonzero((conteo >= minimo_operaciones) & (suma >= umbral))
        marcado = _marcar_rangos(inicio[dispara], dispara, len(orden))
        idx = sub[orden[marcado]]
        piezas.append(_marcas(idx, cuentas[idx]))
    return pd.concat(piezas, ignore_index=True).drop_duplicates()


# --- R3. Concentración (fan-in / fan-out) --------------------------------------
def regla_concentracion(df: pd.DataFrame, direccion: str, percentil: float = 99.9,
                        minimo: int = 5) -> tuple[pd.DataFrame, int, pd.Series]:
    """Cuentas con un número inusual de contrapartes distintas en una semana.

    - `entrada` (fan-in): una cuenta que RECIBE de muchas.
    - `salida` (fan-out): una cuenta que ENVÍA a muchas.
    El umbral es el percentil indicado de la distribución (sin usar la etiqueta).
    Devuelve las marcas, el umbral usado y la distribución de contrapartes.
    """
    foco, contraparte = (("cuenta_destino", "cuenta_origen") if direccion == "entrada"
                         else ("cuenta_origen", "cuenta_destino"))
    pares = df[[foco, "semana", contraparte]].drop_duplicates()
    contrapartes = pares.groupby([foco, "semana"]).size()
    del pares
    umbral = int(max(minimo, np.ceil(np.percentile(contrapartes.to_numpy(), percentil))))
    altas = contrapartes[contrapartes >= umbral].index
    clave_alta = altas.get_level_values(0).to_numpy(np.int64) * 1000 + altas.get_level_values(1).to_numpy(np.int64)
    cuentas = df[foco].to_numpy()
    clave_tx = cuentas.astype(np.int64) * 1000 + df["semana"].to_numpy(np.int64)
    idx = np.flatnonzero(np.isin(clave_tx, clave_alta))
    return _marcas(idx, cuentas[idx]), umbral, contrapartes


# --- R4. Movimiento rápido de fondos (pass-through) ----------------------------
def regla_movimiento_rapido(df: pd.DataFrame, monto_minimo: float = UMBRAL_EFECTIVO_GBP,
                            tolerancia: float = 0.10) -> pd.DataFrame:
    """Una cuenta recibe en un día y saca casi lo mismo (±tolerancia) ese día o el siguiente.

    Es la huella de una "cuenta puente": el dinero solo pasa por ahí.
    """
    monto = df["monto"].astype(np.float64)
    entradas = monto.groupby([df["cuenta_destino"].rename("cuenta"), df["dia"]]).sum()
    salidas = monto.groupby([df["cuenta_origen"].rename("cuenta"), df["dia"]]).sum()
    salidas_siguiente = salidas.copy()
    salidas_siguiente.index = pd.MultiIndex.from_arrays(
        [salidas.index.get_level_values(0), salidas.index.get_level_values(1) - 1], names=salidas.index.names)
    entradas = entradas[entradas >= monto_minimo]
    salida_2_dias = (salidas.reindex(entradas.index, fill_value=0.0)
                     + salidas_siguiente.groupby(level=[0, 1]).sum().reindex(entradas.index, fill_value=0.0))
    dispara = entradas.index[(salida_2_dias >= (1 - tolerancia) * entradas)
                             & (salida_2_dias <= (1 + tolerancia) * entradas)]
    clave = dispara.get_level_values(0).to_numpy(np.int64) * 1000 + dispara.get_level_values(1).to_numpy(np.int64)

    dia = df["dia"].to_numpy(np.int64)
    destino = df["cuenta_destino"].to_numpy().astype(np.int64)
    origen = df["cuenta_origen"].to_numpy().astype(np.int64)
    idx_entrada = np.flatnonzero(np.isin(destino * 1000 + dia, clave))
    idx_salida = np.flatnonzero(np.isin(origen * 1000 + dia, clave) | np.isin(origen * 1000 + dia - 1, clave))
    return pd.concat([_marcas(idx_entrada, destino[idx_entrada]),
                      _marcas(idx_salida, origen[idx_salida])], ignore_index=True).drop_duplicates()


# --- R5. Jurisdicciones de alto riesgo -----------------------------------------
def regla_alto_riesgo(df: pd.DataFrame, lista: dict = LISTA_GAFI) -> pd.DataFrame:
    """Operación internacional con un país que estaba en las listas del GAFI en la fecha de la operación.

    La alerta se genera a la cuenta del lado que NO está en el país de riesgo.
    """
    fecha = df["fecha_hora"]
    riesgo_destino = np.zeros(len(df), dtype=bool)
    riesgo_origen = np.zeros(len(df), dtype=bool)
    for pais, (desde, hasta) in lista.items():
        vigente = np.ones(len(df), dtype=bool)
        if desde:
            vigente &= (fecha >= pd.Timestamp(desde)).to_numpy()
        if hasta:
            vigente &= (fecha < pd.Timestamp(hasta)).to_numpy()
        riesgo_destino |= (df["pais_banco_destino"] == pais).to_numpy() & vigente
        riesgo_origen |= (df["pais_banco_origen"] == pais).to_numpy() & vigente
    marcado = (riesgo_destino | riesgo_origen) & (df["es_internacional"].to_numpy() == 1)
    idx = np.flatnonzero(marcado)
    foco = np.where(riesgo_destino[idx], df["cuenta_origen"].to_numpy()[idx], df["cuenta_destino"].to_numpy()[idx])
    return _marcas(idx, foco)


# --- R6. Cambio de comportamiento ----------------------------------------------
def regla_cambio_comportamiento(df: pd.DataFrame, factor_monto: float = 5, minimo_historial: int = 10,
                                monto_minimo: float = 1_000, dias: int = 7, factor_frecuencia: float = 3,
                                minimo_frecuencia: int = 5, dias_historial: int = 30) -> pd.DataFrame:
    """La cuenta que envía se aparta de su propio historial (solo usa operaciones ANTERIORES).

    - Monto: la operación es `factor_monto` veces mayor que su monto promedio previo
      (con al menos `minimo_historial` operaciones previas y monto >= `monto_minimo`).
    - Frecuencia: en los últimos `dias` hizo `factor_frecuencia` veces más operaciones
      que su promedio semanal previo (con al menos `dias_historial` días de historia).
    """
    cuentas = df["cuenta_origen"].to_numpy()
    segundo = df["segundo"].to_numpy()
    orden, inicio, conteo, _ = _ventana_hacia_atras(cuentas, segundo, df["monto"].to_numpy(), dias)
    c, s = cuentas[orden], segundo[orden].astype(np.int64)
    v = df["monto"].to_numpy()[orden].astype(np.float64)
    pos = np.arange(len(c))
    es_primera = np.r_[True, c[1:] != c[:-1]]
    inicio_cuenta = np.maximum.accumulate(np.where(es_primera, pos, 0))
    previas = pos - inicio_cuenta

    # Monto vs. promedio previo
    acumulado = np.cumsum(v)
    suma_previa = acumulado - v - (acumulado[inicio_cuenta] - v[inicio_cuenta])
    with np.errstate(divide="ignore", invalid="ignore"):
        media_previa = np.where(previas > 0, suma_previa / np.maximum(previas, 1), np.nan)
    salto_monto = (previas >= minimo_historial) & (v >= monto_minimo) & (v >= factor_monto * media_previa)

    # Frecuencia de la última ventana vs. ritmo semanal previo
    historia = s - s[inicio_cuenta]
    semanas_previas = (historia - dias * SEGUNDOS_DIA) / (7 * SEGUNDOS_DIA)
    previas_fuera = previas - (conteo - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ritmo_previo = np.where(semanas_previas > 0, previas_fuera / semanas_previas, np.nan)
    # Si no hay historia suficiente, el ritmo previo es NaN y la condición no se cumple
    salto_frecuencia = ((historia >= dias_historial * SEGUNDOS_DIA) & (conteo >= minimo_frecuencia)
                        & (conteo >= factor_frecuencia * np.nan_to_num(ritmo_previo, nan=np.inf)))

    dispara_f = np.flatnonzero(salto_frecuencia)
    marcado = salto_monto | _marcar_rangos(inicio[dispara_f], dispara_f, len(orden))
    idx = orden[marcado]
    return _marcas(idx, cuentas[idx])


# --- Evaluación ----------------------------------------------------------------
def evaluar_regla(df: pd.DataFrame, marcas: pd.DataFrame, nombre: str) -> tuple[dict, pd.DataFrame]:
    """Métricas de una regla y su tabla de alertas (cuenta + semana).

    - Precisión de alertas: % de alertas que contienen al menos una transacción sospechosa.
    - Precisión por transacción: % de las transacciones marcadas que son sospechosas.
    - Recall: % de TODAS las transacciones sospechosas que la regla marcó.
    """
    y = df[COLUMNA_OBJETIVO].to_numpy()
    idx = marcas["idx"].to_numpy()
    alertas = (pd.DataFrame({"cuenta": marcas["cuenta"].to_numpy(), "semana": df["semana"].to_numpy()[idx],
                             "sospechosa": y[idx]})
               .groupby(["cuenta", "semana"]).agg(transacciones=("sospechosa", "size"),
                                                  con_sospechosa=("sospechosa", "max"))
               .reset_index())
    unicos = np.unique(idx)
    detectadas = int(y[unicos].sum())
    dias = int(df["dia"].max()) + 1
    n_alertas = len(alertas)
    metricas = {
        "regla": nombre,
        "alertas": n_alertas,
        "alertas_por_dia": round(n_alertas / dias, 1),
        "cuentas_alertadas": int(alertas["cuenta"].nunique()),
        "alertas_con_sospechosa": int(alertas["con_sospechosa"].sum()),
        "precision_alertas_pct": round(100 * alertas["con_sospechosa"].mean(), 2) if n_alertas else 0.0,
        "transacciones_marcadas": int(len(unicos)),
        "sospechosas_detectadas": detectadas,
        "precision_transacciones_pct": round(100 * detectadas / len(unicos), 2) if len(unicos) else 0.0,
        "recall_pct": round(100 * detectadas / int(y.sum()), 2),
    }
    alertas.insert(0, "regla", nombre)
    return metricas, alertas


def recall_por_tipologia(df: pd.DataFrame, marcas_por_regla: dict) -> pd.DataFrame:
    """% de las transacciones de cada tipología sospechosa que detecta cada regla."""
    y = df[COLUMNA_OBJETIVO].to_numpy()
    tipologia = df[COLUMNA_TIPOLOGIA]
    total = tipologia[y == 1].value_counts()
    total = total[total > 0]
    tabla = {}
    for nombre, marcas in marcas_por_regla.items():
        idx = np.unique(marcas["idx"].to_numpy())
        idx = idx[y[idx] == 1]
        tabla[nombre] = 100 * tipologia.iloc[idx].value_counts().reindex(total.index, fill_value=0) / total
    resultado = pd.DataFrame(tabla)
    resultado.insert(0, "transacciones", total)
    resultado.index = resultado.index.astype(str)
    resultado.index.name = "tipologia"
    return resultado.sort_values("transacciones", ascending=False)
