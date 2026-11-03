"""Análisis exploratorio orientado a prevención de lavado de dinero (PLD)."""

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from src.config import COLUMNA_OBJETIVO, COLUMNA_TIPOLOGIA, DIR_FIGURAS, DIR_METRICAS, RAIZ

# Código de color en todo el proyecto: azul = normal/total, naranja = sospechosa
COLOR_NORMAL = "#2a78d6"
COLOR_SOSPECHOSA = "#eb6834"
COLOR_REFERENCIA = "#52514e"


# --- Utilidades --------------------------------------------------------------
def _estilo(ax, titulo: str) -> None:
    ax.set_title(titulo, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def guardar_figura(fig, nombre: str) -> None:
    """Guarda la gráfica en resultados/figuras/ para usarla en el README."""
    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_FIGURAS / nombre
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print("Gráfica guardada:", ruta.relative_to(RAIZ))


def guardar_metricas(metricas: dict, nombre: str) -> None:
    """Guarda las métricas en JSON: es la única fuente de cifras del README."""
    DIR_METRICAS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_METRICAS / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2, default=str)
    print("Métricas guardadas:", ruta.relative_to(RAIZ))


def mismo_valor(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Compara dos columnas categóricas por su código común, sin convertirlas a texto."""
    categorias = a.cat.categories.union(b.cat.categories)
    codigos_a = a.cat.set_categories(categorias).cat.codes.to_numpy()
    codigos_b = b.cat.set_categories(categorias).cat.codes.to_numpy()
    return codigos_a == codigos_b


def agregar_banderas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega dos banderas de riesgo geográfico:
    - `es_internacional`: banco de origen y destino en distinto país.
    - `cambio_moneda`: moneda pagada distinta de la recibida.
    """
    df["es_internacional"] = (~mismo_valor(df["pais_banco_origen"], df["pais_banco_destino"])).astype("int8")
    df["cambio_moneda"] = (~mismo_valor(df["moneda_pago"], df["moneda_recibida"])).astype("int8")
    return df


# --- Volumen y desbalance ----------------------------------------------------
def resumen_volumen(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Cifras de volumen y serie de transacciones por día."""
    cuentas = np.union1d(df["cuenta_origen"].unique(), df["cuenta_destino"].unique())
    por_dia = df["fecha_hora"].dt.normalize().value_counts().sort_index()
    resumen = {
        "transacciones": int(len(df)),
        "cuentas_unicas": int(len(cuentas)),
        "cuentas_que_envian": int(df["cuenta_origen"].nunique()),
        "cuentas_que_reciben": int(df["cuenta_destino"].nunique()),
        "fecha_inicial": str(df["fecha_hora"].min()),
        "fecha_final": str(df["fecha_hora"].max()),
        "dias_con_operacion": int(por_dia.size),
        "transacciones_por_dia_promedio": round(float(por_dia.mean()), 1),
        "transacciones_por_dia_minimo": int(por_dia.min()),
        "transacciones_por_dia_maximo": int(por_dia.max()),
        "monto_total": round(float(df["monto"].astype("float64").sum()), 2),
    }
    return resumen, por_dia


def graficar_por_dia(por_dia: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(por_dia.index, por_dia.values, color=COLOR_NORMAL, linewidth=2)
    ax.set_ylabel("Transacciones")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    _estilo(ax, "Transacciones por día")
    fig.tight_layout()
    return fig


def resumen_desbalance(df: pd.DataFrame) -> dict:
    """Qué tan raras son las transacciones sospechosas y cuánto monto representan."""
    y = df[COLUMNA_OBJETIVO]
    n_sosp = int(y.sum())
    tasa = n_sosp / len(df)
    monto = df["monto"].astype("float64")
    monto_sosp = float(monto[y == 1].sum())
    return {
        "transacciones_normales": int(len(df) - n_sosp),
        "transacciones_sospechosas": n_sosp,
        "tasa_sospechosa_pct": round(100 * tasa, 4),
        "una_de_cada": round(1 / tasa, 1),
        "exactitud_si_nada_es_sospechoso_pct": round(100 * (1 - tasa), 4),
        "monto_sospechoso": round(monto_sosp, 2),
        "monto_sospechoso_pct_del_total": round(100 * monto_sosp / float(monto.sum()), 4),
    }


# --- Perfil de riesgo por grupo ----------------------------------------------
def perfil_riesgo(df: pd.DataFrame, columna: str, minimo: int = 1) -> pd.DataFrame:
    """Transacciones, sospechosas y tasa por grupo, ordenado de mayor a menor riesgo.

    `veces_el_promedio` = tasa del grupo / tasa global. Un 3 significa que el
    grupo tiene tres veces más operaciones sospechosas que el promedio.
    """
    tasa_global = df[COLUMNA_OBJETIVO].mean()
    grupos = df.groupby(columna, observed=True)
    tabla = pd.DataFrame({
        "transacciones": grupos.size(),
        "sospechosas": grupos[COLUMNA_OBJETIVO].sum(),
        "monto_total": df["monto"].astype("float64").groupby(df[columna], observed=True).sum(),
        "monto_mediano": grupos["monto"].median(),
    })
    tabla["pct_transacciones"] = 100 * tabla["transacciones"] / tabla["transacciones"].sum()
    tabla["pct_de_las_sospechosas"] = 100 * tabla["sospechosas"] / max(tabla["sospechosas"].sum(), 1)
    tabla["tasa_sospechosa_pct"] = 100 * tabla["sospechosas"] / tabla["transacciones"]
    tabla["veces_el_promedio"] = tabla["tasa_sospechosa_pct"] / (100 * tasa_global)
    tabla = tabla[tabla["transacciones"] >= minimo]
    return tabla.sort_values("tasa_sospechosa_pct", ascending=False)


FORMATO_PERFIL = {
    "transacciones": "{:,.0f}", "sospechosas": "{:,.0f}", "monto_total": "£{:,.0f}",
    "monto_mediano": "£{:,.2f}", "pct_transacciones": "{:.2f}%", "pct_de_las_sospechosas": "{:.2f}%",
    "tasa_sospechosa_pct": "{:.4f}%", "veces_el_promedio": "{:.2f}x",
}


def graficar_tasa(tabla: pd.DataFrame, titulo: str, tasa_global: float):
    """Barras horizontales de tasa sospechosa por grupo, con la tasa global de referencia."""
    t = tabla.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.38 * len(t) + 1)))
    ax.barh(t.index.astype(str), t["tasa_sospechosa_pct"], color=COLOR_SOSPECHOSA, height=0.6)
    ax.axvline(100 * tasa_global, color=COLOR_REFERENCIA, linestyle="--", linewidth=1,
               label=f"Tasa global {100 * tasa_global:.4f}%")
    for y, v in enumerate(t["tasa_sospechosa_pct"]):
        ax.text(v, y, f" {v:.3f}%", va="center", fontsize=8, color=COLOR_REFERENCIA)
    ax.set_xlabel("% de transacciones sospechosas")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _estilo(ax, titulo)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# --- Montos ------------------------------------------------------------------
def resumen_montos(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos del monto con percentiles extremos, normal vs. sospechosa."""
    tabla = (df.groupby(COLUMNA_OBJETIVO)["monto"]
             .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)
    return tabla.rename(columns={0: "normal", 1: "sospechosa"})


def graficar_montos(df: pd.DataFrame):
    """Distribución del monto (escala logarítmica), normal vs. sospechosa."""
    log_monto = np.log10(df["monto"].clip(lower=0.01).to_numpy())
    etiquetas = df[COLUMNA_OBJETIVO].to_numpy()
    bins = np.linspace(log_monto.min(), log_monto.max(), 70)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for valor, color, nombre in [(0, COLOR_NORMAL, "Normal"), (1, COLOR_SOSPECHOSA, "Sospechosa")]:
        mascara = etiquetas == valor
        if mascara.any():
            ax.hist(log_monto[mascara], bins=bins, density=True, histtype="step",
                    linewidth=2, color=color, label=nombre)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{10 ** x:,.0f}"))
    ax.set_xlabel("Monto por transacción (escala logarítmica)")
    ax.set_ylabel("Densidad")
    ax.legend(frameon=False)
    _estilo(ax, "Distribución del monto: normal vs. sospechosa")
    fig.tight_layout()
    return fig


# --- Tipologías (solo análisis) ----------------------------------------------
GLOSARIO_TIPOLOGIAS = {
    # El orden importa: se usa la primera clave contenida en el nombre.
    "structur": "Fraccionamiento: dividir una suma grande en depósitos menores para no rebasar umbrales.",
    "smurf": "Pitufeo: varias personas o cuentas hacen depósitos chicos que terminan en una misma cuenta.",
    "layered_fan": "Dispersión en capas: una cuenta reparte a muchas y estas vuelven a repartir.",
    "stack": "Bipartita apilada: varias capas de grupos de cuentas que se transfieren entre sí.",
    "layer": "Estratificación: capas de transferencias para alejar el dinero de su origen.",
    "fan_in": "Concentración (fan-in): muchas cuentas envían a una sola.",
    "fan_out": "Dispersión (fan-out): una cuenta envía a muchas.",
    "gather": "Reunir y dispersar: se juntan fondos de varias cuentas y luego se reparten.",
    "scatter": "Dispersar y reunir: los fondos se reparten y después se vuelven a juntar.",
    "cycle": "Ciclo: el dinero sale y regresa al origen pasando por varias cuentas.",
    "bipartite": "Bipartita: un grupo de cuentas transfiere a otro grupo.",
    "deposit": "Depósito y envío: entra efectivo y sale rápido por transferencia.",
    "cash_withdrawal": "Retiro en efectivo: los fondos salen del sistema en efectivo.",
    "behavio": "Cambio de comportamiento: la cuenta se aparta de su patrón histórico.",
    "invoic": "Facturación alterada: pagos comerciales a precios inflados para mover valor.",
    "single_large": "Operación única grande: un solo movimiento de monto inusualmente alto.",
}


def describir_tipologia(nombre: str) -> str:
    clave = str(nombre).lower().replace("-", "_").replace(" ", "_")
    return next((v for k, v in GLOSARIO_TIPOLOGIAS.items() if k in clave), "")


def verificar_fuga_tipologia(df: pd.DataFrame) -> pd.DataFrame:
    """% de transacciones sospechosas dentro de cada tipología.

    Si todas son 0 % o 100 %, conocer la tipología equivale a conocer la
    respuesta: por eso no puede usarse en reglas ni modelos.
    """
    tabla = df.groupby(COLUMNA_TIPOLOGIA, observed=True).agg(
        transacciones=(COLUMNA_OBJETIVO, "size"), pct_sospechosa=(COLUMNA_OBJETIVO, "mean"))
    tabla["pct_sospechosa"] *= 100
    return tabla.sort_values(["pct_sospechosa", "transacciones"], ascending=False)


def resumen_tipologias(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil de cada tipología sospechosa: volumen, monto, canal principal y % internacional."""
    sosp = df[df[COLUMNA_OBJETIVO] == 1]
    grupos = sosp.groupby(COLUMNA_TIPOLOGIA, observed=True)
    tabla = grupos.agg(transacciones=("monto", "size"), monto_mediano=("monto", "median"),
                       cuentas_que_envian=("cuenta_origen", "nunique"))
    tabla["pct_de_las_sospechosas"] = 100 * tabla["transacciones"] / tabla["transacciones"].sum()
    tabla["canal_principal"] = grupos["tipo_pago"].agg(lambda s: s.value_counts().idxmax())
    tabla["pct_internacional"] = 100 * grupos["es_internacional"].mean()
    tabla["descripcion"] = [describir_tipologia(t) for t in tabla.index]
    return tabla.sort_values("transacciones", ascending=False)


def graficar_tipologia_canal(df: pd.DataFrame, orden: list):
    """Mapa de calor: % de cada tipología sospechosa por tipo de pago (cada fila suma 100 %)."""
    sosp = df[df[COLUMNA_OBJETIVO] == 1]
    tabla = pd.crosstab(sosp[COLUMNA_TIPOLOGIA], sosp["tipo_pago"], normalize="index") * 100
    tabla = tabla.loc[orden]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(tabla) + 1.6))
    imagen = ax.imshow(tabla.values, cmap="Oranges", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(tabla.shape[1]), tabla.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(range(tabla.shape[0]), tabla.index.astype(str))
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            v = tabla.values[i, j]
            if v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else "#0b0b0b")
    ax.set_title("Tipologías sospechosas por tipo de pago (% de cada fila)", loc="left", fontsize=11)
    fig.colorbar(imagen, ax=ax, label="% de las transacciones de la tipología")
    fig.tight_layout()
    return fig


# --- Cuentas -----------------------------------------------------------------
def resumen_cuentas(df: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Cuántas cuentas tocan operaciones sospechosas y cuánto mezclan con operaciones normales."""
    y = df[COLUMNA_OBJETIVO] == 1
    todas = np.union1d(df["cuenta_origen"].unique(), df["cuenta_destino"].unique())
    involucradas = np.union1d(df.loc[y, "cuenta_origen"].unique(), df.loc[y, "cuenta_destino"].unique())
    envios = df.groupby("cuenta_origen")[COLUMNA_OBJETIVO].agg(["size", "sum"])
    con_sosp = envios[envios["sum"] > 0]
    fraccion = con_sosp["sum"] / con_sosp["size"]
    resumen = {
        "cuentas_con_alguna_operacion_sospechosa": int(len(involucradas)),
        "pct_del_total_de_cuentas": round(100 * len(involucradas) / len(todas), 4),
        "cuentas_que_envian_alguna_sospechosa": int(len(con_sosp)),
        "envios_por_cuenta_mediana": float(envios["size"].median()),
        "envios_por_cuenta_p99": float(envios["size"].quantile(0.99)),
        "pct_sospechoso_de_sus_envios_mediana": round(100 * float(fraccion.median()), 2),
        "pct_de_esas_cuentas_que_tambien_operan_normal": round(100 * float((fraccion < 1).mean()), 2),
    }
    return resumen, fraccion


def graficar_mezcla_cuentas(fraccion: pd.Series):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(100 * fraccion, bins=40, color=COLOR_SOSPECHOSA, edgecolor="white", linewidth=1)
    ax.set_xlabel("% de los envíos de la cuenta que son sospechosos")
    ax.set_ylabel("Cuentas")
    _estilo(ax, "Cuentas que envían alguna operación sospechosa: ¿qué % de sus envíos lo son?")
    fig.tight_layout()
    return fig


# --- Tiempo ------------------------------------------------------------------
def resumen_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Transacciones, sospechosas, días con datos y tasa por mes."""
    mes = df["fecha_hora"].dt.to_period("M")
    tabla = df.groupby(mes).agg(transacciones=(COLUMNA_OBJETIVO, "size"),
                                sospechosas=(COLUMNA_OBJETIVO, "sum"))
    tabla["dias_con_datos"] = df["fecha_hora"].dt.normalize().groupby(mes).nunique()
    tabla["tasa_sospechosa_pct"] = 100 * tabla["sospechosas"] / tabla["transacciones"]
    tabla.index.name = "mes"
    return tabla


def graficar_mensual(tabla: pd.DataFrame, tasa_global: float):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(tabla.index.astype(str), tabla["tasa_sospechosa_pct"], color=COLOR_SOSPECHOSA, width=0.6)
    ax.axhline(100 * tasa_global, color=COLOR_REFERENCIA, linestyle="--", linewidth=1,
               label=f"Tasa global {100 * tasa_global:.4f}%")
    ax.set_ylabel("% sospechosas")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=8)
    _estilo(ax, "Tasa de transacciones sospechosas por mes")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# --- Gráficas de reglas (etapa 2) -------------------------------------------
def graficar_alertas_por_dia(resumen: pd.DataFrame):
    """Barras horizontales: alertas por día que genera cada regla."""
    t = resumen.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(2.5, 0.42 * len(t) + 1)))
    ax.barh(t.index.astype(str), t["alertas_por_dia"], color=COLOR_NORMAL, height=0.6)
    for y, v in enumerate(t["alertas_por_dia"]):
        ax.text(v, y, f" {v:,.1f}", va="center", fontsize=8, color=COLOR_REFERENCIA)
    ax.set_xlabel("Alertas por día (cuenta + semana)")
    _estilo(ax, "Carga de trabajo: alertas por día de cada regla")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def graficar_precision_recall(resumen: pd.DataFrame):
    """Cada regla como un punto: recall (cuánto detecta) vs. precisión (cuánto acierta)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(resumen["recall_pct"], resumen["precision_alertas_pct"], s=70, color=COLOR_SOSPECHOSA,
               edgecolor="white", linewidth=2, zorder=3)
    # Etiqueta corta (R1, R2a, ...) para que no se encimen; el nombre completo está en la tabla
    for nombre, fila in resumen.iterrows():
        corto = "Todas" if nombre.startswith("Todas") else nombre.split()[0]
        ax.annotate(corto, (fila["recall_pct"], fila["precision_alertas_pct"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Recall: % de las transacciones sospechosas detectadas")
    ax.set_ylabel("Precisión: % de alertas con algo sospechoso")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _estilo(ax, "Cada regla: cuánto detecta vs. cuánto acierta")
    fig.tight_layout()
    return fig


def graficar_recall_tipologia(tabla: pd.DataFrame):
    """Mapa de calor: % de cada tipología que detecta cada regla."""
    datos = tabla.drop(columns="transacciones")
    fig, ax = plt.subplots(figsize=(1.1 * datos.shape[1] + 4, 0.42 * datos.shape[0] + 2))
    imagen = ax.imshow(datos.values, cmap="Oranges", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(datos.shape[1]), datos.columns, rotation=35, ha="right")
    ax.set_yticks(range(datos.shape[0]), datos.index)
    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            v = datos.values[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 60 else "#0b0b0b")
    ax.set_title("Recall por tipología: % detectado por cada regla", loc="left", fontsize=11)
    fig.colorbar(imagen, ax=ax, label="% de las transacciones de la tipología")
    fig.tight_layout()
    return fig
