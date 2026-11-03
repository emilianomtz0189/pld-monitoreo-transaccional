# Datos

Este repositorio **no incluye el dataset**: pesa cerca de 1 GB y está publicado en Kaggle por sus autores. Aquí se explica cómo obtenerlo.

## Fuente

**SAML-D: Synthetic Transaction Monitoring Dataset for AML** — Kaggle
https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

Artículo de referencia: B. Oztas, D. Cetinkaya, F. Adedoyin, M. Budka, H. Dogan y G. Aksu, *"Enhancing Anti-Money Laundering: Development of a Synthetic Transaction Monitoring Dataset"*, IEEE ICEBE 2023, pp. 47-54.

Es un dataset **sintético** (simulado a partir de entrevistas con especialistas en PLD, literatura y datasets previos): montos en libras esterlinas y geografía centrada en Reino Unido.

## Opción A: automática en Colab (la que usan los notebooks)

El notebook `01_descarga_y_exploracion` descarga el CSV con `kagglehub`, lo convierte a Parquet con tipos optimizados y lo respalda en `MyDrive/pld-monitoreo-transaccional/`. En sesiones posteriores carga el Parquet desde Drive en segundos.

Si Kaggle pide credenciales:
1. En kaggle.com → *Settings* → *API* → *Create New Token* (descarga `kaggle.json`).
2. En Colab, panel de **Secretos** (ícono de llave): agrega `KAGGLE_USERNAME` y `KAGGLE_KEY` con los valores de ese archivo y habilita el acceso del notebook.

## Opción B: en local

Descarga el CSV desde la página de Kaggle y colócalo en `data/raw/SAML-D.csv`.

## Estructura esperada

```
data/
├── raw/
│   └── SAML-D.csv              # CSV original de Kaggle (no se sube)
└── processed/
    └── transacciones.parquet   # Tipos optimizados y nombres en español (no se sube)
```

## Diccionario de variables

| Nombre original (SAML-D) | Nombre en el proyecto | Descripción |
|---|---|---|
| Date + Time | `fecha_hora` | Fecha y hora de la transacción. |
| Sender_account | `cuenta_origen` | Cuenta que envía. |
| Receiver_account | `cuenta_destino` | Cuenta que recibe. |
| Amount | `monto` | Monto en libras esterlinas (£). |
| Payment_currency | `moneda_pago` | Moneda en que se paga. |
| Received_currency | `moneda_recibida` | Moneda en que se recibe. |
| Sender_bank_location | `pais_banco_origen` | País del banco que envía. |
| Receiver_bank_location | `pais_banco_destino` | País del banco que recibe. |
| Payment_type | `tipo_pago` | Canal: efectivo, transferencia, tarjeta, cheque, etc. |
| Is_laundering | `es_sospechosa` | **Variable objetivo.** 1 si la transacción forma parte de un esquema de lavado. |
| Laundering_type | `tipologia` | Tipología normal o sospechosa. **Revela la etiqueta: solo se usa para analizar resultados, nunca en reglas ni modelos.** |
