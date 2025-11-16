import pandas as pd
import os

def cargar_y_procesar_dataset(ruta):
    """
    Carga y procesa un dataset dado su ruta.
    Detecta automáticamente la columna de fecha.
    Devuelve el dataframe procesado.
    """

    df = pd.read_csv(ruta)

    # Intento 1: Lista extendida de posibles nombres de columnas de fecha
    posibles_fechas = [
        "Fecha", "fecha", "FECHA", "Date", "date",
        "FechaHora_hora", "datetime", "timestamp", "time"
    ]

    col_fecha = None

    # Buscar coincidencias directas
    for c in posibles_fechas:
        if c in df.columns:
            col_fecha = c
            break

    # Intento 2 (más robusto): detectar automáticamente una columna tipo fecha
    if col_fecha is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                col_fecha = col
                break
            except:
                pass

    if col_fecha is None:
        raise ValueError("❌ No se encontró una columna de fecha en el dataset.")

    # Convertir la columna encontrada a datetime
    df[col_fecha] = pd.to_datetime(df[col_fecha])

    # Crear columnas temporales
    df["Year"] = df[col_fecha].dt.year
    df["Month"] = df[col_fecha].dt.month
    df["Day"] = df[col_fecha].dt.day

    # Dataset con fecha/hora normalmente contiene horas
    try:
        df["Hour"] = df[col_fecha].dt.hour
    except:
        df["Hour"] = 0   # Si no tiene hora, ponemos 0 por defecto

    # Revisar si existe Valor_mean, de lo contrario crearlo
    if "Valor_mean" not in df.columns:
        if "Vlr_VB" in df.columns:
            df["Valor_mean"] = df["Vlr_VB"]
        elif "Valor_sum" in df.columns:
            df["Valor_mean"] = df["Valor_sum"]
        else:
            raise ValueError(
                "❌ No existe la columna Valor_mean, Vlr_VB ni Valor_sum en el dataset."
            )

    return df


def listar_datasets():
    """
    Devuelve una lista de todos los .csv dentro de /data
    """
    base = "data"
    archivos = [f for f in os.listdir(base) if f.endswith(".csv")]
    return [os.path.join(base, f) for f in archivos]
