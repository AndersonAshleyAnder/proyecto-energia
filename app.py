import os
import sys
import calendar
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, exceptions
import plotly.express as px
from pandas import Timestamp

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Asegurar que la carpeta del proyecto esté en sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Importar utilidades de dataset
from utils.procesar_dataset import cargar_y_procesar_dataset, listar_datasets

app = dash.Dash(__name__)
app.title = "Predicción Energía (TS)"

# ==========================
# Formateo de valores (COP)
# ==========================
UNIT = "COP/MWh"   # Ajusta si tu dataset usa otra unidad, p. ej. "COP/kWh"
CURRENCY = "COP "  # Prefijo de moneda

# ¿Quieres ver en miles o millones? Descomenta UNA de estas líneas:
# SCALE = 1        ; SCALE_LABEL = ""              # valor completo (por defecto)
# SCALE = 1_000    ; SCALE_LABEL = " mil"          # en miles
# SCALE = 1_000_000; SCALE_LABEL = " millones"     # en millones

# Por defecto, valor completo:
SCALE = 1
SCALE_LABEL = ""

def fmt_value(x: float) -> str:
    """Devuelve 'COP 1,234,567.89 /MWh' (o en miles/millones si lo configuras)."""
    try:
        val = x / SCALE
        # Nota: formato US (coma millares, punto decimal). Si prefieres estilo local, puedes usar 'locale'.
        return f"{CURRENCY}{val:,.2f}{(' ' + SCALE_LABEL) if SCALE != 1 else ''} /{UNIT.split('/')[-1]}"
    except Exception:
        return str(x)

# ---------------------------
# Utilidades internas
# ---------------------------
def build_time_features(df, col_fecha="__auto__"):
    """
    Enriquecer el dataframe con señales cíclicas y rezagos.
    - Detecta la columna datetime si col_fecha="__auto__".
    - Crea: Month_sin/cos, Hour_sin/cos, rezagos: lag_1, lag_24 y rolling_mean_24.
    - Devuelve df ordenado por fecha y con filas NaN eliminadas (tras lags).
    """
    # Detectar columna de fecha
    if col_fecha == "__auto__":
        datetime_cols = [c for c in df.columns if str(df[c].dtype).startswith("datetime64")]
        if not datetime_cols:
            raise ValueError("No se encontró una columna de fecha en formato datetime64.")
        col_fecha = datetime_cols[0]

    # Orden por fecha
    df = df.sort_values(col_fecha).reset_index(drop=True)

    # Señales cíclicas mes/hora
    if "Month" in df.columns:
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    else:
        df["Month_sin"] = np.sin(2 * np.pi * df[col_fecha].dt.month / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df[col_fecha].dt.month / 12)

    if "Hour" in df.columns:
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    else:
        df["Hour_sin"] = np.sin(2 * np.pi * df[col_fecha].dt.hour / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df[col_fecha].dt.hour / 24)

    # Rezagos básicos sobre el target
    if "Valor_mean" not in df.columns:
        raise ValueError("No se encontró la columna 'Valor_mean' tras el procesamiento.")
    df["lag_1"] = df["Valor_mean"].shift(1)
    df["lag_24"] = df["Valor_mean"].shift(24)  # mismo horario del día anterior (si la serie es horaria)
    df["rolling_mean_24"] = df["Valor_mean"].rolling(window=24, min_periods=1).mean().shift(1)

    # Eliminar filas iniciales con NaN por los rezagos (especialmente primeras 24 h)
    df = df.dropna().reset_index(drop=True)
    return df, col_fecha


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = (mean_squared_error(y_true, y_pred)) ** 0.5  # compatible con versiones viejas
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ---------------------------
# Datasets y UI
# ---------------------------
datasets = listar_datasets()
if not datasets:
    raise RuntimeError("No se encontraron archivos .csv en la carpeta 'data/'. "
                       "Asegúrate de colocar tus datasets en data/")

def _basename(p):
    try:
        return os.path.basename(p)
    except Exception:
        return p

help_text = dcc.Markdown("""
### ¿Cómo usar esta app?

**Modelo**
- *Regresión Lineal:* baseline simple, útil para comparar.
- *Random Forest:* captura relaciones no lineales. Ajusta `n_estimators` (árboles) y `max_depth` (profundidad).

**Métricas (sobre validación temporal)**
- **MAE:** error absoluto medio (↓ mejor).
- **RMSE:** raíz del error cuadrático medio (↓ mejor).
- **R²:** proporción de varianza explicada (↑ mejor).

**Gráficas**
- *Serie temporal:* `Valor_mean` a lo largo del tiempo (con estrella del escenario).
- *Predicción vs Real (última partición):* comparación visual del ajuste más reciente (estrella si cae en rango).
- *What‑if por hora:* predicciones 0–23 h para Year/Month seleccionados (estrella en la hora elegida).
- *Importancias (RF):* contribución relativa de cada feature al modelo.

**Features (activables)**
- Señales cíclicas: `Month_sin/cos`, `Hour_sin/cos`.
- Rezagos: `lag_1`, `lag_24`, `rolling_mean_24`.

**Validación de series**
- `TimeSeriesSplit`: no mezcla futuro con pasado. Ajusta `n_splits`.

**Consejos**
- Empieza con RF (`n_estimators≈300`, `max_depth` vacío).
- Activa rezagos si tu serie es horaria. Si no, prueba sin ellos.
""")

app.layout = html.Div([
    html.H1("Predicción de Ventas de Energía en Bolsa Nacional (Time Series)", style={"textAlign": "center"}),

    dcc.Tabs([
        dcc.Tab(label="Modelado", children=[
            html.Div([
                html.Label("Dataset:"),
                dcc.Dropdown(
                    id="dataset-selector",
                    options=[{"label": _basename(ds), "value": ds} for ds in datasets],
                    value=datasets[0],
                    clearable=False
                ),
            ], style={"marginBottom": "8px"}),

            html.Div([
                html.Label("Modelo:"),
                dcc.RadioItems(
                    id="model-selector",
                    options=[
                        {"label": "Regresión Lineal", "value": "lr"},
                        {"label": "Random Forest", "value": "rf"},
                    ],
                    value="rf",
                    inline=True
                ),
            ], style={"marginBottom": "4px"}),

            html.Div([
                html.Label("Parámetros Random Forest:"),
                html.Div([
                    html.Label("n_estimators"),
                    dcc.Slider(id="rf-n-estimators", min=100, max=800, step=50, value=300,
                               marks={i: str(i) for i in range(100, 801, 100)}),
                ], style={"marginTop": "6px"}),
                html.Div([
                    html.Label("max_depth (vacío = None)"),
                    dcc.Input(id="rf-max-depth", type="number", placeholder="None", debounce=True),
                ], style={"marginTop": "6px"}),
            ], id="rf-params", style={"marginBottom": "12px"}),

            html.Div([
                html.Label("Features:"),
                dcc.Checklist(
                    id="feature-options",
                    options=[
                        {"label": "Usar señales cíclicas (sin/cos de Mes y Hora)", "value": "cyc"},
                        {"label": "Usar rezagos (lag_1, lag_24, rolling_mean_24)", "value": "lags"},
                    ],
                    value=["cyc", "lags"],  # activadas por defecto
                    inputStyle={"margin-right": "6px"}
                ),
            ], style={"marginBottom": "12px"}),

            html.Div([
                html.Label("TimeSeriesSplit (n_splits):"),
                dcc.Slider(id="tscv-splits", min=3, max=8, step=1, value=5,
                           marks={i: str(i) for i in range(3, 9)})
            ], style={"marginBottom": "12px"}),

            html.Div([
                html.Label("Año:"), dcc.Input(id="input-year", type="number", value=2024, debounce=True),
                html.Label("Mes:"), dcc.Input(id="input-month", type="number", value=1, debounce=True),
                html.Label("Hora:"), dcc.Input(id="input-hour", type="number", value=10, debounce=True),
            ], style={"display": "flex", "gap": "10px"}),

            # KPI grande del valor estimado
            html.Div(
                id="kpi-estimado",
                style={"marginTop": "12px", "fontSize": "26px", "fontWeight": "700"}
            ),

            html.Div(id="info-dataset", style={"marginTop": "8px", "fontStyle": "italic"}),

            dcc.Graph(id="grafica-lineal"),
            dcc.Graph(id="pred-real-graph"),
            dcc.Graph(id="whatif-graph"),
            dcc.Graph(id="feat-importance"),

            html.Div(id="prediccion-output", style={"marginTop": "12px", "fontSize": "18px"}),
            html.Div(id="cv-output", style={"marginTop": "6px", "color": "#555"}),
        ]),

        dcc.Tab(label="Ayuda", children=[
            html.Div(help_text, style={"padding": "12px"})
        ])
    ])
])

# ---------------------------
# Callback principal
# ---------------------------
@app.callback(
    Output("kpi-estimado", "children"),
    Output("grafica-lineal", "figure"),
    Output("pred-real-graph", "figure"),
    Output("whatif-graph", "figure"),
    Output("feat-importance", "figure"),
    Output("prediccion-output", "children"),
    Output("info-dataset", "children"),
    Output("cv-output", "children"),
    Input("dataset-selector", "value"),
    Input("model-selector", "value"),
    Input("rf-n-estimators", "value"),
    Input("rf-max-depth", "value"),
    Input("feature-options", "value"),
    Input("tscv-splits", "value"),
    Input("input-year", "value"),
    Input("input-month", "value"),
    Input("input-hour", "value"),
)
def actualizar_app(dataset_path, model_key, n_estimators, max_depth, feat_opts, n_splits, year, month, hour):
    # Validaciones básicas
    if not all(v is not None for v in [year, month, hour]):
        raise exceptions.PreventUpdate
    if not (1 <= int(month) <= 12):
        msg = "⚠️ Mes debe estar entre 1 y 12."
        return msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update, msg, dash.no_update, dash.no_update
    if not (0 <= int(hour) <= 23):
        msg = "⚠️ Hora debe estar entre 0 y 23."
        return msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update, msg, dash.no_update, dash.no_update
    if not (3 <= int(n_splits) <= 10):
        n_splits = 5

    # 1) Cargar y procesar dataset base
    df_raw = cargar_y_procesar_dataset(dataset_path)

    # 2) Enriquecer con features (siempre se crean; luego seleccionamos según UI)
    df_full, col_fecha = build_time_features(df_raw)

    # 3) Seleccionar features según UI
    use_cyc = "cyc" in (feat_opts or [])
    use_lags = "lags" in (feat_opts or [])

    base_feats = ["Year", "Month", "Hour"]
    cyc_feats = ["Month_sin", "Month_cos", "Hour_sin", "Hour_cos"] if use_cyc else []
    lag_feats = ["lag_1", "lag_24", "rolling_mean_24"] if use_lags else []

    feature_cols = base_feats + cyc_feats + lag_feats
    if not feature_cols:
        feature_cols = base_feats

    # Si hay muy pocas filas tras lags, desactivar lags automáticamente
    df = df_full.copy()
    info_warning = ""
    if use_lags and len(df) < 30:
        lag_feats = []
        feature_cols = base_feats + cyc_feats
        use_lags = False
        info_warning = " | Nota: dataset corto; se desactivaron rezagos automáticamente."

    X = df[feature_cols]
    y = df["Valor_mean"]

    # 4) Validación tipo serie: TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    mae_list, rmse_list, r2_list = [], [], []
    last_fold_info = None

    for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if model_key == "rf":
            md = RandomForestRegressor(
                n_estimators=int(n_estimators) if n_estimators else 300,
                max_depth=int(max_depth) if max_depth and int(max_depth) > 0 else None,
                random_state=42,
                n_jobs=-1
            )
            model_name = "Random Forest"
        else:
            md = LinearRegression()
            model_name = "Regresión Lineal"

        md.fit(X_tr, y_tr)
        y_pred = md.predict(X_te)

        mae, rmse, r2 = evaluate_regression(y_te, y_pred)
        mae_list.append(mae); rmse_list.append(rmse); r2_list.append(r2)

        last_fold_info = {
            "model": md,
            "X_train": X_tr, "y_train": y_tr,
            "X_test": X_te, "y_test": y_te,
            "test_index": te_idx
        }

    if last_fold_info is None:
        raise exceptions.PreventUpdate

    # 5) Métricas agregadas y del último fold
    mae_avg, rmse_avg, r2_avg = np.mean(mae_list), np.mean(rmse_list), np.mean(r2_list)
    md = last_fold_info["model"]
    X_test, y_test = last_fold_info["X_test"], last_fold_info["y_test"]
    test_index = last_fold_info["test_index"]

    y_pred_test = md.predict(X_test)
    mae_last, rmse_last, r2_last = evaluate_regression(y_test, y_pred_test)

    # 6) Construir timestamp del escenario (Año/Mes/Hora + día válido)
    last_data_day = int(df_full[col_fecha].dt.day.iloc[-1])
    days_in_month = calendar.monthrange(int(year), int(month))[1]
    day = min(last_data_day, days_in_month)
    ts_target = Timestamp(int(year), int(month), int(day), int(hour))

    # 7) Predicción puntual (escenario) — se usa para KPI y estrellas
    #    Para lags usamos como proxy los últimos valores disponibles del dataset
    X_pred_values = {"Year": int(year), "Month": int(month), "Hour": int(hour)}
    if use_cyc:
        X_pred_values.update({
            "Month_sin": np.sin(2*np.pi*int(month)/12),
            "Month_cos": np.cos(2*np.pi*int(month)/12),
            "Hour_sin":  np.sin(2*np.pi*int(hour)/24),
            "Hour_cos":  np.cos(2*np.pi*int(hour)/24),
        })
    if use_lags:
        last_row = df_full.iloc[-1]
        X_pred_values.update({
            "lag_1": last_row["Valor_mean"],
            "lag_24": last_row.get("lag_24", last_row["Valor_mean"]),
            "rolling_mean_24": last_row.get("rolling_mean_24", last_row["Valor_mean"]),
        })
    X_pred = pd.DataFrame([X_pred_values], columns=feature_cols)
    pred_value = md.predict(X_pred)[0]
    pred_esc = fmt_value(pred_value)  # formateado para mostrar

    # 8) Figuras
    # Serie temporal completa + estrella del escenario (A)
    fig_ts = px.line(df_full, x=col_fecha, y="Valor_mean",
                     title="Comportamiento del Valor Promedio en el Tiempo")
    fig_ts.add_scatter(
        x=[ts_target],
        y=[pred_value],
        mode="markers+text",
        name=f"Pred (escenario): {pred_esc}",
        marker=dict(size=12, color="orange", symbol="star"),
        hovertemplate="<b>%{text}</b><extra></extra>",
        text=[pred_esc], textposition="top center", textfont=dict(size=12, color="orange")
    )

    # Predicción vs Real (último fold) + estrella si ts_target cae dentro del rango (B)
    df_te = df_full.iloc[test_index].copy()
    df_te = df_te.assign(Real=y_test.values, Pred=y_pred_test)
    fig_pred_real = px.line(df_te, x=col_fecha, y=["Real", "Pred"],
                            title="Predicción vs Real (última partición temporal)")
    te_min, te_max = df_te[col_fecha].min(), df_te[col_fecha].max()
    if te_min <= ts_target <= te_max:
        fig_pred_real.add_scatter(
            x=[ts_target],
            y=[pred_value],
            mode="markers+text",
            name=f"Pred (escenario): {pred_esc}",
            marker=dict(size=12, color="orange", symbol="star"),
            hovertemplate="<b>%{text}</b><extra></extra>",
            text=[pred_esc], textposition="top center", textfont=dict(size=12, color="orange")
        )

    # What-if por hora (C): Year/Month fijos, horas 0..23 (estrella en la hora elegida)
    hours = np.arange(24)
    wf = pd.DataFrame({"Year": int(year), "Month": int(month), "Hour": hours})
    if use_cyc:
        wf["Month_sin"] = np.sin(2*np.pi*wf["Month"]/12)
        wf["Month_cos"] = np.cos(2*np.pi*wf["Month"]/12)
        wf["Hour_sin"]  = np.sin(2*np.pi*wf["Hour"]/24)
        wf["Hour_cos"]  = np.cos(2*np.pi*wf["Hour"]/24)
    if use_lags:
        last_row = df_full.iloc[-1]
        wf["lag_1"] = last_row["Valor_mean"]
        wf["lag_24"] = last_row.get("lag_24", last_row["Valor_mean"])
        wf["rolling_mean_24"] = last_row.get("rolling_mean_24", last_row["Valor_mean"])
    wf = wf[[c for c in feature_cols]]  # asegurar orden de columnas
    pred_wf = md.predict(wf)

    fig_wf = px.line(x=hours, y=pred_wf, labels={"x": "Hora", "y": "Predicción"},
                     title=f"What-if: Predicción por hora (Year={year}, Month={month})")
    fig_wf.add_scatter(
        x=[int(hour)],
        y=[pred_value],
        mode="markers+text",
        name=f"Pred (escenario): {pred_esc}",
        marker=dict(size=12, color="orange", symbol="star"),
        hovertemplate="<b>%{text}</b><extra></extra>",
        text=[pred_esc], textposition="top center", textfont=dict(size=12, color="orange")
    )

    # Importancias (solo RF)
    if model_key == "rf" and hasattr(md, "feature_importances_"):
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": md.feature_importances_
        }).sort_values("importance", ascending=False)
        fig_imp = px.bar(
            imp_df, x="importance", y="feature", orientation="h",
            title="Importancia de características (Random Forest)"
        )
    else:
        fig_imp = px.scatter(title="Sin importancias: el modelo actual no es Random Forest")

    # 9) Textos informativos (KPI + métricas + detalle)
    kpi_text = f"Valor estimado: {pred_esc}"

    pred_text = (
        f"📈 Predicción estimada ({model_name}) para "
        f"Year={year}, Month={month}, Hour={hour}: "
        f"<b>{pred_esc}</b><br>"
        f"🔎 Última partición → MAE: {fmt_value(mae_last)} | RMSE: {fmt_value(rmse_last)} | R²: {r2_last:,.3f}"
    )

    info = (
        f"Dataset: {os.path.basename(dataset_path)} | Filas tras lags: {len(df_full)} | "
        f"Modelo: {model_name} | TSCV: {len(mae_list)} particiones | "
        f"Features: {', '.join(feature_cols)}"
        f"{info_warning}"
    )

    cv_text = (
        f"📊 Promedio en validación temporal (TSCV) → "
        f"MAE: {fmt_value(mae_avg)} | RMSE: {fmt_value(rmse_avg)} | R²: {r2_avg:,.3f}"
    )

    return kpi_text, fig_ts, fig_pred_real, fig_wf, fig_imp, pred_text, info, cv_text


if __name__ == "__main__":
    app.run(debug=True, port=8050)