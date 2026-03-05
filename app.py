# app.py — Dashboard Streamlit (idéntico al notebook del PDF)
# Incluye:
# - Tablas por método (Originales, PM multi-ventana 2..24, Holt-Winters, Desestacionalización)
# - Tabla comparativa final con heatmap por error %
# - Botón de descarga EDB.xlsx
# - Botón/expander por gráfico para ver el código de ese gráfico
# - Selector de ventana PM (para la visualización del gráfico), manteniendo Forecast_MA12 fijo para la tabla final (igual al notebook)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# CONFIG / UI
# ============================================================
st.set_page_config(
    page_title="Remesas Guatemala | Pronóstico",
    page_icon="🦇",
    layout="wide"
)

st.markdown("""
<style>
.main-title{font-size:42px;font-weight:800;color:#E6EEF7;}
.subtitle{font-size:18px;color:#A8B9CC;margin-bottom:30px;}

.card{
background: linear-gradient(145deg,#0A1F33,#061321);
padding:25px;border-radius:15px;
border:1px solid rgba(201,162,39,0.35);
box-shadow:0px 8px 20px rgba(0,0,0,0.5);
}
.metric{font-size:32px;font-weight:700;color:#C9A227;}
.metric-label{font-size:14px;color:#A8B9CC;}

hr{border:0;height:1px;background:rgba(201,162,39,.35);}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Controles")
W_SHOW = st.sidebar.selectbox(
    "Ventana Promedio Móvil (par) — para visualizar en gráficos",
    [2,4,6,8,10,12,14,16,18,20,22,24],
    index=5
)
st.sidebar.caption("Paleta: azul oscuro + dorado")


# ============================================================
# DATA
# ============================================================
@st.cache_data
def load_data(path="EDB.xlsx"):
    df = pd.read_excel(path, sheet_name="Dataset")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    # Asegurar nombres esperados
    if "Value" not in df.columns:
        raise KeyError("No se encontró la columna 'Value' en la hoja 'Dataset'.")
    return df

df = load_data("EDB.xlsx")


# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-title">Pronóstico de Remesas – Guatemala</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comparación de modelos básicos y proyección 2026</p>', unsafe_allow_html=True)

st.markdown('<p style="font-size:20px;font-weight:700;color:#E6EEF7;margin:0;">Econometría II — Ing. Héctor Francisco Galeros Juárez</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:14px;color:#A8B9CC;margin-top:6px;margin-bottom:0;">Gabriel Estuardo Maas Ordóñez - 1183922</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:14px;color:#A8B9CC;margin-top:2px;margin-bottom:0;">José Eduardo Yoc García - 1056823</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:14px;color:#A8B9CC;margin-top:2px;margin-bottom:0;">María Isabel Puddy Díaz - 1043723</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Descarga del Excel (misma estética del theme)
with open("EDB.xlsx", "rb") as f:
    st.download_button(
        label="⬇️ Descargar EDB.xlsx",
        data=f,
        file_name="EDB.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with st.expander("Ver dataset (primeras filas)"):
    st.dataframe(df.head(15), use_container_width=True)


# ============================================================
# RANGOS (igual al notebook del PDF)
# ============================================================
test = df[(df["Fecha"] >= "2024-04-01") & (df["Fecha"] <= "2025-03-01")].copy()        # abr-2024..mar-2025
train_test = df[df["Fecha"] <= "2024-03-01"].copy()                                    # hasta mar-2024

train_forecast = df[df["Fecha"] <= "2025-03-01"].copy()                                # hasta mar-2025
forecast_range = pd.date_range("2025-04-01", periods=12, freq="MS")                    # abr-2025..mar-2026


# ============================================================
# MODELOS (idénticos al notebook)
# ============================================================
def forecast_original_ols(train_df, H):
    y_train = train_df["Value"].values
    t_train = np.arange(1, len(y_train) + 1)
    X_train = sm.add_constant(t_train)
    model = sm.OLS(y_train, X_train).fit()

    t_fore = np.arange(len(y_train) + 1, len(y_train) + H + 1)
    X_fore = sm.add_constant(t_fore)
    return model.predict(X_fore)


def pm_forecasts_multi(train_df, H):
    """
    Replica el notebook:
    ventanas pares 2..24
    PM(w) -> centrado (rolling 2)
    OLS sobre ma_valid
    forecast H
    """
    y = train_df["Value"].reset_index(drop=True)
    windows = list(range(2, 25, 2))  # 2,4,...,24
    forecasts = {}

    for w in windows:
        ma = y.rolling(window=w).mean()
        ma_c = ma.rolling(window=2).mean()
        ma_valid = ma_c.dropna().reset_index(drop=True)

        t_ma = np.arange(1, len(ma_valid) + 1)
        X_ma = sm.add_constant(t_ma)
        model_ma = sm.OLS(ma_valid.values, X_ma).fit()

        t_fore = np.arange(len(ma_valid) + 1, len(ma_valid) + H + 1)
        X_fore = sm.add_constant(t_fore)
        forecasts[w] = model_ma.predict(X_fore)

    return forecasts


def pm_table_multi(test_df, forecasts_dict):
    windows = sorted(forecasts_dict.keys())
    tabla_pm = pd.DataFrame({
        "Mes-Año": test_df["Fecha"].dt.strftime("%b-%Y").values,
        "Real": test_df["Value"].values
    })
    for w in windows:
        tabla_pm[f"Ancho {w}"] = forecasts_dict[w]
    return tabla_pm


def forecast_hw(train_df, H):
    y_train = train_df.set_index("Fecha")["Value"].asfreq("MS")
    model = ExponentialSmoothing(y_train, trend="add", seasonal="mul", seasonal_periods=12).fit()
    return model.forecast(H).values


def forecast_desest(train_df, test_df):
    train = train_df.copy()
    train["Mes_num"] = train["Fecha"].dt.month

    prom_general = train["Value"].mean()
    prom_mes = train.groupby("Mes_num")["Value"].mean()
    indice_est = prom_mes / prom_general

    train["Indice"] = train["Mes_num"].map(indice_est)
    train["y_sa"] = train["Value"] / train["Indice"]

    t = np.arange(1, len(train) + 1)
    X = sm.add_constant(t)
    model_sa = sm.OLS(train["y_sa"].values, X).fit()

    H = len(test_df)
    t_fore = np.arange(len(train) + 1, len(train) + H + 1)
    X_fore = sm.add_constant(t_fore)
    y_sa_fore = model_sa.predict(X_fore)

    test2 = test_df.copy()
    test2["Mes_num"] = test2["Fecha"].dt.month
    test2["Indice"] = test2["Mes_num"].map(indice_est)

    return y_sa_fore * test2["Indice"].values


# ============================================================
# CÁLCULO DE PRONÓSTICOS (test)
# ============================================================
H = len(test)

# Originales
test["Forecast_Original"] = forecast_original_ols(train_test, H)

# PM multi (2..24) — igual al notebook
forecasts_pm = pm_forecasts_multi(train_test, H)
test["Forecast_MA"] = forecasts_pm[W_SHOW]     # para visualizar
test["Forecast_MA12"] = forecasts_pm[12]       # para tabla final (igual al notebook)

# Holt-Winters
test["Forecast_HW"] = forecast_hw(train_test, H)

# Desestacionalización
test["Forecast_Desest"] = forecast_desest(train_test, test)


# ============================================================
# RMSE
# ============================================================
def rmse(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((real - pred) ** 2)))

rmse_rows = [
    ("Datos originales", rmse(test["Value"].values, test["Forecast_Original"].values)),
    ("Promedio móvil centrado (w=12)", rmse(test["Value"].values, test["Forecast_MA12"].values)),  # igual al notebook
    ("Holt-Winters", rmse(test["Value"].values, test["Forecast_HW"].values)),
    ("Desestacionalización", rmse(test["Value"].values, test["Forecast_Desest"].values)),
]
rmse_table = pd.DataFrame(rmse_rows, columns=["Modelo", "RMSE"]).sort_values("RMSE")
best_model = rmse_table.iloc[0]["Modelo"]
best_rmse = rmse_table.iloc[0]["RMSE"]


# ============================================================
# KPI CARDS
# ============================================================
st.subheader("Resumen del modelo")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Mejor modelo (por RMSE)</div>
            <div class="metric">{best_model}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">RMSE (abr 2024 – mar 2025)</div>
            <div class="metric">{best_rmse:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="metric-label">Periodo evaluado</div>
            <div class="metric">Abr 2024 – Mar 2025</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 1) RMSE BAR CHART + CÓDIGO
# ============================================================
st.subheader("1) RMSE por modelo (menor es mejor)")

fig_rmse = go.Figure(go.Bar(x=rmse_table["Modelo"], y=rmse_table["RMSE"]))
fig_rmse.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020B16",
    paper_bgcolor="#020B16",
    font=dict(color="#E6EEF7"),
    title_font=dict(size=20, color="#C9A227"),
    height=420,
    yaxis_title="RMSE"
)
st.plotly_chart(fig_rmse, use_container_width=True)

with st.expander("📌 Ver código del gráfico: RMSE"):
    st.code("""
fig_rmse = go.Figure(go.Bar(x=rmse_table["Modelo"], y=rmse_table["RMSE"]))
fig_rmse.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020B16",
    paper_bgcolor="#020B16",
    font=dict(color="#E6EEF7"),
    title_font=dict(size=20, color="#C9A227"),
    height=420,
    yaxis_title="RMSE"
)
st.plotly_chart(fig_rmse, use_container_width=True)
""", language="python")


# ============================================================
# 2) EVALUACIÓN (REAL VS PRONÓSTICOS) + CÓDIGO
# ============================================================
st.subheader("2) Evaluación (abr-2024 → mar-2025): Real vs Pronósticos")

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_test["Fecha"], y=train_test["Value"], name="Train", mode="lines"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Value"], name="Real (test)", mode="lines+markers"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_Original"], name="Originales", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_MA"], name=f"PM (w={W_SHOW})", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_HW"], name="Holt-Winters", mode="lines+markers"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_Desest"], name="Desest.", line=dict(dash="dashdot")))
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020B16",
    paper_bgcolor="#020B16",
    font=dict(color="#E6EEF7"),
    title_font=dict(size=20, color="#C9A227"),
    height=520,
    yaxis_title="Remesas (Value)",
    xaxis_title="Fecha"
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📌 Ver código del gráfico: Evaluación (Real vs Pronósticos)"):
    st.code("""
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_test["Fecha"], y=train_test["Value"], name="Train", mode="lines"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Value"], name="Real (test)", mode="lines+markers"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_Original"], name="Originales", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_MA"], name=f"PM (w={W_SHOW})", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_HW"], name="Holt-Winters", mode="lines+markers"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_Desest"], name="Desest.", line=dict(dash="dashdot")))
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020B16",
    paper_bgcolor="#020B16",
    height=520
)
st.plotly_chart(fig, use_container_width=True)
""", language="python")


# ============================================================
# TABLAS POR MÉTODO (idénticas al notebook)
# ============================================================
st.subheader("Tablas por método (abr-2024 → mar-2025)")

with st.expander("Tabla — Datos originales"):
    tabla_originales = test[["Fecha", "Value", "Forecast_Original"]].copy()
    tabla_originales["Mes-Año"] = tabla_originales["Fecha"].dt.strftime("%b-%Y")
    tabla_originales = tabla_originales[["Mes-Año", "Value", "Forecast_Original"]].rename(
        columns={"Value": "Real", "Forecast_Original": "Pronostico"}
    )
    st.dataframe(tabla_originales, use_container_width=True)

with st.expander("Tabla — Promedios móviles (2..24)"):
    tabla_pm_multi = pm_table_multi(test, forecasts_pm)
    st.dataframe(tabla_pm_multi, use_container_width=True)

with st.expander("Tabla — Holt-Winters"):
    tabla_hw = test[["Fecha", "Value", "Forecast_HW"]].copy()
    tabla_hw["Mes-Año"] = tabla_hw["Fecha"].dt.strftime("%b-%Y")
    tabla_hw = tabla_hw[["Mes-Año", "Value", "Forecast_HW"]].rename(
        columns={"Value": "Real", "Forecast_HW": "Pronostico"}
    )
    st.dataframe(tabla_hw, use_container_width=True)

with st.expander("Tabla — Desestacionalización"):
    tabla_des = test[["Fecha", "Value", "Forecast_Desest"]].copy()
    tabla_des["Mes-Año"] = tabla_des["Fecha"].dt.strftime("%b-%Y")
    tabla_des = tabla_des[["Mes-Año", "Value", "Forecast_Desest"]].rename(
        columns={"Value": "Real", "Forecast_Desest": "Pronostico"}
    )
    st.dataframe(tabla_des, use_container_width=True)


# ============================================================
# 3) TABLA COMPARATIVA FINAL + HEATMAP (error %)
# ============================================================
st.subheader("3) Tabla comparativa (test) — heatmap por error % (igual al notebook)")

THRESHOLDS = {"green": 0.05, "ygreen": 0.10, "yellow": 0.20, "orange": 0.30}
COLORS = {
    "green": "#c6efce",
    "ygreen": "#e7f4b5",
    "yellow": "#fff2cc",
    "orange": "#f8cbad",
    "red": "#f4b6b6",
}

tabla = test[["Fecha", "Value", "Forecast_Original", "Forecast_MA12", "Forecast_HW", "Forecast_Desest"]].copy()
tabla["Mes-Año"] = tabla["Fecha"].dt.strftime("%b-%Y")
tabla = tabla[["Mes-Año", "Value", "Forecast_Original", "Forecast_MA12", "Forecast_HW", "Forecast_Desest"]].rename(columns={
    "Value": "Real",
    "Forecast_Original": "Datos_Originales",
    "Forecast_MA12": "Promedio_Movil",
    "Forecast_HW": "Suaviz_Exp",
    "Forecast_Desest": "Desestacionalizacion"
})

forecast_cols = ["Datos_Originales", "Promedio_Movil", "Suaviz_Exp", "Desestacionalizacion"]

def _color_por_error_pct(real, pred):
    if pd.isna(pred) or pd.isna(real) or real == 0:
        return ""
    err = abs((pred - real) / real)
    if err <= THRESHOLDS["green"]:
        return f"background-color: {COLORS['green']}"
    elif err <= THRESHOLDS["ygreen"]:
        return f"background-color: {COLORS['ygreen']}"
    elif err <= THRESHOLDS["yellow"]:
        return f"background-color: {COLORS['yellow']}"
    elif err <= THRESHOLDS["orange"]:
        return f"background-color: {COLORS['orange']}"
    else:
        return f"background-color: {COLORS['red']}"

def _aplicar_heatmap(df_):
    styles = pd.DataFrame("", index=df_.index, columns=df_.columns)
    for col in forecast_cols:
        styles[col] = [_color_por_error_pct(r, p) for r, p in zip(df_["Real"], df_[col])]
    return styles

styled = (
    tabla.style
    .hide(axis="index")
    .apply(_aplicar_heatmap, axis=None)
    .format("{:,.2f}", subset=["Real"] + forecast_cols)
)
st.dataframe(styled, use_container_width=True)


# ============================================================
# 4) PRONÓSTICO 2026 (abr-2025 → mar-2026) + CÓDIGO
# ============================================================
st.subheader("4) Pronóstico (abr-2025 → mar-2026) con el mejor modelo")

H2 = 12

# Re-entrenar con train_forecast (hasta mar-2025)
if best_model == "Holt-Winters":
    yhat_2026 = forecast_hw(train_forecast, H2)
elif "Datos originales" in best_model:
    yhat_2026 = forecast_original_ols(train_forecast, H2)
elif "Promedio móvil" in best_model:
    # si por alguna razón PM fuera el mejor: usar w=12 para consistencia con notebook
    yhat_2026 = pm_forecasts_multi(train_forecast, H2)[12]
else:
    tmp = pd.DataFrame({"Fecha": forecast_range})
    yhat_2026 = forecast_desest(train_forecast, tmp)

forecast_df = pd.DataFrame({"Fecha": forecast_range, "Pronóstico": yhat_2026})
forecast_df["Mes-Año"] = forecast_df["Fecha"].dt.strftime("%b-%Y")

fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=train_forecast["Fecha"], y=train_forecast["Value"], name="Histórico", mode="lines"))
fig_f.add_trace(go.Scatter(x=forecast_df["Fecha"], y=forecast_df["Pronóstico"], name=f"Pronóstico ({best_model})", mode="lines+markers"))
fig_f.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020B16",
    paper_bgcolor="#020B16",
    font=dict(color="#E6EEF7"),
    title_font=dict(size=20, color="#C9A227"),
    height=520,
    yaxis_title="Remesas (Value)",
    xaxis_title="Fecha"
)
st.plotly_chart(fig_f, use_container_width=True)

with st.expander("📌 Ver código del gráfico: Pronóstico 2026"):
    st.code("""
H2 = 12
# Re-entrenar con train_forecast (hasta mar-2025)
if best_model == "Holt-Winters":
    yhat_2026 = forecast_hw(train_forecast, H2)
elif "Datos originales" in best_model:
    yhat_2026 = forecast_original_ols(train_forecast, H2)
elif "Promedio móvil" in best_model:
    yhat_2026 = pm_forecasts_multi(train_forecast, H2)[12]
else:
    tmp = pd.DataFrame({"Fecha": forecast_range})
    yhat_2026 = forecast_desest(train_forecast, tmp)

forecast_df = pd.DataFrame({"Fecha": forecast_range, "Pronóstico": yhat_2026})
forecast_df["Mes-Año"] = forecast_df["Fecha"].dt.strftime("%b-%Y")

fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=train_forecast["Fecha"], y=train_forecast["Value"], name="Histórico", mode="lines"))
fig_f.add_trace(go.Scatter(x=forecast_df["Fecha"], y=forecast_df["Pronóstico"], name=f"Pronóstico ({best_model})", mode="lines+markers"))
fig_f.update_layout(template="plotly_dark", plot_bgcolor="#020B16", paper_bgcolor="#020B16")
st.plotly_chart(fig_f, use_container_width=True)
""", language="python")

st.dataframe(forecast_df[["Mes-Año", "Pronóstico"]], use_container_width=True)

