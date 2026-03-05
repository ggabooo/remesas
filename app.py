import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(
    page_title="Remesas Guatemala | Pronóstico",
    page_icon="🦇",
    layout="wide"
)

st.markdown("""
<style>

.main-title{
font-size:42px;
font-weight:800;
color:#E6EEF7;
}

.subtitle{
font-size:18px;
color:#A8B9CC;
margin-bottom:30px;
}

.card{
background: linear-gradient(145deg,#0A1F33,#061321);
padding:25px;
border-radius:15px;
border:1px solid rgba(201,162,39,0.35);
box-shadow:0px 8px 20px rgba(0,0,0,0.5);
}

.metric{
font-size:32px;
font-weight:700;
color:#C9A227;
}

.metric-label{
font-size:14px;
color:#A8B9CC;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.gold-card{border:1px solid #D4AF37;border-radius:16px;padding:16px;background:rgba(11,42,69,.65);}
.big{font-size:34px;font-weight:800;color:#E8EEF7;margin:0}
.small{font-size:14px;color:rgba(232,238,247,.85);margin-top:4px}
.kpi{font-size:26px;font-weight:800;color:#D4AF37}
hr{border:0;height:1px;background:rgba(212,175,55,.35)}
</style>
""", unsafe_allow_html=True)



st.sidebar.title("Controles")
W_SHOW = st.sidebar.selectbox("Ventana Promedio Móvil (par)", [2,4,6,8,10,12,14,16,18,20,22,24], index=5)
st.sidebar.caption("Paleta: azul oscuro + dorado")



@st.cache_data
def load_data(path="EDB.xlsx"):
    df = pd.read_excel(path, sheet_name="Dataset")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha")
    return df

df = load_data()


st.markdown(
'<p class="main-title">Pronóstico de Remesas – Guatemala</p>',
unsafe_allow_html=True
)
st.markdown(
'<p class="subtitle">Comparación de modelos básicos y proyección 2026</p>',
unsafe_allow_html=True
)
st.markdown('<p class="big">Econometría II — Ing. Héctor Francisco Galeros Juárez</p>', unsafe_allow_html=True)
st.markdown('<p class="small">Gabriel Estuardo Maas Ordóñez - 1183922</p>', unsafe_allow_html=True)
st.markdown('<p class="small">José Eduardo Yoc García - 1056823</p>', unsafe_allow_html=True)
st.markdown('<p class="small">María Isabel Puddy Díaz - 1043723</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

with st.expander("Ver dataset (primeras filas)"):
    st.dataframe(df.head(15), use_container_width=True)


test = df[(df["Fecha"] >= "2024-04-01") & (df["Fecha"] <= "2025-03-01")].copy()
train_test = df[df["Fecha"] <= "2024-03-01"].copy()

train_forecast = df[df["Fecha"] <= "2025-03-01"].copy()
forecast_range = pd.date_range("2025-04-01", periods=12, freq="MS")


def forecast_original_ols(train_df, H):
    y_train = train_df["Value"].values
    t_train = np.arange(1, len(y_train) + 1)
    X_train = sm.add_constant(t_train)
    model = sm.OLS(y_train, X_train).fit()

    t_fore = np.arange(len(y_train) + 1, len(y_train) + H + 1)
    X_fore = sm.add_constant(t_fore)
    y_fore = model.predict(X_fore)
    return y_fore


def forecast_ma_centered(train_df, H, w=12):
    y = train_df["Value"].reset_index(drop=True)
    ma = y.rolling(window=w).mean()
    ma_c = ma.rolling(window=2).mean()
    ma_valid = ma_c.dropna().reset_index(drop=True)

    t_ma = np.arange(1, len(ma_valid) + 1)
    X_ma = sm.add_constant(t_ma)
    model_ma = sm.OLS(ma_valid.values, X_ma).fit()

    t_fore = np.arange(len(ma_valid) + 1, len(ma_valid) + H + 1)
    X_fore = sm.add_constant(t_fore)
    return model_ma.predict(X_fore)


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
    y_hat = y_sa_fore * test2["Indice"].values
    return y_hat




H = len(test)

test["Forecast_Original"] = forecast_original_ols(train_test, H)
test["Forecast_MA"] = forecast_ma_centered(train_test, H, w=W_SHOW)
test["Forecast_HW"] = forecast_hw(train_test, H)
test["Forecast_Desest"] = forecast_desest(train_test, test)

def rmse(real, pred):
    return float(np.sqrt(np.mean((real - pred)**2)))

rmse_rows = [
    ("Datos originales", rmse(test["Value"].values, test["Forecast_Original"].values)),
    (f"Promedio móvil centrado (w={W_SHOW})", rmse(test["Value"].values, test["Forecast_MA"].values)),
    ("Holt-Winters", rmse(test["Value"].values, test["Forecast_HW"].values)),
    ("Desestacionalización", rmse(test["Value"].values, test["Forecast_Desest"].values)),
]
rmse_table = pd.DataFrame(rmse_rows, columns=["Modelo","RMSE"]).sort_values("RMSE")
best_model = rmse_table.iloc[0]["Modelo"]
best_rmse = rmse_table.iloc[0]["RMSE"]


st.subheader("Resumen del modelo")
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(
    f"""
    <div class="card">
    <div class="metric-label">Mejor modelo</div>
    <div class="metric">{best_model}</div>
    </div>
    """,
    unsafe_allow_html=True
    )

with col2:
    st.markdown(
    f"""
    <div class="card">
    <div class="metric-label">RMSE</div>
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



st.subheader("1) RMSE por modelo (menor es mejor)")

fig_rmse = go.Figure(go.Bar(x=rmse_table["Modelo"], y=rmse_table["RMSE"]))
fig_rmse.update_layout( template="plotly_dark", plot_bgcolor="#020B16", paper_bgcolor="#020B16", font=dict(color="#E6EEF7"), title_font=dict(size=20,color="#C9A227") )
st.plotly_chart(fig_rmse, use_container_width=True)



st.subheader("2) Evaluación (abr-2024 → mar-2025): Real vs Pronósticos")

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_test["Fecha"], y=train_test["Value"], name="Train", mode="lines"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Value"], name="Real (test)", mode="lines+markers"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_Original"], name="Originales", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_MA"], name=f"PM (w={W_SHOW})", line=dict(dash="dot")))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_HW"], name="Holt-Winters", mode="lines+markers"))
fig.add_trace(go.Scatter(x=test["Fecha"], y=test["Forecast_Desest"], name="Desest.", line=dict(dash="dashdot")))
fig.update_layout( template="plotly_dark", plot_bgcolor="#020B16", paper_bgcolor="#020B16", font=dict(color="#E6EEF7"), title_font=dict(size=20,color="#C9A227") )
st.plotly_chart(fig, use_container_width=True)



st.subheader("3) Tabla comparativa (test)")
tabla = test[["Fecha","Value","Forecast_Original","Forecast_MA","Forecast_HW","Forecast_Desest"]].copy()
tabla["Mes-Año"] = tabla["Fecha"].dt.strftime("%b-%Y")
tabla = tabla.rename(columns={
    "Value":"Real",
    "Forecast_Original":"Datos_Originales",
    "Forecast_MA":"Promedio_Movil",
    "Forecast_HW":"Holt_Winters",
    "Forecast_Desest":"Desestacionalizacion"
})
tabla = tabla[["Mes-Año","Real","Datos_Originales","Promedio_Movil","Holt_Winters","Desestacionalizacion"]]

st.dataframe(tabla, use_container_width=True)



st.subheader("4) Pronóstico (abr-2025 → mar-2026) con el mejor modelo")

# re-entrenar con train_forecast
H2 = 12

if best_model == "Holt-Winters":
    yhat_2026 = forecast_hw(train_forecast, H2)
elif "Datos originales" in best_model:
    yhat_2026 = forecast_original_ols(train_forecast, H2)
elif "Promedio móvil" in best_model:
    yhat_2026 = forecast_ma_centered(train_forecast, H2, w=W_SHOW)
else:
    # desest: usamos forecast_range como “test_df” artificial con fechas
    tmp = pd.DataFrame({"Fecha": forecast_range})
    yhat_2026 = forecast_desest(train_forecast, tmp)

forecast_df = pd.DataFrame({"Fecha": forecast_range, "Pronóstico": yhat_2026})
forecast_df["Mes-Año"] = forecast_df["Fecha"].dt.strftime("%b-%Y")

fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=train_forecast["Fecha"], y=train_forecast["Value"], name="Histórico", mode="lines"))
fig_f.add_trace(go.Scatter(x=forecast_df["Fecha"], y=forecast_df["Pronóstico"], name=f"Pronóstico ({best_model})", mode="lines+markers"))
fig_f.update_layout( template="plotly_dark", plot_bgcolor="#020B16", paper_bgcolor="#020B16", font=dict(color="#E6EEF7"), title_font=dict(size=20,color="#C9A227") )
st.plotly_chart(fig_f, use_container_width=True)

st.dataframe(forecast_df[["Mes-Año","Pronóstico"]], use_container_width=True)




