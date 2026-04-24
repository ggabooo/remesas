# ============================================================
#  app.py  -  Cointegracion de Remesas Guatemala
#  Econometria II  |  Proyecto 2
# ============================================================
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
from statsmodels.stats.stattools import durbin_watson
import streamlit as st

try:
    from arch.unitroot import PhillipsPerron
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# -- PAGE CONFIG ----------------------------------------------
st.set_page_config(
    page_title="Remesas GT - Cointegracion",
    page_icon="GT",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- PALETA ---------------------------------------------------
GOLD   = "#C9A227"
DARK   = "#020B16"
CARD   = "#0A1F33"
TEXT   = "#E6EEF7"
MUTED  = "#A8B9CC"
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"

# -- CSS ------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

[data-testid="stAppViewContainer"]  {{background:{DARK};}}
[data-testid="stSidebar"]           {{background:#040f1e;border-right:1px solid {GOLD}44;}}
[data-testid="stSidebar"] *         {{color:{TEXT} !important;}}
.block-container                     {{padding-top:1.8rem;padding-bottom:2rem;}}
h1,h2,h3                             {{font-family:'Playfair Display',serif !important;color:{TEXT};}}
p,li,span,div                        {{font-family:'DM Sans',sans-serif;}}

.kpi {{
    background:{CARD};border:1px solid {GOLD}55;border-radius:14px;
    padding:18px 12px;text-align:center;
}}
.kpi-v {{
    font-family:'Playfair Display',serif;font-size:28px;font-weight:800;
    color:{GOLD};margin:0;word-break:break-word;
}}
.kpi-l {{font-size:11px;color:{MUTED};margin-top:6px;text-transform:uppercase;letter-spacing:.07em;}}

.scard {{
    background:{CARD};border:1px solid {GOLD}44;border-radius:14px;
    padding:20px 24px;margin-bottom:14px;
}}
.scard-title {{font-family:'Playfair Display',serif;font-size:16px;color:{TEXT};margin-bottom:8px;}}
.scard-body  {{font-size:13px;color:{MUTED};line-height:1.7;}}

.verdict-ok {{
    background:linear-gradient(135deg,#0d2b1d 0%,{CARD} 100%);
    border:2px solid {GREEN};border-radius:14px;
    padding:28px 24px;text-align:center;
}}
.verdict-warn {{
    background:linear-gradient(135deg,#2b1d0d 0%,{CARD} 100%);
    border:2px solid {ORANGE};border-radius:14px;
    padding:28px 24px;text-align:center;
}}
.verdict-v   {{font-family:'Playfair Display',serif;font-size:24px;font-weight:800;color:{TEXT};}}
.verdict-sub {{font-size:13px;color:{MUTED};margin-top:10px;line-height:1.7;}}

.insight {{
    background:linear-gradient(135deg,#0d1f3d 0%,{CARD} 100%);
    border-left:4px solid {GOLD};border-radius:0 10px 10px 0;
    padding:14px 18px;margin:16px 0;
}}
.insight p {{font-size:13px;color:{TEXT};margin:0;line-height:1.7;}}

.divider {{border:0;height:1px;background:{GOLD}44;margin:18px 0;}}
</style>
""", unsafe_allow_html=True)


# -- PLOTLY HELPER --------------------------------------------
def layout(**kw):
    """Return a clean plotly layout dict with project defaults."""
    # Extract title string to wrap in proper dict
    title_val = kw.pop("title", "")
    base = dict(
        template="plotly_dark",
        plot_bgcolor=DARK,
        paper_bgcolor=DARK,
        font=dict(color=TEXT, family="sans-serif"),
        title=dict(text=title_val, font=dict(size=15, color=GOLD)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(201,162,39,0.2)"),
        margin=dict(t=50, b=42, l=50, r=18),
    )
    base.update(kw)
    return base


# -- DATA -----------------------------------------------------
RENAME = {
    "PERIODO":"Periodo","REMESAS-GT":"REMESAS_GT_MUSD","PIB-GT":"PIB_GT_MQUETZ",
    "DESEMPLEO-GT":"DESEMPLEO_GT_PCT","GASOLINA-GT":"GASOLINA_GT_QTQGAL",
    "TC-GT":"TC_GT_QTQUSD","WTI-USA":"PETROLEO_WTI_USD","WIT-USA":"PETROLEO_WTI_USD",
    "DESEMPLEO-USA":"DESEMPLEO_EEUU_PCT","PIB-USA":"PIB_EEUU_BUSD",
}
LABELS = {
    "REMESAS_GT_MUSD":   "Remesas GT (MUSD)",
    "PIB_GT_MQUETZ":     "PIB Guatemala (MQuetz)",
    "DESEMPLEO_GT_PCT":  "Desempleo Guatemala (%)",
    "GASOLINA_GT_QTQGAL":"Gasolina GT (Q/gal)",
    "TC_GT_QTQUSD":      "Tipo de Cambio (Q/USD)",
    "PETROLEO_WTI_USD":  "Petroleo WTI (USD)",
    "DESEMPLEO_EEUU_PCT":"Desempleo EE.UU. (%)",
    "PIB_EEUU_BUSD":     "PIB EE.UU. (BUSD)",
}

@st.cache_data
def load_data(path="EDB-Proyecto2.xlsx"):
    df_raw = pd.read_excel(path, sheet_name="Datos")
    ar = {}
    for col in df_raw.columns:
        for k, v in RENAME.items():
            if col.strip().upper() == k.upper():
                ar[col] = v; break
    df = df_raw.rename(columns=ar)
    df["idx"] = df["Periodo"].apply(
        lambda x: pd.Period(
            f"{x.split('-')[1]}Q{x.split('-')[0].replace('Q','')}", freq="Q"))
    df = df.set_index("idx").sort_index().drop(columns=["Periodo"], errors="ignore")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.loc[:pd.Period("2024Q3", freq="Q")]


@st.cache_data
def build_z(df):
    log_c  = [c for c in ["REMESAS_GT_MUSD","PIB_GT_MQUETZ","GASOLINA_GT_QTQGAL",
                           "TC_GT_QTQUSD","PETROLEO_WTI_USD","PIB_EEUU_BUSD"]
              if c in df.columns]
    rate_c = [c for c in ["DESEMPLEO_GT_PCT","DESEMPLEO_EEUU_PCT"] if c in df.columns]
    z = df.copy()
    for c in log_c:  z[f"L_{c}"] = np.log(z[c])
    for c in rate_c: z[f"L_{c}"] = z[c]
    for c in log_c + rate_c: z[f"D_L_{c}"] = z[f"L_{c}"].diff()
    return z


def _integ(series, trend=False):
    s   = series.dropna()
    reg = "ct" if trend else "c"
    adf_p  = adfuller(s, regression=reg, autolag="AIC")[1]
    pp_p   = PhillipsPerron(s, trend=reg).pvalue if HAS_ARCH else np.nan
    try:   kpss_p = kpss(s, regression=reg, nlags="auto")[1]
    except: kpss_p = np.nan
    adf_d  = adfuller(s.diff().dropna(), regression="c", autolag="AIC")[1]
    pp_d   = PhillipsPerron(s.diff().dropna(), trend="c").pvalue if HAS_ARCH else np.nan
    try:   kpss_d = kpss(s.diff().dropna(), regression="c", nlags="auto")[1]
    except: kpss_d = np.nan
    lv = sum([adf_p<0.05,
              not np.isnan(pp_p)   and pp_p<0.05,
              not np.isnan(kpss_p) and kpss_p>0.05])
    dv = sum([adf_d<0.05,
              not np.isnan(pp_d)   and pp_d<0.05,
              not np.isnan(kpss_d) and kpss_d>0.05])
    orden = "I(0)" if lv >= 2 else ("I(1)" if dv >= 2 else "N/C")
    return (orden,
            round(adf_p, 4),
            round(pp_p,  4) if not np.isnan(pp_p)   else np.nan,
            round(kpss_p,4) if not np.isnan(kpss_p) else np.nan)


@st.cache_data
def run_unit_root(z):
    l_cols = [c for c in z.columns
              if c.startswith("L_") and not c.startswith("D_L_")]
    rows = []
    for v in l_cols:
        trend = any(k in v for k in ["REMESAS","PIB"])
        orden, adf_p, pp_p, kpss_p = _integ(z[v], trend)
        rows.append({"Serie": v, "ADF": adf_p, "PP": pp_p,
                     "KPSS": kpss_p, "Orden": orden})
    return pd.DataFrame(rows)


@st.cache_data
def run_coint(z, _ur_tbl):
    dep  = "L_REMESAS_GT_MUSD"
    i1   = _ur_tbl.loc[_ur_tbl["Orden"] == "I(1)", "Serie"].tolist()
    ind  = [v for v in i1 if v != dep]
    df_r = z[[dep] + ind].dropna()
    Y    = df_r[dep]
    X    = sm.add_constant(df_r[ind])
    ols  = sm.OLS(Y, X).fit()
    res  = ols.resid
    adf_r = adfuller(res, regression="n", autolag="AIC")
    dw_r  = durbin_watson(res)

    pairs = []
    for xv in ind:
        pdf = z[[dep, xv]].dropna()
        m   = sm.OLS(pdf[dep], sm.add_constant(pdf[xv])).fit()
        r   = m.resid
        ap  = adfuller(r, regression="n", autolag="AIC")
        dw  = durbin_watson(r)
        lab = LABELS.get(xv.replace("L_",""), xv.replace("L_",""))
        pairs.append({
            "Variable":       lab,
            "Beta":           round(m.params[xv], 4),
            "R2":             round(m.rsquared, 4),
            "DW":             round(dw, 4),
            "ADF p-valor":    round(ap[1], 4),
            "Cointegrados":   "SI" if ap[1] < 0.05 else "NO",
        })
    return ols, res, adf_r, dw_r, pd.DataFrame(pairs), df_r, ind


def ts_idx(df):
    return df.index.to_timestamp(how="end")


def acf_pacf_fig(series, label="", nlags=16):
    s  = series.dropna().values
    ci = 1.96 / np.sqrt(len(s))
    av = sm_acf(s,  nlags=nlags)
    pv = sm_pacf(s, nlags=nlags, method="ywm")
    lags = list(range(len(av)))

    fig = make_subplots(1, 2, subplot_titles=["ACF", "PACF"])
    for lg, v in zip(lags, av):
        clr = GOLD if (abs(v) > ci and lg > 0) else "rgba(37,99,235,0.55)"
        fig.add_trace(go.Bar(x=[lg], y=[v], marker_color=clr,
                             showlegend=False), row=1, col=1)
    for lg, v in zip(lags, pv):
        clr = GOLD if (abs(v) > ci and lg > 0) else "rgba(37,99,235,0.55)"
        fig.add_trace(go.Bar(x=[lg], y=[v], marker_color=clr,
                             showlegend=False), row=1, col=2)
    for col in [1, 2]:
        fig.add_hline(y= ci, line_dash="dash", line_color=RED,  line_width=1, row=1, col=col)
        fig.add_hline(y=-ci, line_dash="dash", line_color=RED,  line_width=1, row=1, col=col)
        fig.add_hline(y=  0, line_color="white", line_width=0.4, row=1, col=col)
    fig.update_layout(**layout(
        title=f"ACF / PACF  -  {label}",
        height=320,
        bargap=0.15,
    ))
    return fig


# -- PAGES ----------------------------------------------------


def q_banner(num, question, companion=""):
    color_map = {1: "#60A5FA", 2: "#F87171", 3: "#34D399", 4: GOLD}
    clr = color_map.get(num, GOLD)
    companion_html = (
        f'<div style="font-size:11px;color:{clr};margin-top:8px;">'
        f'Complementar con: {companion}</div>'
    ) if companion else ""
    st.markdown(
        f'''<div style="background:#0d1f3d;border-left:4px solid {clr};
            border-radius:0 12px 12px 0;padding:16px 22px;margin-bottom:22px;">
          <div style="font-size:10px;color:{MUTED};text-transform:uppercase;
                      letter-spacing:.1em;margin-bottom:8px;">
            Pregunta de Investigacion {num} de 4</div>
          <div style="font-size:14px;color:{TEXT};line-height:1.7;font-style:italic;">
            &ldquo;{question}&rdquo;</div>
          {companion_html}
        </div>''',
        unsafe_allow_html=True,
    )

def pg_inicio(df):
    st.markdown(f"""
    <h1 style='font-family:Playfair Display,serif;font-size:44px;
        background:linear-gradient(90deg,{TEXT},{GOLD});
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:4px;'>
        Cointegracion de Remesas
    </h1>
    <p style='font-size:19px;color:{MUTED};margin-bottom:4px;'>
        Guatemala - Analisis econometrico 2004 - 2024
    </p>
    <p style='font-size:14px;color:{GOLD};'>
        Econometria II  |  Ing. Hector Francisco Galeros Juarez
    </p>
    <p style='font-size:12px;color:{MUTED};'>
        Gabriel Estuardo Maas Ordonez &nbsp;|&nbsp;
        Jose Eduardo Yoc Garcia &nbsp;|&nbsp;
        Maria Isabel Puddy Diaz
    </p>
    <hr class='divider'>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (str(len(df)),   "Observaciones"),
        ("8",            "Variables"),
        ("2004-2024",    "Periodo"),
        ("I(1)",         "Orden de integracion"),
    ]
    for col, (v, l) in zip([c1,c2,c3,c4], kpis):
        col.markdown(
            f'<div class="kpi"><div class="kpi-v">{v}</div>'
            f'<div class="kpi-l">{l}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Hero chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_idx(df), y=df["REMESAS_GT_MUSD"],
        mode="lines", name="Remesas GT",
        line=dict(color=GOLD, width=2.5),
        fill="tozeroy", fillcolor="rgba(201,162,39,0.10)",
    ))
    fig.update_layout(**layout(
        title="Remesas de Guatemala - Serie Historica (millones de USD)",
        height=380,
        xaxis=dict(title="Trimestre"),
        yaxis=dict(title="Millones USD"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="scard">
      <div class="scard-title">Que analiza este estudio?</div>
      <div class="scard-body">
        Las remesas de Guatemala crecieron de forma sostenida desde 2004.
        Este proyecto evalua si esa trayectoria esta
        <strong style='color:{GOLD}'>vinculada de forma estable a largo plazo</strong>
        con variables macroeconomicas clave de Guatemala y de Estados Unidos,
        aplicando el enfoque de <strong style='color:{GOLD}'>cointegracion de Engle-Granger</strong>:
        pruebas de raiz unitaria (ADF, PP, KPSS), diagnostico de regresion espuria
        y test ADF sobre residuales del modelo de regresion multiple.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("Hipotesis",
         "Existe una relacion de equilibrio de largo plazo entre las remesas y las variables macroeconomicas analizadas?"),
        ("Metodologia",
         "Pruebas ADF, PP, KPSS  ->  Regresion espuria  ->  Engle-Granger  ->  Analisis por pares"),
        ("Resultado principal",
         "Los residuales del modelo multiple son estacionarios (ADF p < 0.001): cointegracion confirmada."),
    ]
    for col, (title, body) in zip([col1, col2, col3], cards):
        col.markdown(f"""
        <div class="scard">
          <div class="scard-title">{title}</div>
          <div class="scard-body">{body}</div>
        </div>""", unsafe_allow_html=True)


def pg_datos(df):
    st.markdown("<h2>Exploracion de las Series</h2>", unsafe_allow_html=True)
    q_banner(1, "Que tipo de procesos estocasticos caracterizan el comportamiento de las series de tiempo y que implicaciones tiene esta clasificacion para su modelacion econometrica?", "Estacionariedad")
    st.markdown(f'<p style="color:{MUTED}">Comportamiento de las 8 variables en el periodo 2004-2024</p>',
                unsafe_allow_html=True)

    cols_list = list(df.columns)
    clrs = [GOLD,"#60A5FA","#34D399","#F87171","#A78BFA","#FBBF24","#38BDF8","#FB923C"]
    fig = make_subplots(rows=4, cols=2,
                        subplot_titles=[LABELS.get(c, c) for c in cols_list],
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    for i, (col, clr) in enumerate(zip(cols_list, clrs)):
        r, c = divmod(i, 2); r += 1; c += 1
        fig.add_trace(go.Scatter(x=ts_idx(df), y=df[col], mode="lines",
                                 line=dict(color=clr, width=1.8),
                                 showlegend=False), row=r, col=c)
    fig.update_layout(**layout(height=900, title="Series de tiempo - Niveles"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h3>Tendencia lineal simple (OLS vs tiempo)</h3>", unsafe_allow_html=True)

    rows = []
    for c in df.columns:
        y = df[c].dropna().values
        t = np.arange(1, len(y)+1)
        m = sm.OLS(y, sm.add_constant(t)).fit()
        rows.append({
            "Variable":    LABELS.get(c, c),
            "Pendiente":   round(m.params[1], 4),
            "p-valor":     round(m.pvalues[1], 4),
            "R2":          round(m.rsquared, 4),
            "Significativa": "Si" if m.pvalues[1] < 0.05 else "No",
        })
    tdf = pd.DataFrame(rows)

    fig2 = go.Figure(go.Bar(
        x=tdf["Variable"], y=tdf["R2"],
        marker_color=[GOLD if r > 0.5 else BLUE for r in tdf["R2"]],
        text=[f"{v:.3f}" for v in tdf["R2"]], textposition="outside",
    ))
    fig2.update_layout(**layout(
        title="R2 de tendencia lineal por serie",
        height=360,
        yaxis=dict(range=[0, 1.1], title="R2"),
    ))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight"><p>
    Las series con R2 alto (Remesas, PIB Guatemala, PIB EE.UU.) presentan
    <strong>tendencia deterministica clara</strong>, lo que justifica incluirla
    como especificacion en las pruebas de raiz unitaria.
    Gasolina, Tipo de Cambio y Petroleo no exhiben tendencia lineal significativa.
    </p></div>""", unsafe_allow_html=True)


def pg_estacion(z, ur_tbl):
    st.markdown("<h2>Estacionariedad</h2>", unsafe_allow_html=True)
    q_banner(1, "Que tipo de procesos estocasticos caracterizan el comportamiento de las series de tiempo y que implicaciones tiene esta clasificacion para su modelacion econometrica?", "Los Datos")
    q_banner(3, "Las series de remesas, PIB, desempleo, tipo de cambio, petroleo y gasolina presentan raiz unitaria, y que transformaciones son necesarias para convertirlas en estacionarias y permitir su correcta modelacion econometrica?", "Regresion Espuria")

    l_cols = [c for c in z.columns
              if c.startswith("L_") and not c.startswith("D_L_")]
    var_sel = st.selectbox(
        "Selecciona una variable para ver su ACF / PACF:",
        l_cols,
        format_func=lambda x: LABELS.get(x.replace("L_",""), x),
    )
    fig = acf_pacf_fig(z[var_sel], label=LABELS.get(var_sel.replace("L_",""), var_sel))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight"><p>
    Un <strong>decaimiento lento y persistente del ACF</strong> (barras significativas
    en muchos rezagos) es la firma visual de una serie con
    <strong>raiz unitaria (no estacionaria)</strong>.
    Un corte abrupto despues del lag 1 sugiere proceso estacionario o MA(1).
    </p></div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h3>Resultados de Raiz Unitaria (ADF, PP, KPSS)</h3>", unsafe_allow_html=True)

    disp = ur_tbl.copy()
    disp["Variable"] = disp["Serie"].apply(
        lambda x: LABELS.get(x.replace("L_",""), x.replace("L_","")))
    disp = disp[["Variable","ADF","PP","KPSS","Orden"]]

    fig2 = go.Figure()
    cols_info = [("ADF", GOLD), ("PP", BLUE), ("KPSS", "#34D399")]
    for col_name, clr in cols_info:
        vals = pd.to_numeric(disp[col_name], errors="coerce").fillna(0.5)
        fig2.add_trace(go.Bar(
            name=col_name, x=disp["Variable"], y=vals,
            marker_color=clr, opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10),
        ))
    fig2.add_hline(y=0.05, line_dash="dash", line_color=RED, line_width=2,
                   annotation_text="alfa = 0.05", annotation_font_color=RED)
    fig2.update_layout(**layout(
        title="p-valores de las pruebas de raiz unitaria en NIVELES",
        barmode="group",
        height=420,
        yaxis=dict(range=[0, 1.1], title="p-valor"),
        legend=dict(orientation="h", y=1.12),
    ))
    st.plotly_chart(fig2, use_container_width=True)

    i1 = (ur_tbl["Orden"] == "I(1)").sum()
    i0 = (ur_tbl["Orden"] == "I(0)").sum()
    c1, c2, c3 = st.columns(3)
    for col, (v, l) in zip([c1,c2,c3],[
        (str(i1), "Series I(1)  -  no estacionarias"),
        (str(i0), "Series I(0)  -  estacionarias"),
        ("I(1)",  "Orden predominante"),
    ]):
        col.markdown(
            f'<div class="kpi"><div class="kpi-v">{v}</div>'
            f'<div class="kpi-l">{l}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(
        disp.rename(columns={"ADF":"ADF p-val","PP":"PP p-val","KPSS":"KPSS p-val"}),
        use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight"><p>
    <strong>7 de 8 series resultan I(1)</strong>: raiz unitaria en niveles,
    estacionarias en primeras diferencias.
    El Petroleo WTI es I(0) y queda excluido del analisis de cointegracion
    (no se puede cointegrar series de distinto orden de integracion).
    </p></div>""", unsafe_allow_html=True)


def pg_comparacion(df):
    st.markdown("<h2>Remesas vs Variables</h2>", unsafe_allow_html=True)

    otras   = [c for c in df.columns if c != "REMESAS_GT_MUSD"]
    var_sel = st.selectbox(
        "Comparar Remesas con:",
        otras, format_func=lambda x: LABELS.get(x, x))

    paired = df[["REMESAS_GT_MUSD", var_sel]].dropna()
    corr   = paired.corr().iloc[0, 1]

    c1, c2 = st.columns(2)

    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=ts_idx(df), y=df["REMESAS_GT_MUSD"],
            name="Remesas GT", line=dict(color=GOLD, width=2)),
            secondary_y=False)
        fig.add_trace(go.Scatter(
            x=ts_idx(df), y=df[var_sel],
            name=LABELS.get(var_sel, var_sel),
            line=dict(color=BLUE, width=2, dash="dot")),
            secondary_y=True)
        fig.update_layout(**layout(
            title="Series en el tiempo (doble eje)", height=360))
        fig.update_yaxes(title_text="Remesas (MUSD)",        secondary_y=False, showgrid=True,  gridcolor="rgba(201,162,39,0.13)")
        fig.update_yaxes(title_text=LABELS.get(var_sel,""),  secondary_y=True,  showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        xv  = paired[var_sel].values
        yv  = paired["REMESAS_GT_MUSD"].values
        p   = np.polyfit(xv, yv, 1)
        xl  = np.linspace(xv.min(), xv.max(), 200)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xv, y=yv, mode="markers",
            marker=dict(color=GOLD, size=6, opacity=0.7), name="Datos"))
        fig2.add_trace(go.Scatter(
            x=xl, y=np.polyval(p, xl), mode="lines",
            line=dict(color=BLUE, width=2), name="Tendencia"))
        fig2.update_layout(**layout(
            title=f"Dispersion  |  r = {corr:.3f}",
            height=360,
            xaxis=dict(title=LABELS.get(var_sel, var_sel)),
            yaxis=dict(title="Remesas GT (MUSD)"),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    strength  = ("muy fuerte" if abs(corr) > 0.85
                 else ("fuerte" if abs(corr) > 0.60 else "moderada o debil"))
    direction = "positiva" if corr > 0 else "negativa"
    st.markdown(f"""
    <div class="scard">
      <div class="scard-title">Correlacion de Pearson: {corr:.3f}</div>
      <div class="scard-body">
        Existe una correlacion <strong style='color:{GOLD}'>{strength} y {direction}</strong>
        entre las Remesas y <strong>{LABELS.get(var_sel, var_sel)}</strong>.
        Sin embargo, una correlacion alta entre series no estacionarias puede ser
        resultado de una <em>regresion espuria</em> - por eso se requieren pruebas
        formales de cointegracion.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h3>Correlaciones con Remesas - todas las variables</h3>", unsafe_allow_html=True)

    corrs  = {LABELS.get(c, c): df[["REMESAS_GT_MUSD", c]].dropna().corr().iloc[0,1]
              for c in otras}
    corr_s = pd.Series(corrs).sort_values(ascending=True)
    fig3 = go.Figure(go.Bar(
        x=corr_s.values, y=corr_s.index, orientation="h",
        marker_color=[GOLD if v > 0 else RED for v in corr_s.values],
        text=[f"{v:.3f}" for v in corr_s.values], textposition="outside",
    ))
    fig3.add_vline(x=0, line_color="white", line_width=1)
    fig3.update_layout(**layout(
        title="Correlacion de Pearson con Remesas GT",
        height=380,
        xaxis=dict(range=[-1.1, 1.1]),
    ))
    st.plotly_chart(fig3, use_container_width=True)


def pg_espuria(z):
    st.markdown("<h2>Diagnostico de Regresion Espuria</h2>", unsafe_allow_html=True)
    q_banner(2, "En que medida la estimacion de relaciones entre remesas y las variables macroeconomicas consideradas puede conducir a problemas de regresion espuria, y como se puede validar la estacionariedad de las series para evitar inferencias incorrectas?", "Cointegracion")
    st.markdown(
        f'<p style="color:{MUTED}">Regresar series I(1) sin verificar cointegracion '
        f'puede producir resultados aparentemente significativos pero falsos.</p>',
        unsafe_allow_html=True)

    dep   = "L_REMESAS_GT_MUSD"
    all_l = [c for c in z.columns
             if c.startswith("L_") and not c.startswith("D_L_") and c != dep]
    df_r  = z[[dep] + all_l].dropna()
    ols_e = sm.OLS(df_r[dep], sm.add_constant(df_r[all_l])).fit()
    res   = ols_e.resid
    dw    = durbin_watson(res)
    r2    = ols_e.rsquared
    adf_p = adfuller(res, regression="c", autolag="AIC")[1]

    c1, c2, c3, c4 = st.columns(4)
    for col, (v, l, warn) in zip([c1,c2,c3,c4],[
        (f"{r2:.4f}",                   "R2 del modelo en niveles",      False),
        (f"{dw:.3f}",                   "Durbin-Watson",                  dw < 1.5),
        (f"{adf_p:.4f}",               "ADF p-val (residuales)",          False),
        ("Posible espuria" if dw<1.5 else "Sin senal", "R2 alto + DW bajo", dw<1.5),
    ]):
        color = ORANGE if warn else GOLD
        col.markdown(
            f'<div class="kpi">'
            f'<div class="kpi-v" style="color:{color};font-size:22px;">{v}</div>'
            f'<div class="kpi-l">{l}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Real vs Ajustado: muestra el R2 sospechosamente alto
        fitted = ols_e.fittedvalues
        tx = df_r.index.to_timestamp(how="end")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tx, y=df_r[dep],
            mode="lines", line=dict(color=GOLD, width=2), name="Real"))
        fig.add_trace(go.Scatter(
            x=tx, y=fitted,
            mode="lines", line=dict(color=ORANGE, width=2, dash="dot"),
            name="Ajustado"))
        fig.update_layout(**layout(
            title=f"Real vs Ajustado  (R2 = {r2:.4f} sospechosamente alto)",
            height=330,
            legend=dict(orientation="h", y=1.1),
        ))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # ACF de residuales: muestra la autocorrelacion que delata la regresion espuria
        n  = len(res)
        ci = 1.96 / np.sqrt(n)
        av = sm_acf(res.values, nlags=16)
        fig2 = go.Figure(go.Bar(
            x=list(range(len(av))), y=av,
            marker_color=[ORANGE if (abs(v) > ci and i > 0) else "rgba(37,99,235,0.55)"
                          for i, v in enumerate(av)],
        ))
        fig2.add_hline(y= ci, line_dash="dash", line_color=RED, line_width=1)
        fig2.add_hline(y=-ci, line_dash="dash", line_color=RED, line_width=1)
        fig2.update_layout(**layout(
            title=f"ACF de residuales  (DW = {dw:.3f} autocorrelacion persistente)",
            height=330, bargap=0.15))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div class="verdict-warn">
      <div class="verdict-v">Senal de Alarma Detectada</div>
      <div class="verdict-sub">
        R2 = {r2:.4f} (muy alto) combinado con DW = {dw:.3f} (menor a 1.5)
        y autocorrelacion persistente en residuales indica que la regresion en niveles
        <strong>puede ser espuria</strong>. Esto no invalida el analisis -
        <strong>justifica proceder al test formal de cointegracion</strong>.
      </div>
    </div>""", unsafe_allow_html=True)


def pg_coint(z, ur_tbl):
    ols, res, adf_r, dw_r, pair_tbl, df_reg, ind_vars = run_coint(z, ur_tbl)

    st.markdown("<h2>Cointegracion - Engle-Granger</h2>", unsafe_allow_html=True)
    q_banner(4, "Existe una relacion de equilibrio de largo plazo (cointegracion) entre las remesas en Guatemala y las variables macroeconomicas internas, externas y globales, y como se puede modelar esta relacion mediante un enfoque multivariado?", "Regresion Espuria + Conclusiones")

    # 6.1 Coefficients
    st.markdown("<h3>6.1  Modelo de Regresion Multiple (niveles)</h3>", unsafe_allow_html=True)

    params = ols.params.drop("const")
    pvals  = ols.pvalues.drop("const")
    short  = [LABELS.get(v.replace("L_",""), v.replace("L_","")) for v in params.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=params.values, y=short, orientation="h",
        marker_color=[GREEN if p < 0.05 else "rgba(220,38,38,0.6)" for p in pvals],
        text=[f"b={v:.3f}  p={p:.3f}" for v, p in zip(params.values, pvals)],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(**layout(
        title="Coeficientes del modelo  -  Verde = significativo (p<0.05)",
        height=360,
        xaxis=dict(title="Valor del coeficiente"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, (v, l) in zip([c1,c2,c3],[
        (f"{ols.rsquared:.4f}",  "R2"),
        (f"{dw_r:.3f}",          "Durbin-Watson"),
        (str(len(df_reg)),       "Observaciones"),
    ]):
        col.markdown(
            f'<div class="kpi"><div class="kpi-v">{v}</div>'
            f'<div class="kpi-l">{l}</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br><hr class='divider'>", unsafe_allow_html=True)

    # 6.2 ADF on residuals
    st.markdown("<h3>6.2  Test ADF sobre Residuales del Modelo</h3>", unsafe_allow_html=True)

    adf_stat = adf_r[0]; adf_p = adf_r[1]; adf_cv = adf_r[4]
    ok = adf_p < 0.05

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_reg.index.to_timestamp(how="end"), y=res,
            mode="lines", line=dict(color=GREEN if ok else ORANGE, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(22,163,74,0.08)" if ok else "rgba(234,88,12,0.08)",
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
        fig2.update_layout(**layout(
            title="Residuales de la regresion de cointegracion", height=300))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        box  = "verdict-ok" if ok else "verdict-warn"
        msg  = "COINTEGRACION CONFIRMADA" if ok else "NO SE CONFIRMA COINTEGRACION"
        det  = ("Los residuales son <strong>estacionarios</strong>. La relacion entre remesas "
                "y las variables independientes es de equilibrio de largo plazo real, no espuria."
                if ok else
                "Los residuales tienen raiz unitaria. La relacion podria ser espuria.")
        st.markdown(f"""
        <div class="{box}">
          <div class="verdict-v">{msg}</div>
          <div class="verdict-sub">
            ADF estadistico: <strong>{adf_stat:.4f}</strong> &nbsp;|&nbsp;
            p-valor: <strong>{adf_p:.4f}</strong><br>
            Valor critico 1%: {adf_cv["1%"]:.4f} &nbsp;
            5%: {adf_cv["5%"]:.4f} &nbsp;
            10%: {adf_cv["10%"]:.4f}<br><br>
            {det}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # 6.3 Pairwise
    st.markdown("<h3>6.3  Analisis por Pares</h3>", unsafe_allow_html=True)

    fig3 = go.Figure(go.Bar(
        x=pair_tbl["Variable"], y=pair_tbl["ADF p-valor"],
        marker_color=[GREEN if v < 0.05 else "rgba(220,38,38,0.7)"
                      for v in pair_tbl["ADF p-valor"]],
        text=[f"{v:.3f}" for v in pair_tbl["ADF p-valor"]], textposition="outside",
    ))
    fig3.add_hline(y=0.05, line_dash="dash", line_color=GOLD, line_width=2,
                   annotation_text="umbral a=0.05", annotation_font_color=GOLD)
    fig3.update_layout(**layout(
        title="ADF p-valor de residuales por par  -  Verde = cointegrado",
        height=370,
        yaxis=dict(range=[0, 0.75], title="p-valor ADF"),
    ))
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(pair_tbl, use_container_width=True, hide_index=True)

    coint_vars = pair_tbl.loc[pair_tbl["Cointegrados"] == "SI", "Variable"].tolist()
    st.markdown(f"""
    <div class="insight"><p>
    Del analisis por pares, <strong style='color:{GOLD}'>{len(coint_vars)} variable(s)
    presentan cointegracion bilateral con Remesas</strong>:
    <strong>{', '.join(coint_vars) if coint_vars else 'ninguna'}</strong>.
    El vinculo bilateral mas solido es con el <strong>PIB de EE.UU.</strong>
    El modelo multiple confirma cointegracion conjunta: la relacion de equilibrio
    emerge del <em>conjunto</em> de variables, no de cada par de forma aislada.
    </p></div>""", unsafe_allow_html=True)


def pg_conclusiones():
    st.markdown("<h2>Conclusiones</h2>", unsafe_allow_html=True)
    # Research question map
    gold = GOLD; muted = MUTED; txt = TEXT
    st.markdown(
        f'<div style="background:#0d1f3d;border:1px solid {GOLD};border-radius:12px;padding:18px 24px;margin-bottom:24px;">'
        f'<div style="font-size:12px;color:{MUTED};text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;">Mapa de Preguntas de Investigacion</div>'
        f'<div style="font-size:13px;color:{TEXT};line-height:2.2;">'
        '<span style="color:#60A5FA;">P1</span> Procesos estocasticos &rarr; <strong>Los Datos + Estacionariedad</strong><br>'
        '<span style="color:#F87171;">P2</span> Regresion espuria &rarr; <strong>Regresion Espuria</strong><br>'
        '<span style="color:#34D399;">P3</span> Raiz unitaria y transformaciones &rarr; <strong>Estacionariedad</strong><br>'
        f'<span style="color:{GOLD};">P4</span> Cointegracion de largo plazo &rarr; <strong>Cointegracion</strong>'
        '</div></div>',
        unsafe_allow_html=True)


    hallazgos = [
        ("Series No Estacionarias",
         "7 de 8 variables son I(1). Confirman raiz unitaria en niveles mediante ADF, PP y KPSS, pero son estacionarias en primeras diferencias. El Petroleo WTI es I(0)."),
        ("Senal de Regresion Espuria",
         "El modelo en niveles produce R2 = 0.986 y DW = 1.12. El DW bajo indica autocorrelacion en residuales, advirtiendo sobre posible regresion espuria sin verificacion adicional."),
        ("Cointegracion Confirmada",
         "ADF sobre residuales del modelo multiple: estadistico -3.35, p-valor = 0.0008. Los residuales son estacionarios, existe una relacion de equilibrio de largo plazo real."),
        ("PIB de EE.UU. como Factor Clave",
         "Es la unica variable individualmente cointegrada con Remesas (p=0.027). Refleja que el nivel de actividad economica en EE.UU. es el determinante principal del flujo de remesas."),
        ("Implicacion Econometrica",
         "La cointegracion implica que las series convergen a un equilibrio de largo plazo. Esto valida el uso de un Modelo de Correccion de Error (VECM) para analisis dinamico."),
        ("Sobre Multicolinealidad",
         "El condition number = 6,730 senala multicolinealidad severa. No invalida la conclusion de cointegracion, pero los coeficientes individuales deben interpretarse con cautela."),
    ]

    for i in range(0, len(hallazgos), 2):
        c1, c2 = st.columns(2)
        for col, (title, body) in zip([c1, c2], hallazgos[i:i+2]):
            col.markdown(f"""
            <div class="scard">
              <div class="scard-title">{title}</div>
              <div class="scard-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="verdict-ok">
      <div class="verdict-v">Conclusion General</div>
      <div class="verdict-sub" style="font-size:15px;color:{TEXT};max-width:700px;margin:14px auto 0;">
        Las remesas de Guatemala mantienen una
        <strong>relacion de equilibrio de largo plazo</strong>
        con el conjunto de variables macroeconomicas analizadas.
        El PIB de Estados Unidos emerge como el ancla individual mas importante.
        Esta evidencia de cointegracion descarta la interpretacion espuria
        y sienta las bases para modelar la dinamica de ajuste de corto plazo
        mediante un VECM.
      </div>
    </div>""", unsafe_allow_html=True)


# -- MAIN -----------------------------------------------------
def main():
    df = load_data()
    z  = build_z(df)

    with st.spinner("Calculando pruebas de raiz unitaria..."):
        ur_tbl = run_unit_root(z)

    # Sidebar
    st.sidebar.markdown(f"""
    <div style='text-align:center;padding:14px 0 18px;'>
      <div style='font-family:Playfair Display,serif;font-size:17px;
                  color:{GOLD};font-weight:800;'>Remesas GT</div>
      <div style='font-size:10px;color:{MUTED};margin-top:3px;'>
        Cointegracion - Proyecto 2</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navegar", [
        "Inicio",
        "Los Datos",
        "Estacionariedad",
        "Remesas vs Variables",
        "Regresion Espuria",
        "Cointegracion",
        "Conclusiones",
    ])

    st.sidebar.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style='font-size:11px;color:{MUTED};padding:6px 0;line-height:1.8;'>
      <strong style='color:{TEXT};'>Equipo</strong><br>
      Gabriel Maas  1183922<br>
      Jose Yoc  1056823<br>
      Maria Puddy  1043723<br><br>
      <strong style='color:{TEXT};'>Periodo</strong><br>
      2004Q1 - 2024Q3  (n = {len(df)})
    </div>
    """, unsafe_allow_html=True)

    if   page == "Inicio":              pg_inicio(df)
    elif page == "Los Datos":           pg_datos(df)
    elif page == "Estacionariedad":     pg_estacion(z, ur_tbl)
    elif page == "Remesas vs Variables":pg_comparacion(df)
    elif page == "Regresion Espuria":   pg_espuria(z)
    elif page == "Cointegracion":       pg_coint(z, ur_tbl)
    elif page == "Conclusiones":        pg_conclusiones()


if __name__ == "__main__":
    main()
