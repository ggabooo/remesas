# ═══════════════════════════════════════════════════════════════════════════
#  app.py  ·  Cointegración de Remesas — Guatemala
#  Econometría II · Proyecto 2
# ═══════════════════════════════════════════════════════════════════════════
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

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Remesas GT · Cointegración",
    page_icon="🇬🇹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PALETA ──────────────────────────────────────────────────────────────────
GOLD   = "#C9A227"
DARK   = "#020B16"
CARD   = "#0A1F33"
TEXT   = "#E6EEF7"
MUTED  = "#A8B9CC"
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

[data-testid="stAppViewContainer"]   {{background:{DARK};}}
[data-testid="stSidebar"]            {{background:#040f1e;border-right:1px solid {GOLD}44;}}
[data-testid="stSidebar"] *          {{color:{TEXT} !important;}}
.block-container                      {{padding-top:1.8rem;padding-bottom:2rem;}}
h1,h2,h3                              {{font-family:'Playfair Display',serif !important;color:{TEXT};}}
p, li, span, div                      {{font-family:'DM Sans',sans-serif;}}

/* Sidebar nav pills */
[data-testid="stRadio"] label {{
    display:block;padding:10px 16px;border-radius:8px;
    margin-bottom:4px;transition:all .2s;cursor:pointer;
    font-family:'DM Sans',sans-serif;font-size:14px;
}}
[data-testid="stRadio"] label:hover {{background:{GOLD}22;}}

/* ── KPI cards ── */
.kpi {{
    background:{CARD};border:1px solid {GOLD}55;border-radius:14px;
    padding:20px 22px;text-align:center;
}}
.kpi-v {{font-family:'Playfair Display',serif;font-size:34px;font-weight:800;color:{GOLD};margin:0;}}
.kpi-l {{font-size:12px;color:{MUTED};margin-top:6px;text-transform:uppercase;letter-spacing:.08em;}}

/* ── Section cards ── */
.scard {{
    background:{CARD};border:1px solid {GOLD}44;border-radius:14px;
    padding:22px 26px;margin-bottom:16px;
}}
.scard-title {{font-family:'Playfair Display',serif;font-size:17px;color:{TEXT};margin-bottom:8px;}}
.scard-body  {{font-size:14px;color:{MUTED};line-height:1.7;}}

/* ── Verdict boxes ── */
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
.verdict-v   {{font-family:'Playfair Display',serif;font-size:26px;font-weight:800;color:{TEXT};}}
.verdict-sub {{font-size:13px;color:{MUTED};margin-top:10px;}}

/* ── Insight boxes ── */
.insight {{
    background:linear-gradient(135deg,#0d1f3d 0%,{CARD} 100%);
    border-left:4px solid {GOLD};border-radius:0 10px 10px 0;
    padding:16px 20px;margin:18px 0;
}}
.insight p {{font-size:14px;color:{TEXT};margin:0;line-height:1.7;}}

hr.g {{border:0;height:1px;background:{GOLD}44;margin:20px 0;}}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DEFAULTS ─────────────────────────────────────────────────────────
def ql(**kw):
    base = dict(
        template="plotly_dark", plot_bgcolor=DARK, paper_bgcolor=DARK,
        font=dict(color=TEXT, family="DM Sans"), title_font=dict(size=16, color=GOLD),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=f"{GOLD}33"),
        margin=dict(t=52, b=44, l=52, r=20),
    )
    base.update(kw); return base

# ─── DATA ────────────────────────────────────────────────────────────────────
RENAME_MAP = {
    "PERIODO":"Periodo","REMESAS-GT":"REMESAS_GT_MUSD","PIB-GT":"PIB_GT_MQUETZ",
    "DESEMPLEO-GT":"DESEMPLEO_GT_PCT","GASOLINA-GT":"GASOLINA_GT_QTQGAL",
    "TC-GT":"TC_GT_QTQUSD","WTI-USA":"PETROLEO_WTI_USD","WIT-USA":"PETROLEO_WTI_USD",
    "DESEMPLEO-USA":"DESEMPLEO_EEUU_PCT","PIB-USA":"PIB_EEUU_BUSD",
}
LABELS = {
    "REMESAS_GT_MUSD":"Remesas GT (MUSD)","PIB_GT_MQUETZ":"PIB Guatemala (MQuetz)",
    "DESEMPLEO_GT_PCT":"Desempleo Guatemala (%)","GASOLINA_GT_QTQGAL":"Gasolina GT (Q/gal)",
    "TC_GT_QTQUSD":"Tipo de Cambio (Q/USD)","PETROLEO_WTI_USD":"Petróleo WTI (USD)",
    "DESEMPLEO_EEUU_PCT":"Desempleo EE.UU. (%)","PIB_EEUU_BUSD":"PIB EE.UU. (BUSD)",
}

@st.cache_data
def load_data(path="EDB-Proyecto2.xlsx"):
    df_raw = pd.read_excel(path, sheet_name="Datos")
    ar = {}
    for col in df_raw.columns:
        for k,v in RENAME_MAP.items():
            if col.strip().upper() == k.upper(): ar[col]=v; break
    df = df_raw.rename(columns=ar)
    df["idx"] = df["Periodo"].apply(
        lambda x: pd.Period(f"{x.split('-')[1]}Q{x.split('-')[0].replace('Q','')}", freq="Q"))
    df = df.set_index("idx").sort_index().drop(columns=["Periodo"], errors="ignore")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.loc[:pd.Period("2024Q3", freq="Q")]

@st.cache_data
def build_z(df):
    log_c  = [c for c in ["REMESAS_GT_MUSD","PIB_GT_MQUETZ","GASOLINA_GT_QTQGAL",
                            "TC_GT_QTQUSD","PETROLEO_WTI_USD","PIB_EEUU_BUSD"] if c in df.columns]
    rate_c = [c for c in ["DESEMPLEO_GT_PCT","DESEMPLEO_EEUU_PCT"] if c in df.columns]
    z = df.copy()
    for c in log_c:  z[f"L_{c}"] = np.log(z[c])
    for c in rate_c: z[f"L_{c}"] = z[c]
    for c in log_c+rate_c: z[f"D_L_{c}"] = z[f"L_{c}"].diff()
    return z

def _integ(series, trend=False):
    s = series.dropna()
    reg = "ct" if trend else "c"
    adf_p  = adfuller(s, regression=reg, autolag="AIC")[1]
    pp_p   = PhillipsPerron(s, trend=reg).pvalue if HAS_ARCH else np.nan
    try: kpss_p = kpss(s, regression=reg, nlags="auto")[1]
    except: kpss_p = np.nan
    adf_d  = adfuller(s.diff().dropna(), regression="c", autolag="AIC")[1]
    pp_d   = PhillipsPerron(s.diff().dropna(), trend="c").pvalue if HAS_ARCH else np.nan
    try: kpss_d = kpss(s.diff().dropna(), regression="c", nlags="auto")[1]
    except: kpss_d = np.nan
    lv = sum([adf_p<0.05, (not np.isnan(pp_p)) and pp_p<0.05,
              (not np.isnan(kpss_p)) and kpss_p>0.05])
    dv = sum([adf_d<0.05, (not np.isnan(pp_d)) and pp_d<0.05,
              (not np.isnan(kpss_d)) and kpss_d>0.05])
    orden = "I(0)" if lv>=2 else ("I(1)" if dv>=2 else "N/C")
    return orden, round(adf_p,4), round(pp_p,4) if not np.isnan(pp_p) else np.nan, \
           round(kpss_p,4) if not np.isnan(kpss_p) else np.nan

@st.cache_data
def run_unit_root(z):
    l_cols = [c for c in z.columns if c.startswith("L_") and not c.startswith("D_L_")]
    rows = []
    for v in l_cols:
        trend = any(k in v for k in ["REMESAS","PIB"])
        orden, adf_p, pp_p, kpss_p = _integ(z[v], trend)
        rows.append({"Serie":v, "ADF":adf_p, "PP":pp_p, "KPSS":kpss_p, "Orden":orden})
    return pd.DataFrame(rows)

@st.cache_data
def run_coint(z, ur_tbl):
    dep = "L_REMESAS_GT_MUSD"
    i1  = ur_tbl.loc[ur_tbl["Orden"]=="I(1)","Serie"].tolist()
    ind = [v for v in i1 if v != dep]
    df_reg = z[[dep]+ind].dropna()
    Y = df_reg[dep]; X = sm.add_constant(df_reg[ind])
    ols = sm.OLS(Y,X).fit()
    resid = ols.resid
    adf_r = adfuller(resid, regression="n", autolag="AIC")
    dw_r  = durbin_watson(resid)

    pairs = []
    for xv in ind:
        pdf = z[[dep,xv]].dropna()
        m   = sm.OLS(pdf[dep], sm.add_constant(pdf[xv])).fit()
        r   = m.resid
        ap  = adfuller(r, regression="n", autolag="AIC")
        dw  = durbin_watson(r)
        pairs.append({
            "Variable": LABELS.get(xv.replace("L_",""), xv.replace("L_","")),
            "β": round(m.params[xv],4), "R²": round(m.rsquared,4),
            "DW": round(dw,4), "ADF p": round(ap[1],4),
            "¿Cointegrados?": "✅" if ap[1]<0.05 else "❌",
        })
    return ols, resid, adf_r, dw_r, pd.DataFrame(pairs), df_reg, ind

# ─── CHART HELPERS ───────────────────────────────────────────────────────────
def ts_idx(df): return df.index.to_timestamp(how="end")

def acf_pacf_fig(series, label="", nlags=16):
    s = series.dropna().values; n = len(s)
    ci = 1.96/np.sqrt(n)
    av = sm_acf(s, nlags=nlags)
    pv = sm_pacf(s, nlags=nlags, method="ywm")
    lags = list(range(len(av)))
    fig = make_subplots(1,2,subplot_titles=["ACF","PACF"])
    for lg,v in zip(lags,av):
        fig.add_trace(go.Bar(x=[lg],y=[v],marker_color=GOLD if abs(v)>ci and lg>0
                             else f"rgba(37,99,235,0.55)",showlegend=False),row=1,col=1)
    for lg,v in zip(lags,pv):
        fig.add_trace(go.Bar(x=[lg],y=[v],marker_color=GOLD if abs(v)>ci and lg>0
                             else f"rgba(37,99,235,0.55)",showlegend=False),row=1,col=2)
    for col in [1,2]:
        fig.add_hline(y=ci, line_dash="dash",line_color=RED,line_width=1,row=1,col=col)
        fig.add_hline(y=-ci,line_dash="dash",line_color=RED,line_width=1,row=1,col=col)
        fig.add_hline(y=0,  line_color="white",line_width=0.4,row=1,col=col)
    fig.update_layout(**ql(title=f"ACF / PACF — {label}", height=320, bargap=0.15))
    return fig

# ─── PAGES ───────────────────────────────────────────────────────────────────
def pg_inicio(df):
    st.markdown(f"""
    <h1 style='font-family:Playfair Display,serif;font-size:46px;
               background:linear-gradient(90deg,{TEXT},{GOLD});
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               margin-bottom:4px;'>
        Cointegración de Remesas
    </h1>
    <p style='font-size:20px;color:{MUTED};margin-bottom:6px;'>
        Guatemala · Análisis econométrico 2004 – 2024
    </p>
    <p style='font-size:15px;color:{GOLD};font-family:DM Sans,sans-serif;'>
        Econometría II · Ing. Héctor Francisco Galeros Juárez
    </p>
    <p style='font-size:13px;color:{MUTED};'>
        Gabriel Estuardo Maas Ordóñez · José Eduardo Yoc García · María Isabel Puddy Díaz
    </p>
    <hr class='g'>
    """, unsafe_allow_html=True)

    # KPI cards
    cols = st.columns(4)
    kpis = [
        (f"{len(df)}", "Observaciones"),
        ("8", "Variables analizadas"),
        ("2004 – 2024", "Período"),
        ("I(1)", "Orden de integración"),
    ]
    for c,(v,l) in zip(cols,kpis):
        c.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Hero chart — Remesas over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_idx(df), y=df["REMESAS_GT_MUSD"],
        mode="lines", name="Remesas GT",
        line=dict(color=GOLD, width=2.5),
        fill="tozeroy", fillcolor=f"rgba(201,162,39,0.10)"
    ))
    fig.update_layout(**ql(
        title="Remesas de Guatemala · Serie Histórica (millones de USD)",
        xaxis_title="Trimestre", yaxis_title="MUSD", height=380
    ))
    st.plotly_chart(fig, use_container_width=True)

    # About the study
    st.markdown(f"""
    <div class="scard">
      <div class="scard-title">¿Qué analiza este estudio?</div>
      <div class="scard-body">
        Las remesas enviadas a Guatemala han crecido de manera sostenida desde 2004.
        Este proyecto evalúa si esa trayectoria está <strong style='color:{GOLD}'>
        vinculada de forma estable y de largo plazo</strong> con variables macroeconómicas
        clave de Guatemala y de Estados Unidos, aplicando el enfoque de
        <strong style='color:{GOLD}'>cointegración de Engle-Granger</strong>:
        pruebas de raíz unitaria (ADF, PP, KPSS), diagnóstico de regresión espuria
        y test ADF sobre residuales del modelo de regresión múltiple.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, (icon, title, body) in zip([col1,col2,col3],[
        ("📐","Hipótesis","¿Existe una relación de equilibrio de largo plazo entre las remesas y las variables macroeconómicas analizadas?"),
        ("🔬","Metodología","Pruebas ADF · PP · KPSS → Regresión espuria → Engle-Granger → Análisis por pares"),
        ("✅","Resultado principal","Los residuales del modelo múltiple son estacionarios (ADF p < 0.001): cointegración confirmada."),
    ]):
        col.markdown(f"""
        <div class="scard">
          <div class="scard-title">{icon} {title}</div>
          <div class="scard-body">{body}</div>
        </div>""", unsafe_allow_html=True)


def pg_datos(df):
    st.markdown(f"<h2>📈 Exploración de las Series</h2>", unsafe_allow_html=True)
    st.markdown(f'<p style="color:{MUTED}">Comportamiento de las 8 variables en el período 2004-2024</p>',
                unsafe_allow_html=True)

    # 4×2 mini charts grid
    cols_list = list(df.columns)
    colors_cycle = [GOLD, "#60A5FA", "#34D399", "#F87171", "#A78BFA", "#FBBF24", "#38BDF8", "#FB923C"]
    fig = make_subplots(rows=4, cols=2,
                        subplot_titles=[LABELS.get(c,c) for c in cols_list],
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    for i, (col, clr) in enumerate(zip(cols_list, colors_cycle)):
        r, c = divmod(i, 2); r += 1; c += 1
        fig.add_trace(go.Scatter(x=ts_idx(df), y=df[col], mode="lines",
                                 line=dict(color=clr, width=1.8), showlegend=False), row=r, col=c)
    fig.update_layout(**ql(height=900, title="Series de tiempo — Niveles"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='g'>", unsafe_allow_html=True)

    # Trend table
    st.markdown(f"<h3>Tendencia lineal simple (OLS vs tiempo)</h3>", unsafe_allow_html=True)
    trend_rows = []
    for c in df.columns:
        y = df[c].dropna().values
        t = np.arange(1, len(y)+1)
        m = sm.OLS(y, sm.add_constant(t)).fit()
        trend_rows.append({
            "Variable": LABELS.get(c, c),
            "Pendiente": round(m.params[1], 4),
            "p-valor": round(m.pvalues[1], 4),
            "R²": round(m.rsquared, 4),
            "Tendencia significativa": "✅ Sí" if m.pvalues[1] < 0.05 else "❌ No",
        })
    tdf = pd.DataFrame(trend_rows)

    # Bar chart of R² 
    fig2 = go.Figure(go.Bar(
        x=tdf["Variable"], y=tdf["R²"],
        marker_color=[GOLD if r>0.5 else BLUE for r in tdf["R²"]],
        text=[f"{v:.3f}" for v in tdf["R²"]], textposition="outside"
    ))
    fig2.update_layout(**ql(title="R² de tendencia lineal por serie", height=360,
                            yaxis=dict(range=[0,1.05])))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="insight"><p>
    Las series con R² alto (Remesas, PIB Guatemala, PIB EE.UU.) presentan
    <strong>tendencia determinística clara</strong>, lo que justifica incluirla
    como especificación en las pruebas de raíz unitaria. Gasolina, Tipo de Cambio
    y Petróleo no exhiben tendencia lineal significativa.
    </p></div>""", unsafe_allow_html=True)


def pg_estacion(z, ur_tbl):
    st.markdown("<h2>〰️ Estacionariedad</h2>", unsafe_allow_html=True)

    # Variable selector for ACF/PACF
    l_cols = [c for c in z.columns if c.startswith("L_") and not c.startswith("D_L_")]
    var_sel = st.selectbox(
        "Selecciona una variable para ver su ACF / PACF:",
        l_cols,
        format_func=lambda x: LABELS.get(x.replace("L_",""), x)
    )
    fig = acf_pacf_fig(z[var_sel], label=LABELS.get(var_sel.replace("L_",""), var_sel))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight"><p>
    Un <strong>decaimiento lento y persistente del ACF</strong> (barras que siguen siendo
    significativas durante muchos rezagos) es la firma visual de una serie con
    <strong>raíz unitaria (no estacionaria)</strong>.
    Un corte abrupto después del lag 1 sugiere proceso estacionario o MA(1).
    </p></div>""", unsafe_allow_html=True)

    st.markdown("<hr class='g'>", unsafe_allow_html=True)
    st.markdown("<h3>Resultados de Raíz Unitaria (ADF · PP · KPSS)</h3>", unsafe_allow_html=True)

    # Styled results table
    disp = ur_tbl.copy()
    disp["Variable"] = disp["Serie"].apply(
        lambda x: LABELS.get(x.replace("L_",""), x.replace("L_","")))
    disp = disp[["Variable","ADF","PP","KPSS","Orden"]]

    # Color-code p-values in a bar chart
    fig2 = go.Figure()
    for col_name, offset, clr in [("ADF", -0.25, GOLD), ("PP", 0, BLUE), ("KPSS", 0.25, "#34D399")]:
        vals = pd.to_numeric(disp[col_name], errors="coerce").fillna(0.5)
        fig2.add_trace(go.Bar(
            name=col_name, x=disp["Variable"], y=vals,
            offsetgroup=col_name,
            marker_color=clr, opacity=0.85,
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10)
        ))
    fig2.add_hline(y=0.05, line_dash="dash", line_color=RED,
                   line_width=2, annotation_text="α = 0.05", annotation_font_color=RED)
    fig2.update_layout(**ql(
        title="p-valores de las pruebas de raíz unitaria en NIVELES",
        barmode="group", height=420,
        yaxis=dict(range=[0, 1.05], title="p-valor"),
        legend=dict(orientation="h", y=1.12)
    ))
    st.plotly_chart(fig2, use_container_width=True)

    # Result cards
    i1_count = (ur_tbl["Orden"]=="I(1)").sum()
    i0_count = (ur_tbl["Orden"]=="I(0)").sum()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="kpi"><div class="kpi-v">{i1_count}</div>'
                f'<div class="kpi-l">Series I(1) — no estacionarias</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="kpi-v">{i0_count}</div>'
                f'<div class="kpi-l">Series I(0) — estacionarias</div></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="kpi-v">I(1)</div>'
                f'<div class="kpi-l">Orden de integración predominante</div></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(disp.rename(columns={"ADF":"ADF p-val","PP":"PP p-val","KPSS":"KPSS p-val"}),
                 use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight"><p>
    <strong>7 de 8 series resultan I(1)</strong>: se confirma raíz unitaria en niveles
    pero no en primeras diferencias. El Petróleo WTI es I(0), por lo que queda
    excluido del análisis de cointegración (no se puede cointegrar series de diferente
    orden de integración).
    </p></div>""", unsafe_allow_html=True)


def pg_comparacion(df):
    st.markdown("<h2>🔗 Remesas vs Variables</h2>", unsafe_allow_html=True)

    otras = [c for c in df.columns if c != "REMESAS_GT_MUSD"]
    var_sel = st.selectbox(
        "Comparar Remesas con:",
        otras, format_func=lambda x: LABELS.get(x, x)
    )

    paired = df[["REMESAS_GT_MUSD", var_sel]].dropna()
    corr = paired.corr().iloc[0,1]

    c1, c2 = st.columns(2)

    # Dual-axis time series
    with c1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=ts_idx(df), y=df["REMESAS_GT_MUSD"],
                                 name="Remesas GT", line=dict(color=GOLD, width=2)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=ts_idx(df), y=df[var_sel],
                                 name=LABELS.get(var_sel, var_sel),
                                 line=dict(color=BLUE, width=2, dash="dot")),
                      secondary_y=True)
        fig.update_layout(**ql(title="Series en el tiempo (doble eje)", height=360))
        fig.update_yaxes(title_text="Remesas (MUSD)", secondary_y=False,
                         showgrid=True, gridcolor=f"{GOLD}22")
        fig.update_yaxes(title_text=LABELS.get(var_sel, var_sel),
                         secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    # Scatter
    with c2:
        x_v = paired[var_sel].values
        y_v = paired["REMESAS_GT_MUSD"].values
        p   = np.polyfit(x_v, y_v, 1)
        x_l = np.linspace(x_v.min(), x_v.max(), 200)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_v, y=y_v, mode="markers",
                                  marker=dict(color=GOLD, size=6, opacity=0.7),
                                  name="Datos"))
        fig2.add_trace(go.Scatter(x=x_l, y=np.polyval(p, x_l),
                                  mode="lines", line=dict(color=BLUE, width=2),
                                  name="Tendencia"))
        fig2.update_layout(**ql(
            title=f"Dispersión  |  r = {corr:.3f}",
            xaxis_title=LABELS.get(var_sel, var_sel),
            yaxis_title="Remesas GT (MUSD)", height=360
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation color
    strength = "muy fuerte" if abs(corr)>0.85 else ("fuerte" if abs(corr)>0.60 else "moderada/débil")
    direction = "positiva" if corr > 0 else "negativa"
    st.markdown(f"""
    <div class="scard">
      <div class="scard-title">Correlación de Pearson: {corr:.3f}</div>
      <div class="scard-body">
        Existe una correlación <strong style='color:{GOLD}'>{strength} y {direction}</strong>
        entre las Remesas y <strong>{LABELS.get(var_sel, var_sel)}</strong>.
        Sin embargo, una correlación alta entre series no estacionarias puede ser
        resultado de una <em>regresión espuria</em> — por eso se requieren pruebas
        formales de cointegración.
      </div>
    </div>""", unsafe_allow_html=True)

    # All correlations ranked
    st.markdown("<hr class='g'>", unsafe_allow_html=True)
    st.markdown("<h3>Correlaciones con Remesas — todas las variables</h3>", unsafe_allow_html=True)
    corrs = {LABELS.get(c,c): df[["REMESAS_GT_MUSD",c]].dropna().corr().iloc[0,1]
             for c in otras}
    corr_s = pd.Series(corrs).sort_values(ascending=True)
    fig3 = go.Figure(go.Bar(
        x=corr_s.values, y=corr_s.index, orientation="h",
        marker_color=[GOLD if v>0 else RED for v in corr_s.values],
        text=[f"{v:.3f}" for v in corr_s.values], textposition="outside"
    ))
    fig3.add_vline(x=0, line_color="white", line_width=1)
    fig3.update_layout(**ql(title="Correlación de Pearson con Remesas GT", height=380,
                            xaxis=dict(range=[-1.1, 1.1])))
    st.plotly_chart(fig3, use_container_width=True)


def pg_espuria(z):
    st.markdown("<h2>⚠️ Diagnóstico de Regresión Espuria</h2>", unsafe_allow_html=True)
    st.markdown(f'<p style="color:{MUTED}">Regresar series I(1) sin verificar cointegración'
                f' puede producir resultados aparentemente significativos pero falsos.</p>',
                unsafe_allow_html=True)

    dep = "L_REMESAS_GT_MUSD"
    all_l = [c for c in z.columns if c.startswith("L_") and not c.startswith("D_L_") and c != dep]
    df_r  = z[[dep]+all_l].dropna()
    ols_e = sm.OLS(df_r[dep], sm.add_constant(df_r[all_l])).fit()
    resid = ols_e.resid
    dw    = durbin_watson(resid)
    r2    = ols_e.rsquared
    adf_p = adfuller(resid, regression="c", autolag="AIC")[1]

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="kpi"><div class="kpi-v">{r2:.4f}</div>'
                f'<div class="kpi-l">R² del modelo en niveles</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="kpi-v" style="color:{"#EA580C" if dw<1.5 else GOLD}">'
                f'{dw:.3f}</div><div class="kpi-l">Durbin-Watson</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="kpi-v">{adf_p:.4f}</div>'
                f'<div class="kpi-l">ADF p-val (residuales)</div></div>', unsafe_allow_html=True)
    verdict_txt = "⚠️ Posible espuria" if dw < 1.5 else "✅ Sin señal"
    c4.markdown(f'<div class="kpi"><div class="kpi-v" style="font-size:22px;">{verdict_txt}</div>'
                f'<div class="kpi-l">R² alto + DW bajo</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_r.index.to_timestamp(how="end"), y=resid,
                                 mode="lines", line=dict(color=ORANGE, width=1.8), name="Residuales"))
        fig.add_hline(y=0, line_color="white", line_width=1, line_dash="dash")
        fig.update_layout(**ql(title=f"Residuales en el tiempo  (DW = {dw:.3f})", height=330))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        n  = len(resid)
        ci = 1.96/np.sqrt(n)
        av = sm_acf(resid.values, nlags=16)
        fig2 = go.Figure(go.Bar(
            x=list(range(len(av))), y=av,
            marker_color=[ORANGE if abs(v)>ci and i>0 else f"rgba(37,99,235,0.55)"
                          for i,v in enumerate(av)]
        ))
        fig2.add_hline(y=ci,  line_dash="dash", line_color=RED, line_width=1)
        fig2.add_hline(y=-ci, line_dash="dash", line_color=RED, line_width=1)
        fig2.update_layout(**ql(title="ACF de residuales — ¿hay memoria?", height=330, bargap=0.15))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div class="verdict-warn">
      <div class="verdict-v">⚠️ Señal de Alarma Detectada</div>
      <div class="verdict-sub">
        R² = {r2:.4f} (muy alto) combinado con DW = {dw:.3f} (menor a 1.5)
        y autocorrelación persistente en residuales indica que la regresión en niveles
        <strong>puede ser espuria</strong>. Esto no invalida el análisis —
        <strong>justifica proceder al test formal de cointegración</strong>.
      </div>
    </div>""", unsafe_allow_html=True)


def pg_coint(z, ur_tbl):
    ols, resid, adf_r, dw_r, pair_tbl, df_reg, ind_vars = run_coint(z, ur_tbl)

    st.markdown("<h2>✅ Cointegración — Engle-Granger</h2>", unsafe_allow_html=True)

    # ── 6.1 Multiple regression key coefficients
    st.markdown("<h3>6.1 Modelo de Regresión Múltiple (niveles)</h3>", unsafe_allow_html=True)

    params  = ols.params.drop("const")
    pvals   = ols.pvalues.drop("const")
    sig_col = [GREEN if p<0.05 else RED for p in pvals]
    short   = [LABELS.get(v.replace("L_",""), v.replace("L_","")) for v in params.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=params.values, y=short, orientation="h",
        marker_color=[GREEN if p<0.05 else f"rgba(220,38,38,0.6)" for p in pvals],
        text=[f"β={v:.3f}  p={p:.3f}" for v,p in zip(params.values, pvals)],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(**ql(title="Coeficientes del modelo · Verde = significativo (p<0.05)",
                           xaxis_title="Valor del coeficiente", height=360))
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="kpi"><div class="kpi-v">{ols.rsquared:.4f}</div>'
                f'<div class="kpi-l">R²</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="kpi-v">{dw_r:.3f}</div>'
                f'<div class="kpi-l">Durbin-Watson</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="kpi-v">{len(df_reg)}</div>'
                f'<div class="kpi-l">Observaciones</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr class='g'>", unsafe_allow_html=True)

    # ── 6.2 ADF on residuals — THE VERDICT
    st.markdown("<h3>6.2 Test ADF sobre Residuales del Modelo</h3>", unsafe_allow_html=True)
    adf_stat = adf_r[0]; adf_p = adf_r[1]; adf_cv = adf_r[4]

    c1, c2 = st.columns([1,1])

    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_reg.index.to_timestamp(how="end"), y=resid,
                                  mode="lines", line=dict(color=GREEN, width=1.8),
                                  fill="tozeroy", fillcolor="rgba(22,163,74,0.08)"))
        fig2.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
        fig2.update_layout(**ql(title="Residuales de la regresión de cointegración", height=300))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        ok = adf_p < 0.05
        box_class = "verdict-ok" if ok else "verdict-warn"
        icon = "✅" if ok else "❌"
        msg  = "COINTEGRACIÓN CONFIRMADA" if ok else "NO SE CONFIRMA COINTEGRACIÓN"
        detail = (f"Los residuales son <strong>estacionarios</strong> → la relación entre remesas "
                  f"y las variables independientes es de <strong>equilibrio de largo plazo real</strong>,"
                  f" no espuria." if ok else
                  "Los residuales tienen raíz unitaria → la relación podría ser espuria.")
        st.markdown(f"""
        <div class="{box_class}">
          <div class="verdict-v">{icon} {msg}</div>
          <div class="verdict-sub">
            ADF estadístico: <strong>{adf_stat:.4f}</strong> &nbsp;|&nbsp;
            p-valor: <strong style='color:{"#4ade80" if ok else RED}'>{adf_p:.4f}</strong><br>
            Valor crítico 1%: {adf_cv["1%"]:.4f} &nbsp;·&nbsp;
            5%: {adf_cv["5%"]:.4f} &nbsp;·&nbsp; 10%: {adf_cv["10%"]:.4f}<br><br>
            {detail}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='g'>", unsafe_allow_html=True)

    # ── 6.3 Pairwise
    st.markdown("<h3>6.3 Análisis por Pares</h3>", unsafe_allow_html=True)

    # ADF p-values bar chart
    fig3 = go.Figure(go.Bar(
        x=pair_tbl["Variable"], y=pair_tbl["ADF p"],
        marker_color=[GREEN if v<0.05 else f"rgba(220,38,38,0.7)" for v in pair_tbl["ADF p"]],
        text=[f'{v:.3f}' for v in pair_tbl["ADF p"]], textposition="outside"
    ))
    fig3.add_hline(y=0.05, line_dash="dash", line_color=GOLD,
                   annotation_text="umbral α=0.05", annotation_font_color=GOLD, line_width=2)
    fig3.update_layout(**ql(
        title="ADF p-valor de residuales por par · Verde = cointegrado",
        yaxis=dict(range=[0, 0.75], title="p-valor ADF"), height=370
    ))
    st.plotly_chart(fig3, use_container_width=True)

    # Pairwise table
    display_pairs = pair_tbl.copy()
    st.dataframe(display_pairs, use_container_width=True, hide_index=True)

    coint_vars = pair_tbl.loc[pair_tbl["¿Cointegrados?"]=="✅","Variable"].tolist()
    st.markdown(f"""
    <div class="insight"><p>
    Del análisis por pares, <strong style='color:{GOLD}'>{len(coint_vars)} variable(s)
    presentan cointegración bilateral con Remesas</strong>:
    <strong>{', '.join(coint_vars) if coint_vars else 'ninguna'}</strong>.
    El vínculo bilateral más sólido es con el <strong>PIB de EE.UU.</strong>,
    lo que refleja la dependencia directa del flujo de remesas con la
    salud económica de la comunidad guatemalteca en ese país.
    El modelo múltiple sí confirma cointegración conjunta, lo que indica que
    la relación de equilibrio emerge del <em>conjunto</em> de variables, no
    de cada par de forma aislada.
    </p></div>""", unsafe_allow_html=True)


def pg_conclusiones(ur_tbl, adf_r_p):
    st.markdown("<h2>📋 Conclusiones</h2>", unsafe_allow_html=True)

    hallazgos = [
        ("📉", "Series No Estacionarias",
         f"7 de 8 variables son I(1). Confirman raíz unitaria en niveles mediante ADF, PP y KPSS, pero son estacionarias en primeras diferencias. El Petróleo WTI es I(0)."),
        ("⚠️", "Señal de Regresión Espuria",
         "El modelo en niveles produce R² = 0.986 y DW = 1.12. El DW bajo indica autocorrelación en residuales, lo que advierte sobre posible regresión espuria sin verificación adicional."),
        ("✅", "Cointegración Confirmada",
         f"ADF sobre residuales del modelo múltiple: estadístico -3.35, p-valor = 0.0008. Los residuales son estacionarios → existe una relación de equilibrio de largo plazo real entre las Remesas y las variables I(1) conjuntamente."),
        ("🇺🇸", "PIB de EE.UU. — El Factor Clave",
         "Es la única variable individualmente cointegrada con Remesas (p=0.027, DW=1.07). Refleja que el nivel de actividad económica en EE.UU. es el determinante principal del flujo de remesas hacia Guatemala."),
        ("🔬", "Implicación Econométrica",
         "La cointegración implica que, aunque las series se alejan en el corto plazo, regresan a un equilibrio de largo plazo. Esto valida el uso de un Modelo de Corrección de Error (VECM) para análisis dinámico."),
        ("📊", "Sobre Multicolinealidad",
         "El condition number = 6,730 señala multicolinealidad severa en el modelo múltiple. Esto no invalida la conclusión de cointegración, pero los coeficientes individuales deben interpretarse con cautela."),
    ]

    for i in range(0, len(hallazgos), 2):
        c1, c2 = st.columns(2)
        for col, (icon, title, body) in zip([c1,c2], hallazgos[i:i+2]):
            col.markdown(f"""
            <div class="scard">
              <div class="scard-title">{icon} {title}</div>
              <div class="scard-body">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="verdict-ok">
      <div class="verdict-v">✅ Conclusión General</div>
      <div class="verdict-sub" style="font-size:16px;color:{TEXT};max-width:700px;margin:16px auto 0;">
        Las remesas de Guatemala mantienen una <strong>relación de equilibrio de largo plazo</strong>
        con el conjunto de variables macroeconómicas analizadas. El PIB de Estados Unidos
        emerge como el ancla individual más importante. Esta evidencia de cointegración
        descarta la interpretación espuria de la relación y sienta las bases para modelar
        la dinámica de ajuste de corto plazo mediante un VECM.
      </div>
    </div>""", unsafe_allow_html=True)

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    z  = build_z(df)

    with st.spinner("Calculando pruebas de raíz unitaria..."):
        ur_tbl = run_unit_root(z)

    # Sidebar navigation
    st.sidebar.markdown(f"""
    <div style='text-align:center;padding:16px 0 20px;'>
      <div style='font-family:Playfair Display,serif;font-size:18px;color:{GOLD};
                  font-weight:800;'>Remesas GT</div>
      <div style='font-size:11px;color:{MUTED};margin-top:4px;'>Cointegración · Proyecto 2</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navegar", [
        "🏠  Inicio",
        "📈  Los Datos",
        "〰️  Estacionariedad",
        "🔗  Remesas vs Variables",
        "⚠️  Regresión Espuria",
        "✅  Cointegración",
        "📋  Conclusiones",
    ])

    st.sidebar.markdown("<hr class='g'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style='font-size:11px;color:{MUTED};padding:8px 0;line-height:1.7;'>
      <strong style='color:{TEXT};'>Equipo</strong><br>
      Gabriel Maas · 1183922<br>
      José Yoc · 1056823<br>
      María Puddy · 1043723<br><br>
      <strong style='color:{TEXT};'>Período</strong><br>
      2004Q1 – 2024Q3 &nbsp;(n = {len(df)})
    </div>
    """, unsafe_allow_html=True)

    if   "Inicio"        in page: pg_inicio(df)
    elif "Los Datos"     in page: pg_datos(df)
    elif "Estacionariedad" in page:
        with st.spinner("Corriendo pruebas de integración..."): pass
        pg_estacion(z, ur_tbl)
    elif "Remesas vs"    in page: pg_comparacion(df)
    elif "Espuria"       in page: pg_espuria(z)
    elif "Cointegración" in page:
        with st.spinner("Calculando modelo de cointegración..."):
            pass
        pg_coint(z, ur_tbl)
    elif "Conclusiones"  in page:
        _, _, adf_r, *_ = run_coint(z, ur_tbl)
        pg_conclusiones(ur_tbl, adf_r[1])

if __name__ == "__main__":
    main()
