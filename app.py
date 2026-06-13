import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AeroRight",
    page_icon="✈️",
    layout="centered"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    .stApp { background-color: #f0f4f8; }
    h1, h2, h3 { color: #000000 !important; }
    p { color: #000000 !important; }
    [data-testid="stMarkdownContainer"] p { color: #000000 !important; }
    [data-testid="stMarkdownContainer"] h3 { color: #000000 !important; }

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #000000;
        padding: 8px 8px 0px 8px;
        border-radius: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #000000;
        color: #888888;
        border-radius: 0px;
        padding: 8px 20px;
        font-size: 14px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f4f8 !important;
        color: #000000 !important;
        font-weight: 600;
    }

    /* inputs */
    .stSelectbox > div > div {
        border-radius: 0px !important;
        border: 1.5px solid #000000 !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* buttons */
    .stButton > button {
        border-radius: 0px !important;
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500;
        width: 100%;
        padding: 10px;
        font-size: 14px;
    }
    .stButton > button:hover { background-color: #333333 !important; }

    /* cards */
    .stat-card { border: 1.5px solid #000; padding: 12px 16px; background: #ffffff; }
    .stat-label { font-size: 11px; color: #666; margin-bottom: 4px; }
    .stat-value { font-size: 22px; font-weight: 500; color: #000; }
    .finding-card { border: 1.5px solid #000; padding: 12px 16px; background: #ffffff; }
    .finding-title { font-size: 11px; color: #666; margin-bottom: 4px; }
    .finding-value { font-size: 14px; font-weight: 500; color: #000; }
    .finding-reason { font-size: 11px; color: #888; margin-top: 4px; line-height: 1.4; }

    /* result box */
    .result-box { background-color: #000; color: #fff; padding: 20px; border: 1.5px solid #000; margin-top: 15px; font-size: 14px; line-height: 1.9; }
    .risk-high { color: #ff6b6b; font-weight: bold; }
    .risk-low  { color: #69db7c; font-weight: bold; }
    .risk-mid  { color: #ffd43b; font-weight: bold; }
    .result-divider { border-top: 1px solid #333; margin: 8px 0; }

    .cta-box { background: #000; color: #fff; padding: 12px 16px; font-size: 13px; }
    .section-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px; font-weight: 500; }
    .page-sub { color: #666; font-size: 14px; margin-bottom: 1.5rem; }
    .historical-note { font-size: 11px; color: #888; margin-top: 8px; font-style: italic; margin-bottom: 12px; }
    .footer-text { font-size: 10px; color: #aaa; text-align: center; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

@st.cache_data
def load_data():
    return pd.read_parquet("airline_data.parquet")

try:
    model = load_model()
    data  = load_data()
except Exception as e:
    st.error(f"Could not load model or data: {e}")
    st.stop()

features = [
    'arr_flights', 'nas_delay', 'security_delay',
    'delay_minutes_per_flight_capped_p999',
    'avg_delay_minutes_given_delayed_capped_p999',
    'month', 'delay_rate', 'weather_pct', 'carrier_pct'
]

month_map = {
    1:'January', 2:'February', 3:'March', 4:'April',
    5:'May', 6:'June', 7:'July', 8:'August',
    9:'September', 10:'October', 11:'November', 12:'December'
}

def get_month_number(name):
    return [k for k,v in month_map.items() if v==name][0]

def show_risk(prob):
    if prob >= 0.5:
        return f'<span class="risk-high">{prob:.0%} — HIGH RISK ⚠️</span>'
    elif prob >= 0.25:
        return f'<span class="risk-mid">{prob:.0%} — MODERATE RISK</span>'
    else:
        return f'<span class="risk-low">{prob:.0%} — LOW RISK ✅</span>'

airport_lookup = (
    data[['airport','airport_name']].dropna().drop_duplicates()
    .set_index('airport')['airport_name'].to_dict()
)
airline_list      = sorted(data['carrier_name'].dropna().unique().tolist())
airport_full_list = sorted(airport_lookup.values())
month_list        = list(month_map.values())
name_to_code      = {v:k for k,v in airport_lookup.items()}

def get_code(name):
    return name_to_code.get(name, name)

# header
st.markdown("# ✈️ AeroRight")
st.markdown("<div style='color:#666; margin-bottom:1rem;'>Right choice for your travel</div>", unsafe_allow_html=True)
st.markdown("---")

# tabs - always visible, always works
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🧳 Customer Tool",
    "🏢 Airline Tool",
    "🏆 Recommendation System"
])

# ── HOME ───────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Before you book your next flight")
    st.markdown('<div class="page-sub">Here is what 10 years of U.S. airline data tells you</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown('<div class="stat-card"><div class="stat-label">Records analysed</div><div class="stat-value">224K</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="stat-card"><div class="stat-label">Airlines covered</div><div class="stat-value">21</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="stat-card"><div class="stat-label">Airports covered</div><div class="stat-value">300+</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="stat-card"><div class="stat-label">Model accuracy</div><div class="stat-value">83%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Key findings from 10 years of data</div>', unsafe_allow_html=True)

    f1,f2,f3 = st.columns(3)
    f1.markdown('<div class="finding-card"><div class="finding-title">Best airline (2015–2025)</div><div class="finding-value">Delta Air Lines — 0.1%</div><div class="finding-reason">Lowest cancellation rate across 10 years of data</div></div>', unsafe_allow_html=True)
    f2.markdown('<div class="finding-card"><div class="finding-title">Worst airline (2015–2025)</div><div class="finding-value">Peninsula Airways — 15.6%</div><div class="finding-reason">1 in 6 flights gets cancelled — avoid if possible</div></div>', unsafe_allow_html=True)
    f3.markdown('<div class="finding-card"><div class="finding-title">Best managed airport</div><div class="finding-value">Atlanta (ATL) — lowest cancellation</div><div class="finding-reason">Handles the most flights yet stays lowest risk</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    f4,f5,f6 = st.columns(3)
    f4.markdown('<div class="finding-card"><div class="finding-title">Most dangerous month</div><div class="finding-value">April — 3.5% cancellation</div><div class="finding-reason">Spring storms cause the highest disruptions of the year</div></div>', unsafe_allow_html=True)
    f5.markdown('<div class="finding-card"><div class="finding-title">Safest month to fly</div><div class="finding-value">November — lowest risk</div><div class="finding-reason">Best time to book — consistently lowest cancellation rate</div></div>', unsafe_allow_html=True)
    f6.markdown('<div class="finding-card"><div class="finding-title">Biggest delay cause</div><div class="finding-value">Carrier operations — 39%</div><div class="finding-reason">Most delays are the airline\'s fault — not weather</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cta-box">Want to check YOUR specific flight? Click a tool tab above ↑</div>', unsafe_allow_html=True)

# ── CUSTOMER TOOL ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🧳 Customer Tool")
    st.markdown('<div class="page-sub">Check your flight\'s cancellation risk — based on 10 years of historical patterns (2015–2025)</div>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        sel_airport_name = st.selectbox("Airport", airport_full_list, key="ct_ap")
        sel_airport = get_code(sel_airport_name)
    with col2:
        airlines_here = sorted(data[data['airport']==sel_airport]['carrier_name'].dropna().unique().tolist())
        sel_airline = st.selectbox("Airline", airlines_here, key="ct_al")
    with col3:
        sel_month = st.selectbox("Month", month_list, key="ct_mo")

    st.markdown('<div class="historical-note">Based on historical patterns from 2015–2025 BTS data — not real-time predictions</div>', unsafe_allow_html=True)

    if st.button("Check Historical Risk", key="ct_btn"):
        m = get_month_number(sel_month)
        filtered = data[(data['carrier_name']==sel_airline)&(data['airport']==sel_airport)&(data['month']==m)].dropna(subset=features)

        if filtered.empty:
            st.warning("No data found for this combination.")
        else:
            risk    = model.predict_proba(filtered[features])[:,1].mean()
            others  = data[(data['airport']==sel_airport)&(data['month']==m)].dropna(subset=features).copy()
            others['risk'] = model.predict_proba(others[features])[:,1]
            avg_risk    = others['risk'].mean()
            safest      = others.groupby('carrier_name')['risk'].mean().idxmin()
            safest_risk = others.groupby('carrier_name')['risk'].mean().min()

            st.markdown(f"""
            <div class="result-box">
                <b>Airline:</b> {sel_airline}<br>
                <b>Airport:</b> {sel_airport_name} &nbsp;|&nbsp; <b>Month:</b> {sel_month}
                <div class="result-divider"></div>
                <b>Historical Cancellation Risk:</b> {show_risk(risk)}<br>
                <b>Airport Average Risk:</b> {avg_risk:.0%}
                <div class="result-divider"></div>
                <b>✅ Safer Alternative:</b> {safest} ({safest_risk:.0%})
                <div class="result-divider"></div>
                <span style="font-size:11px;color:#888;">Based on 10 years of U.S. BTS data (2015–2025)</span>
            </div>
            """, unsafe_allow_html=True)

# ── AIRLINE TOOL ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🏢 Airline Tool")
    st.markdown('<div class="page-sub">Identify your airline\'s biggest delay driver — based on 10 years of historical patterns (2015–2025)</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        sel_airline2 = st.selectbox("Airline", airline_list, key="at_al")
    with col2:
        sel_month2 = st.selectbox("Month", month_list, key="at_mo")

    st.markdown('<div class="historical-note">Based on historical patterns from 2015–2025 BTS data — not real-time predictions</div>', unsafe_allow_html=True)

    if st.button("Analyse Historical Delays", key="at_btn"):
        m2 = get_month_number(sel_month2)
        filtered2 = data[(data['carrier_name']==sel_airline2)&(data['month']==m2)]

        if filtered2.empty:
            st.warning("No data found.")
        else:
            dcols   = ['carrier_delay','weather_delay','nas_delay','late_aircraft_delay']
            dlabels = ['Carrier Operations','Weather','NAS / Air Traffic Control','Late Arriving Aircraft']
            totals  = filtered2[dcols].sum()
            total   = totals.sum()
            biggest = dlabels[totals.values.argmax()]

            rows = ""
            for label,col in zip(dlabels,dcols):
                pct = totals[col]/total*100
                bar = "█"*int(pct/5)
                rows += f"<tr><td style='padding:6px 8px 6px 0;'>{label}</td><td style='padding:6px 8px;'>{totals[col]:,.0f} mins</td><td style='padding:6px 0;'><b>{pct:.0f}%</b> <span style='color:#555;'>{bar}</span></td></tr>"

            st.markdown(f"""
            <div class="result-box">
                <b>Airline:</b> {sel_airline2} &nbsp;|&nbsp; <b>Month:</b> {sel_month2}
                <div class="result-divider"></div>
                <table width="100%" style="color:#fff;border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #333;color:#aaa;">
                        <th align="left" style="padding:4px 8px 6px 0;font-weight:400;font-size:12px;">Delay Cause</th>
                        <th align="left" style="padding:4px 8px 6px;font-weight:400;font-size:12px;">Minutes</th>
                        <th align="left" style="padding:4px 0 6px;font-weight:400;font-size:12px;">Share</th>
                    </tr>
                    {rows}
                </table>
                <div class="result-divider"></div>
                <b>🔴 Biggest Delay Driver: {biggest}</b><br>
                <span style="font-size:11px;color:#888;">Based on 10 years of U.S. BTS data (2015–2025)</span>
            </div>
            """, unsafe_allow_html=True)

# ── RECOMMENDATION ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🏆 Recommendation System")
    st.markdown('<div class="page-sub">Find the safest airline at any airport — based on 10 years of historical patterns (2015–2025)</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        sel_airport_name3 = st.selectbox("Airport", airport_full_list, key="rs_ap")
        sel_airport3 = get_code(sel_airport_name3)
    with col2:
        sel_month3 = st.selectbox("Month", month_list, key="rs_mo")

    st.markdown('<div class="historical-note">Based on historical patterns from 2015–2025 BTS data — not real-time predictions</div>', unsafe_allow_html=True)

    if st.button("Find Safest Airlines", key="rs_btn"):
        m3 = get_month_number(sel_month3)
        filtered3 = data[(data['airport']==sel_airport3)&(data['month']==m3)].dropna(subset=features).copy()

        if filtered3.empty:
            st.warning("No data found.")
        else:
            filtered3['risk'] = model.predict_proba(filtered3[features])[:,1]
            ranked = filtered3.groupby('carrier_name')['risk'].mean().sort_values()
            top3   = ranked.head(3)
            worst  = ranked.index[-1]
            worst_risk = ranked.iloc[-1]

            medals = ["🥇","🥈","🥉"]
            top3_rows = ""
            for i,(name,val) in enumerate(top3.items()):
                top3_rows += f"<tr><td style='padding:6px 0;'>{medals[i]} {name}</td><td style='padding:6px 0;text-align:right;'><span class='risk-low'>{val:.0%}</span></td></tr>"

            st.markdown(f"""
            <div class="result-box">
                <b>Airport:</b> {sel_airport_name3} &nbsp;|&nbsp; <b>Month:</b> {sel_month3}
                <div class="result-divider"></div>
                <div style="font-size:11px;color:#aaa;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Top 3 safest airlines</div>
                <table width="100%" style="color:#fff;border-collapse:collapse;">{top3_rows}</table>
                <div class="result-divider"></div>
                <b>⚠️ Highest Risk Airline:</b><br>
                <span style="color:#ff6b6b;">{worst} — {worst_risk:.0%}</span><br>
                <span style="font-size:11px;color:#888;">Based on 10 years of U.S. BTS data (2015–2025)</span>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<div class="footer-text">AeroRight &nbsp;|&nbsp; Sai Charan Raju Bollepalli &nbsp;|&nbsp; Business Analytics Capstone — Group 4 &nbsp;|&nbsp; U.S. BTS Data 2015–2025</div>', unsafe_allow_html=True)
