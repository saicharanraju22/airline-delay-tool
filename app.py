import streamlit as st
import pandas as pd
import joblib

# page setup
st.set_page_config(
    page_title="AeroRight",
    page_icon="✈️",
    layout="wide"
)

# styling
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #000000; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    h1, h2, h3 { color: #000000 !important; }

    .stSelectbox > div > div {
        border-radius: 0px !important;
        border: 1.5px solid #000000 !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stButton > button {
        border-radius: 0px !important;
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 500;
        width: 100%;
        padding: 10px 16px;
        font-size: 14px;
        text-align: left !important;
    }
    .stButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }

    /* active nav button */
    .nav-active > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-left: 4px solid #ffffff !important;
        font-weight: 700 !important;
    }

    .stat-card {
        border: 1.5px solid #000000;
        padding: 12px 16px;
        background: #ffffff;
    }
    .stat-label { font-size: 11px; color: #666666; margin-bottom: 4px; }
    .stat-value { font-size: 22px; font-weight: 500; color: #000000; }
    .finding-card {
        border: 1.5px solid #000000;
        padding: 12px 16px;
        background: #ffffff;
    }
    .finding-title { font-size: 11px; color: #666666; margin-bottom: 4px; }
    .finding-value { font-size: 14px; font-weight: 500; color: #000000; }
    .finding-reason { font-size: 11px; color: #888888; margin-top: 4px; line-height: 1.4; }
    .result-box {
        background-color: #000000;
        color: #ffffff;
        padding: 20px;
        border: 1.5px solid #000000;
        margin-top: 15px;
        font-size: 14px;
        line-height: 1.9;
    }
    .risk-high { color: #ff6b6b; font-weight: bold; font-size: 1.1em; }
    .risk-low  { color: #69db7c; font-weight: bold; font-size: 1.1em; }
    .risk-mid  { color: #ffd43b; font-weight: bold; font-size: 1.1em; }
    .cta-box {
        background: #000000;
        color: #ffffff;
        padding: 12px 16px;
        font-size: 13px;
    }
    .section-label {
        font-size: 11px;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .result-divider { border-top: 1px solid #333333; margin: 8px 0; }
    .footer-text {
        font-size: 10px;
        color: #aaaaaa;
        text-align: center;
        margin-top: 2rem;
    }
    .page-title { color: #000000 !important; font-size: 26px; font-weight: 500; margin-bottom: 4px; }
    .page-sub { color: #666666; font-size: 14px; margin-bottom: 1.5rem; }
    .historical-note {
        font-size: 11px;
        color: #888888;
        margin-top: 8px;
        font-style: italic;
    }

    /* sidebar nav buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: transparent !important;
        color: #888888 !important;
        border: none !important;
        border-left: 3px solid transparent !important;
        border-radius: 0px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        text-align: left !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# load model and data
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

# features
features = [
    'arr_flights', 'nas_delay', 'security_delay',
    'delay_minutes_per_flight_capped_p999',
    'avg_delay_minutes_given_delayed_capped_p999',
    'month', 'delay_rate', 'weather_pct', 'carrier_pct'
]

# month mapping
month_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

def get_month_number(month_name):
    for num, name in month_map.items():
        if name == month_name:
            return num

def show_risk(prob):
    if prob >= 0.5:
        return f'<span class="risk-high">{prob:.0%} — HIGH RISK ⚠️</span>'
    elif prob >= 0.25:
        return f'<span class="risk-mid">{prob:.0%} — MODERATE RISK</span>'
    else:
        return f'<span class="risk-low">{prob:.0%} — LOW RISK ✅</span>'

# airport lookup from real data
airport_lookup = (
    data[['airport', 'airport_name']]
    .dropna()
    .drop_duplicates()
    .set_index('airport')['airport_name']
    .to_dict()
)

airline_list      = sorted(data['carrier_name'].dropna().unique().tolist())
airport_codes     = sorted(data['airport'].dropna().unique().tolist())
airport_full_list = sorted([airport_lookup.get(c, c) for c in airport_codes])
month_list        = list(month_map.values())
name_to_code      = {v: k for k, v in airport_lookup.items()}

def get_code(full_name):
    return name_to_code.get(full_name, full_name)

# ── SESSION STATE FOR NAVIGATION ───────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ AeroRight")
    st.markdown("<small style='color:#888;'>Right choice for your travel</small>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:11px; color:#555; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:4px;'>Menu</div>", unsafe_allow_html=True)

    if st.button("🏠  Home"):
        st.session_state.page = "Home"
    if st.button("🧳  Customer Tool"):
        st.session_state.page = 'Customer Tool'
    if st.button("🏢  Airline Tool"):
        st.session_state.page = 'Airline Tool'
    if st.button("🏆  Recommendation System"):
        st.session_state.page = 'Recommendation System'

page = st.session_state.page

# ── HOME PAGE ──────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown('<div class="page-title">Before you book your next flight</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Here is what 10 years of U.S. airline data tells you</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="stat-card"><div class="stat-label">Records analysed</div><div class="stat-value">224K</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="stat-card"><div class="stat-label">Airlines covered</div><div class="stat-value">21</div></div>', unsafe_allow_html=True)
    c3.markdown('<div class="stat-card"><div class="stat-label">Airports covered</div><div class="stat-value">300+</div></div>', unsafe_allow_html=True)
    c4.markdown('<div class="stat-card"><div class="stat-label">Model accuracy</div><div class="stat-value">83%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Key findings from 10 years of data</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    f1.markdown('<div class="finding-card"><div class="finding-title">Best airline (2015–2025)</div><div class="finding-value">Delta Air Lines — 0.1%</div><div class="finding-reason">Lowest cancellation rate across 10 years of data</div></div>', unsafe_allow_html=True)
    f2.markdown('<div class="finding-card"><div class="finding-title">Worst airline (2015–2025)</div><div class="finding-value">Peninsula Airways — 15.6%</div><div class="finding-reason">1 in 6 flights gets cancelled — avoid if possible</div></div>', unsafe_allow_html=True)
    f3.markdown('<div class="finding-card"><div class="finding-title">Best managed airport</div><div class="finding-value">Atlanta (ATL) — lowest cancellation</div><div class="finding-reason">Handles the most flights yet stays lowest risk</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    f4, f5, f6 = st.columns(3)
    f4.markdown('<div class="finding-card"><div class="finding-title">Most dangerous month to fly</div><div class="finding-value">April — 3.5% cancellation</div><div class="finding-reason">Spring storms cause the highest disruptions of the year</div></div>', unsafe_allow_html=True)
    f5.markdown('<div class="finding-card"><div class="finding-title">Safest month to fly</div><div class="finding-value">November — lowest risk</div><div class="finding-reason">Best time to book — consistently lowest cancellation rate</div></div>', unsafe_allow_html=True)
    f6.markdown('<div class="finding-card"><div class="finding-title">Biggest delay cause</div><div class="finding-value">Carrier operations — 39%</div><div class="finding-reason">Most delays are the airline\'s fault — not weather</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="cta-box">Want to check YOUR specific flight? Use the tools in the sidebar →</div>', unsafe_allow_html=True)

# ── CUSTOMER TOOL ──────────────────────────────────────────────────────────────
elif page == "Customer Tool":
    st.markdown('<div class="page-title">🧳 Customer Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Check your flight\'s cancellation risk — based on 10 years of historical patterns (2015–2025)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_airport_name = st.selectbox("Airport", airport_full_list)
        selected_airport = get_code(selected_airport_name)
    with col2:
        airlines_at_airport = sorted(data[data['airport'] == selected_airport]['carrier_name'].dropna().unique().tolist())
        selected_airline = st.selectbox("Airline", airlines_at_airport)
    with col3:
        selected_month = st.selectbox("Month", month_list)

    st.markdown('<div class="historical-note">Results are based on historical patterns from 2015–2025 BTS data — not real-time predictions</div>', unsafe_allow_html=True)

    if st.button("Check Historical Risk", key="ct_btn"):
        month_num = get_month_number(selected_month)

        filtered = data[
            (data['carrier_name'] == selected_airline) &
            (data['airport'] == selected_airport) &
            (data['month'] == month_num)
        ].dropna(subset=features)

        if filtered.empty:
            st.warning("No data found for this combination.")
        else:
            cancel_risk  = model.predict_proba(filtered[features])[:, 1].mean()
            all_airlines = data[(data['airport'] == selected_airport) & (data['month'] == month_num)].dropna(subset=features).copy()
            all_airlines['risk'] = model.predict_proba(all_airlines[features])[:, 1]
            avg_risk       = all_airlines['risk'].mean()
            safest_airline = all_airlines.groupby('carrier_name')['risk'].mean().idxmin()
            safest_risk    = all_airlines.groupby('carrier_name')['risk'].mean().min()

            st.markdown(f"""
            <div class="result-box">
                <b>Airline:</b> {selected_airline}<br>
                <b>Airport:</b> {selected_airport_name} &nbsp;|&nbsp; <b>Month:</b> {selected_month}
                <div class="result-divider"></div>
                <b>Historical Cancellation Risk:</b> {show_risk(cancel_risk)}<br>
                <b>Airport Average Risk:</b> {avg_risk:.0%}
                <div class="result-divider"></div>
                <b>✅ Safer Alternative:</b> {safest_airline} &nbsp;({safest_risk:.0%})<br>
                <div class="result-divider"></div>
                <span style="font-size:11px; color:#888;">Based on 10 years of U.S. BTS data (2015–2025)</span>
            </div>
            """, unsafe_allow_html=True)

# ── AIRLINE TOOL ───────────────────────────────────────────────────────────────
elif page == "Airline Tool":
    st.markdown('<div class="page-title">🏢 Airline Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Identify your airline\'s biggest delay driver — based on 10 years of historical patterns (2015–2025)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_airline2 = st.selectbox("Airline", airline_list)
    with col2:
        selected_month2 = st.selectbox("Month", month_list)

    st.markdown('<div class="historical-note">Results are based on historical patterns from 2015–2025 BTS data — not real-time predictions</div>', unsafe_allow_html=True)

    if st.button("Analyse Historical Delays", key="at_btn"):
        month_num2 = get_month_number(selected_month2)
        filtered2  = data[(data['carrier_name'] == selected_airline2) & (data['month'] == month_num2)]

        if filtered2.empty:
            st.warning("No data found for this airline and month.")
        else:
            delay_columns = ['carrier_delay', 'weather_delay', 'nas_delay', 'late_aircraft_delay']
            delay_labels  = ['Carrier Operations', 'Weather', 'NAS / Air Traffic Control', 'Late Arriving Aircraft']
            delay_totals  = filtered2[delay_columns].sum()
            grand_total   = delay_totals.sum()
            biggest_cause = delay_labels[delay_totals.values.argmax()]

            table_rows = ""
            for label, col in zip(delay_labels, delay_columns):
                pct = delay_totals[col] / grand_total * 100
                bar = "█" * int(pct / 5)
                table_rows += f"<tr><td style='padding:6px 8px 6px 0;'>{label}</td><td style='padding:6px 8px;'>{delay_totals[col]:,.0f} mins</td><td style='padding:6px 0;'><b>{pct:.0f}%</b> <span style='color:#555;'>{bar}</span></td></tr>"

            st.markdown(f"""
            <div class="result-box">
                <b>Airline:</b> {selected_airline2} &nbsp;|&nbsp; <b>Month:</b> {selected_month2}
                <div class="result-divider"></div>
                <table width="100%" style="color:#fff; border-collapse:collapse; margin-top:4px;">
                    <tr style="border-bottom:1px solid #333; color:#aaa;">
                        <th align="left" style="padding:4px 8px 6px 0; font-weight:400; font-size:12px;">Delay Cause</th>
                        <th align="left" style="padding:4px 8px 6px; font-weight:400; font-size:12px;">Minutes</th>
                        <th align="left" style="padding:4px 0 6px; font-weight:400; font-size:12px;">Share</th>
                    </tr>
                    {table_rows}
                </table>
                <div class="result-divider"></div>
                <b>🔴 Biggest Delay Driver: {biggest_cause}</b><br>
                <span style="font-size:11px; color:#888;">Based on 10 years of U.S. BTS data (2015–2025)</span>
            </div>
            """, unsafe_allow_html=True)

# ── RECOMMENDATION SYSTEM ──────────────────────────────────────────────────────
elif page == "Recommendation System":
    st.markdown('<div class="page-title">🏆 Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Find the safest airline at any airport — based on 10 years of historical patterns (2015–2025)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        selected_airport_name3 = st.selectbox("Airport", airport_full_list)
        selected_airport3 = get_code(selected_airport_name3)
    with col2:
        selected_month3 = st.selectbox("Month", month_list)

    st.markdown('<div class="historical-note">Results are based on historical patterns from 2015–2025 BTS data — not real-time predictions</div>', unsafe_allow_html=True)

    if st.button("Find Safest Airlines", key="rs_btn"):
        month_num3 = get_month_number(selected_month3)
        filtered3  = data[(data['airport'] == selected_airport3) & (data['month'] == month_num3)].dropna(subset=features).copy()

        if filtered3.empty:
            st.warning("No data found for this airport and month.")
        else:
            filtered3['risk'] = model.predict_proba(filtered3[features])[:, 1]
            ranked        = filtered3.groupby('carrier_name')['risk'].mean().sort_values()
            top3          = ranked.head(3)
            worst_airline = ranked.index[-1]
            worst_risk    = ranked.iloc[-1]

            medals    = ["🥇", "🥈", "🥉"]
            top3_rows = ""
            for i, (airline_name, risk_val) in enumerate(top3.items()):
                top3_rows += f"<tr><td style='padding:6px 0;'>{medals[i]} {airline_name}</td><td style='padding:6px 0; text-align:right;'><span class='risk-low'>{risk_val:.0%}</span></td></tr>"

            st.markdown(f"""
            <div class="result-box">
                <b>Airport:</b> {selected_airport_name3} &nbsp;|&nbsp; <b>Month:</b> {selected_month3}
                <div class="result-divider"></div>
                <div style="font-size:11px; color:#aaa; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px;">Top 3 safest airlines</div>
                <table width="100%" style="color:#fff; border-collapse:collapse;">
                    {top3_rows}
                </table>
                <div class="result-divider"></div>
                <b>⚠️ Highest Risk Airline:</b><br>
                <span style="color:#ff6b6b;">{worst_airline} — {worst_risk:.0%}</span><br>
                <span style="font-size:11px; color:#888;">Based on 10 years of U.S. BTS data (2015–2025)</span>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    AeroRight &nbsp;|&nbsp; Sai Charan Raju Bollepalli &nbsp;|&nbsp; Business Analytics Capstone — Group 4 &nbsp;|&nbsp; U.S. BTS Data 2015–2025
</div>
""", unsafe_allow_html=True)
