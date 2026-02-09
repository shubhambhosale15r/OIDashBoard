import streamlit as st
import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel
import plotly.graph_objects as go
from datetime import datetime
import pytz  # Added for timezone handling
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")

# ================= TIMEZONE CONFIG =================
IST = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

def get_current_ist():
    """Get current time in IST"""
    return datetime.now(IST)

def convert_to_ist(dt):
    """Convert datetime to IST"""
    if dt.tzinfo is None:
        # If naive datetime, assume it's IST
        return IST.localize(dt)
    else:
        # Convert from any timezone to IST
        return dt.astimezone(IST)

# ================= CONFIG PANEL =================
st.sidebar.header("⚙️ Dashboard Settings")

CLIENT_ID = st.sidebar.text_input("Fyers Client ID")
ACCESS_TOKEN = st.sidebar.text_input("Access Token", type="password")
SYMBOL = st.sidebar.text_input("Symbol", value="NSE:NIFTY50-INDEX")

STRIKE_COUNT = st.sidebar.number_input(
    "Strike Count", 5, 50, 40
)

MAX_HISTORY = st.sidebar.number_input(
    "History Points", 50, 1000, 1000
)

REFRESH_MS = st.sidebar.number_input(
    "Refresh Interval (ms)", 5000, 600000, 300000
)

# ----- Validate Credentials -----
if CLIENT_ID == "" or ACCESS_TOKEN == "":
    st.warning("Enter Fyers credentials in sidebar")
    st.stop()

# ----- Auto Refresh -----
st_autorefresh(interval=int(REFRESH_MS), key="refresh")

# ----- Fyers Init -----
fyers = fyersModel.FyersModel(
    client_id=CLIENT_ID,
    token=ACCESS_TOKEN,
    is_async=False
)

st.title("🏦 Dealer Positioning Dashboard")

# Initialize history if not exists
if "history" not in st.session_state:
    st.session_state.history = []

# ================= FETCH =================
def fetch_chain():
    try:
        res = fyers.optionchain({
            "symbol": SYMBOL,
            "strikecount": STRIKE_COUNT,
            "timestamp": ""
        })
        return pd.DataFrame(res["data"]["optionsChain"])
    except:
        return None


# ================= SPOT =================
def get_spot(df):
    return float(df[df["option_type"] == ""].iloc[0]["ltp"])


# ================= BUILD TABLE =================
def build_table(df):

    opt = df[df["option_type"] != ""].copy()

    ce = opt[opt["option_type"] == "CE"]
    pe = opt[opt["option_type"] == "PE"]

    merged = pd.merge(
        ce[["strike_price", "oich"]],
        pe[["strike_price", "oich"]],
        on="strike_price",
        suffixes=("_CE", "_PE")
    )

    # ---------- FLOW SEPARATION ----------
    merged["Build_CE"] = merged["oich_CE"].clip(lower=0)
    merged["Build_PE"] = merged["oich_PE"].clip(lower=0)

    merged["Unwind_CE"] = merged["oich_CE"].clip(upper=0)
    merged["Unwind_PE"] = merged["oich_PE"].clip(upper=0)

    return merged.sort_values("strike_price")


# ================= FLOW MP =================
def calc_flow_mp(opt_df):

    strikes = opt_df["strike_price"].values
    ce_flow = opt_df["Build_CE"].values
    pe_flow = opt_df["Build_PE"].values

    pain = []

    for test in strikes:
        call = np.sum(ce_flow * np.maximum(0, test - strikes))
        put  = np.sum(pe_flow * np.maximum(0, strikes - test))
        pain.append(call + put)

    return strikes[np.argmin(pain)]


# ================= DEALER ZONES =================
def dealer_zones(opt_df):

    support = opt_df.loc[opt_df["Build_PE"].idxmax(), "strike_price"]
    resistance = opt_df.loc[opt_df["Build_CE"].idxmax(), "strike_price"]

    return support, resistance


# ================= SNAPSHOT =================
def positioning_snapshot(spot, mp, net_build, velocity):

    comfort = "Above Comfort Zone" if spot > mp else "Below Comfort Zone"

    if net_build > 0:
        flow_bias = "Put Writing Dominant"
    elif net_build < 0:
        flow_bias = "Call Writing Dominant"
    else:
        flow_bias = "Balanced"

    if abs(velocity) > 50:
        speed = "Aggressive Reposition"
    elif abs(velocity) > 10:
        speed = "Moderate"
    else:
        speed = "Passive"

    return comfort, flow_bias, speed


# ================= MAIN =================
df = fetch_chain()

if df is None:
    st.error("Chain fetch failed")
    st.stop()

spot = get_spot(df)
opt_df = build_table(df)

# ----- FLOW MP -----
mp = calc_flow_mp(opt_df)
support, resistance = dealer_zones(opt_df)

# ----- NET BUILD FLOW -----
net_build = opt_df["Build_PE"].sum() - opt_df["Build_CE"].sum()

# ----- VELOCITY -----
if len(st.session_state.history) > 0:
    prev_mp = st.session_state.history[-1]["mp"]
    velocity = mp - prev_mp
else:
    velocity = 0

distance = spot - mp

# ----- STORE HISTORY -----
# Store timestamp in IST
current_time_ist = get_current_ist()
st.session_state.history.append({
    "time": current_time_ist,  # Now in IST
    "spot": spot,
    "mp": mp,
    "velocity": velocity,
    "distance": distance,
    "net_build": net_build
})

st.session_state.history = st.session_state.history[-MAX_HISTORY:]
hist = pd.DataFrame(st.session_state.history)

# ===== LAST REFRESH DISPLAY =====
# NOW display the refresh time AFTER updating history
if len(st.session_state.history) > 0:
    last_refresh = st.session_state.history[-1]["time"]
    # Ensure time is in IST
    last_refresh_ist = convert_to_ist(last_refresh)
    col1, col2 = st.columns(2) 
    col1.caption(f"🕒 Last Refresh (IST): {last_refresh_ist.strftime('%H:%M:%S')}")
    col2.caption(f"🔄 Auto Refresh: {REFRESH_MS//1000} sec")
else:
    # This should never happen now, but keep as fallback
    col1, col2 = st.columns(2)
    col1.caption("🕒 Last Refresh (IST): Just fetched data...")
    col2.caption(f"🔄 Auto Refresh: {REFRESH_MS//1000} sec")

# Display current IST time
current_ist = get_current_ist()
st.sidebar.caption(f"🕐 Current IST Time: {current_ist.strftime('%H:%M:%S')}")

# ================= SNAPSHOT PANEL =================
comfort, flow_bias, speed = positioning_snapshot(
    spot, mp, net_build, velocity
)

c1, c2, c3, c4 = st.columns(4)

price_status = "🟢 Above MP" if spot > mp else "🔴 Below MP"

c1.metric("Underlying LTP", round(spot, 2), price_status)
c2.metric("Comfort Level", comfort)
c3.metric("Flow Bias", flow_bias)
c4.metric("Reposition Speed", speed)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Dealer Support", support)
c2.metric("Dealer Resistance", resistance)
c3.metric("Spot - MP Distance", round(distance, 2))
c4.metric("Flow Max Pain", mp)


st.divider()

# ================= CHART =================
fig = go.Figure()

# Ensure times in history are displayed in IST format
hist_display = hist.copy()
# Convert times to string format for display
hist_display["time_display"] = hist_display["time"].apply(
    lambda x: convert_to_ist(x).strftime('%H:%M:%S')
)

fig.add_trace(go.Scatter(x=hist["time"], y=hist["spot"], name="Spot"))
fig.add_trace(go.Scatter(x=hist["time"], y=hist["mp"], name="Flow MP"))

fig.add_hline(y=support, line_dash="dot")
fig.add_hline(y=resistance, line_dash="dot")

# Update layout with IST timezone info
fig.update_layout(
    title=f"Spot: {round(spot,2)} | Flow MP: {mp} (IST Timezone)",
    xaxis_title="Time (IST)",
    xaxis=dict(
        tickformat='%H:%M:%S',
        title="Time (IST)"
    )
)
fig.update_layout(template="plotly_dark")

# Updated: use width='stretch' instead of use_container_width=True
st.plotly_chart(fig, width='stretch')

# ================= LADDER =================
st.subheader("🪜 Strike Flow Ladder")

ladder_df = opt_df.copy()

ladder_df["Net_Build"] = ladder_df["Build_PE"] - ladder_df["Build_CE"]

# Updated: use width='stretch' instead of use_container_width=True
st.dataframe(
    ladder_df[
        ["strike_price",
         "Build_CE",
         "Build_PE",
         "Unwind_CE",
         "Unwind_PE",
         "Net_Build"]
    ].sort_values("strike_price", ascending=False),
    width='stretch'
)


