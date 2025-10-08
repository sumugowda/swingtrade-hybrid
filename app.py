import os, json, datetime as dt
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests


import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- CONFIG ----------------
SHEET_NAME = "Swing_Trader_Signals"

DEFAULT_SYMBOLS = [
    "ABFRL","ADANIENT","ADANIPORTS","AMBUJACEM","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BANKBARODA","BEL","BERGEPAINT","BHARTIARTL",
    "BIOCON","BOSCHLTD","BPCL","CANBK","CHOLAFIN","CIPLA","COALINDIA","COLPAL","DRREDDY",
    "EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO",
    "HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC","JSWSTEEL","KOTAKBANK","LT",
    "M&M","MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE","SBIN",
    "SHRIRAMFIN","SUNPHARMA","TATACONSUM","TATAMOTORS","TATASTEEL","TCS","TECHM","TITAN",
    "TRENT","ULTRACEMCO","WIPRO"
]

def to_yahoo(sym: str) -> str:
    return f"{sym}.NS"

# ---------------- GOOGLE SHEET CLIENT ----------------
def gs_client():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        json.load(open("service_account.json")), scope
    )
    return gspread.authorize(creds)

def write_signal_rows(rows):
    gc = gs_client()
    ws = gc.open(SHEET_NAME).worksheet("signals")
    ws.append_rows(rows, value_input_option="USER_ENTERED")

def write_journal_rows(rows):
    gc = gs_client()
    ws = gc.open(SHEET_NAME).worksheet("journal")
    ws.append_rows(rows, value_input_option="USER_ENTERED")

# ---------------- DATA LOADER ----------------



import os
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_history_local(folder="data"):
    frames = []
    if not os.path.exists(folder):
        st.error(f"Data folder '{folder}' not found.")
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])

    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    for file in files:
        try:
            path = os.path.join(folder, file)
            df = pd.read_csv(path)

            # 1) normalize headers
            df.columns = [c.strip().lower() for c in df.columns]

            # 2) validate required columns
            required = ["date","open","high","low","close","volume"]
            if not all(col in df.columns for col in required):
                st.warning(f"Skipping {file}: missing one of {required}")
                continue

            # 3) symbol
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            else:
                df["symbol"] = file.rsplit(".", 1)[0].upper()

            # 4) robust datetime parsing:
            #    - accept strings with timezone (e.g., '+05:30')
            #    - coerce invalid rows to NaT
            #    - remove timezone (make naive) so all files match
            d = pd.to_datetime(df["date"], utc=True, errors="coerce")
            d = d.dt.tz_localize(None)  # drop UTC tz info → naive datetime
            df["date"] = d
            df = df.dropna(subset=["date"]).sort_values("date")

            # 5) keep consistent subset
            df = df[["date","symbol","open","high","low","close","volume"]]
            frames.append(df)

        except Exception as e:
            st.warning(f"Error loading {file}: {e}")

    if not frames:
        st.error("No valid data files found.")
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])

    # Concatenate WITHOUT re-parsing date again (it’s already normalized)
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["symbol","date"], ignore_index=True)
    return out



# ---------------- STRATEGY ENGINE ----------------
def compute_signals(prices: pd.DataFrame, dist52_max=10.0,
                    bo_len=20, atr_len=14, ema_fast=20, ema_slow=50):
    out_rows = []
    for sym, df in prices.groupby("symbol"):
        df = df.copy().sort_values("date")
        df["ema20"] = ta.ema(df["close"], ema_fast)
        df["ema50"] = ta.ema(df["close"], ema_slow)
        df["atr14"] = ta.atr(df["high"], df["low"], df["close"], atr_len)
        df["hh20"]  = pd.Series(df["high"]).rolling(bo_len).max().shift(1)
        hh_52       = pd.Series(df["high"]).rolling(252).max()
        df["dist52"] = (hh_52 - df["close"]) / hh_52.replace(0,np.nan) * 100

        # Momentum in trend
        mom_cond = (df["close"] > df["hh20"]) & (df["ema20"] > df["ema50"]) & \
                   (df["close"] > df["ema50"]) & (df["dist52"] <= dist52_max)

        # PEAD proxy
        volma20 = df["volume"].rolling(20).mean()
        up_gap  = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100 >= 3.0
        vol_spk = df["volume"] > volma20 * 1.8
        pead_up = up_gap & vol_spk
        event_bar = pead_up
        event_high = df["high"].where(event_bar).ffill()
        ema10 = ta.ema(df["close"], 10)
        since = (~event_bar).groupby(event_bar.cumsum()).cumcount()
        in_window = since <= 3
        pead_long = in_window & pead_up & ((df["close"] > event_high) | ((df["low"] <= ema10) & (df["close"] > ema10)))

        last = df.iloc[-1]
        asof = str(df["date"].iloc[-1].date())

        if bool(mom_cond.iloc[-1]):
            entry = float(last["close"]); stop = float(entry - 2.0*last["atr14"])
            out_rows.append([asof, sym, "MomentumTrend", "BUY", entry, stop, "ATRtrail_or_Time", 0.70, "EMA20>EMA50 + 20d breakout", "NEW"])

        if bool(pead_long.iloc[-1]):
            entry = float(last["close"]); stop = float(entry - 2.0*last["atr14"])
            out_rows.append([asof, sym, "PEAD", "BUY", entry, stop, "ATRtrail_or_Time", 0.65, "Gap+VolUp continuation", "NEW"])
    return out_rows

import pandas_ta as ta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- Compute technical indicators and hybrid signals ---
def compute_signals(df: pd.DataFrame):
    out = []
    today = datetime.now().strftime("%Y-%m-%d")

    for sym, data in df.groupby("symbol"):
        data = data.copy().sort_values("date")
        if len(data) < 60:
            continue

        data["ema20"] = ta.ema(data["close"], length=20)
        data["ema50"] = ta.ema(data["close"], length=50)
        data["rsi"] = ta.rsi(data["close"], length=14)

        # PEAD proxy: recent 10-day return vs prior 10-day return
        data["drift"] = data["close"].pct_change(10) * 100
        latest = data.iloc[-1]

        # --- Rule engine ---
        ema_signal = "BUY" if latest["ema20"] > latest["ema50"] else "SELL"
        drift_signal = "BUY" if latest["drift"] > 0 else "SELL"
        rsi_signal = "BUY" if latest["rsi"] > 55 else ("SELL" if latest["rsi"] < 45 else "HOLD")

        # Weighted consensus
        signals = [ema_signal, drift_signal, rsi_signal]
        action = max(set(signals), key=signals.count)
        confidence = round(sum([s == action for s in signals]) / len(signals), 2)

        note = f"EMA20 {ema_signal}, Drift {drift_signal}, RSI {rsi_signal}"
        entry = round(float(latest['close']), 2)
        stop = round(entry * (0.97 if action == "BUY" else 1.03), 2)
        target = round(entry * (1.03 if action == "BUY" else 0.97), 2)

        out.append({
            "asof": today,
            "symbol": sym,
            "strategy": "Momentum+PEAD",
            "action": action,
            "entry": entry,
            "stop": stop,
            "target": target,
            "confidence": confidence,
            "notes": note,
            "status": "Active"
        })
    return pd.DataFrame(out)


# --- Export to Google Sheets ---
def export_to_sheets(df: pd.DataFrame, sheet_name="Swing_Trader_Signals", tab="signals"):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(creds)

        ws = client.open(sheet_name).worksheet(tab)
        ws.clear()

        ws.append_row(list(df.columns))
        ws.append_rows(df.values.tolist())
        st.success(f"✅ Exported {len(df)} signals to Google Sheet → '{sheet_name}' / '{tab}'")
    except Exception as e:
        st.error(f"❌ Failed to export to Google Sheets: {e}")

# ---------------- UI ----------------
st.set_page_config(page_title="SwingTrade Hybrid", layout="wide")
st.title("SwingTrade — Hybrid (Momentum + PEAD)")

with st.sidebar:
    st.markdown("### 1️⃣ Choose Symbols")
    symbols_text = st.text_area("NSE symbols (comma separated)", value=",".join(DEFAULT_SYMBOLS), height=120)
    symbols = [s.strip() for s in symbols_text.split(",") if s.strip()]
    st.markdown("---")
    if st.button("Fetch / Refresh Prices"):
        with st.spinner("Loading local data..."):
            hist = load_history_local("data")
        st.session_state["prices"] = hist
        st.success(f"Loaded {len(hist)} rows for {len(hist['symbol'].unique())} symbols.")

    if st.button("Generate Signals"):
        prices = st.session_state.get("prices")
        if prices is None or prices.empty:
            st.error("Load prices first.")
        else:
            rows = compute_signals(prices)
            if rows:
                write_signal_rows(rows)
                st.success(f"Wrote {len(rows)} signals to Google Sheet → 'signals' tab.")
            else:
                st.info("No fresh signals today.")
    if st.button("📤 Generate & Export Signals"):
        prices = st.session_state.get("prices")
        if prices is None or prices.empty:
            st.warning("No price data loaded yet.")
        else:
            with st.spinner("Computing hybrid signals..."):
                sig_df = compute_signals(prices)
                if sig_df.empty:
                    st.warning("No signals computed — check data.")
                else:
                    st.dataframe(sig_df.head(10))
                    export_to_sheets(sig_df)

# Chart
prices = st.session_state.get("prices")
if prices is not None and not prices.empty:
    symbols_avail = sorted(prices["symbol"].unique().tolist())
    sel = st.selectbox("Chart Symbol", symbols_avail, index=0)
    df = prices[prices["symbol"]==sel].copy().sort_values("date")
    df["ema20"] = ta.ema(df["close"], 20)
    df["ema50"] = ta.ema(df["close"], 50)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema50"], name="EMA50"))
    st.plotly_chart(fig, use_container_width=True)
