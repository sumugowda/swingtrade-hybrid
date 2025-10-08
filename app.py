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



@st.cache_data(show_spinner=False)
def load_history_local(folder="data"):
    frames = []
    if not os.path.exists(folder):
        st.error(f"Data folder '{folder}' not found.")
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])

    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    for file in files:
        try:
            path = os.path.join(folder, file)
            df = pd.read_csv(path)

            # normalize headers
            df.columns = [c.strip().lower() for c in df.columns]

            # skip files missing essential columns
            required = ["date", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                st.warning(f"Skipping {file}: missing one of {required}")
                continue

            # clean and type-cast
            df["symbol"] = (
                df["symbol"].astype(str).str.strip().str.upper()
                if "symbol" in df.columns
                else file.replace(".csv", "").upper()
            )

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.sort_values("date")

            # select consistent subset
            df = df[["date", "symbol", "open", "high", "low", "close", "volume"]]
            frames.append(df)

        except Exception as e:
            st.warning(f"Error loading {file}: {e}")

    if not frames:
        st.error("No valid data files found.")
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])

    # ✅ clean column types explicitly before concat
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"], ignore_index=True)
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
