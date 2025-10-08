# ============================================================
#  HYBRID TRADER – Momentum + PEAD Swing Signal Generator
# ============================================================

import streamlit as st
import pandas as pd
import pandas_ta as ta
import os
from datetime import datetime

st.set_page_config(page_title="Swing Trader Hybrid Dashboard", layout="wide")

# ------------------------------------------------------------
# 1️⃣ Load historical data from /data (already working earlier)
# ------------------------------------------------------------
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
            df.columns = [c.strip().lower() for c in df.columns]

            # normalize date column
            d = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
            df["date"] = d
            df["symbol"] = file.rsplit(".",1)[0].upper()
            df = df.dropna(subset=["date"])
            df = df[["date","symbol","open","high","low","close","volume"]]
            frames.append(df)
        except Exception as e:
            st.warning(f"Error loading {file}: {e}")

    if not frames:
        st.error("No valid data files found.")
        return pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])

    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str)
    out = out.sort_values(["symbol","date"], ignore_index=True)
    return out


# ------------------------------------------------------------
# 2️⃣ Compute EMA / RSI / Drift and generate hybrid signals
# ------------------------------------------------------------
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
        data["drift"] = data["close"].pct_change(10) * 100
        latest = data.iloc[-1]

        ema_signal = "BUY" if latest["ema20"] > latest["ema50"] else "SELL"
        drift_signal = "BUY" if latest["drift"] > 0 else "SELL"
        rsi_signal = "BUY" if latest["rsi"] > 55 else ("SELL" if latest["rsi"] < 45 else "HOLD")

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


# ------------------------------------------------------------
# 3️⃣ Export signals and journal to CSV for GitHub commit
# ------------------------------------------------------------
def export_to_csv(df: pd.DataFrame, filename="outputs/signals.csv"):
    try:
        os.makedirs("outputs", exist_ok=True)
        df.to_csv(filename, index=False)
        st.success(f"✅ Signals saved to {filename}")
    except Exception as e:
        st.error(f"❌ Failed to save CSV: {e}")

def init_journal_file():
    os.makedirs("outputs", exist_ok=True)
    journal_path = "outputs/journal.csv"
    if not os.path.exists(journal_path):
        cols = ["opened_at","symbol","strategy","entry","stop","risk_pct",
                "notes","closed_at","exit_price","pnl"]
        pd.DataFrame(columns=cols).to_csv(journal_path, index=False)
        st.info("🗒️ Initialized journal.csv file.")


# ------------------------------------------------------------
# 4️⃣ Streamlit UI
# ------------------------------------------------------------
st.title("📊 Swing Trader Hybrid Dashboard")

if "prices" not in st.session_state:
    st.session_state["prices"] = pd.DataFrame()

if st.button("Fetch / Refresh Prices"):
    with st.spinner("Loading local data..."):
        hist = load_history_local("data")
    st.session_state["prices"] = hist
    st.success(f"Loaded {len(hist)} rows for {len(hist['symbol'].unique())} symbols.")

# Placeholder for chart (optional)
# st.line_chart(...)

if st.button("📤 Generate & Save Signals"):
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
                export_to_csv(sig_df)
                init_journal_file()

