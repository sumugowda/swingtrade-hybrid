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

# ------------------------------------------------------------
# 5️⃣ TRADE JOURNAL UPDATER  (Agent 4)
# ------------------------------------------------------------

def append_trade_to_journal(opened_at, symbol, strategy, entry, stop, risk_pct, notes):
    """Add a new open trade to journal.csv"""
    os.makedirs("outputs", exist_ok=True)
    journal_path = "outputs/journal.csv"

    if not os.path.exists(journal_path):
        init_journal_file()

    new_row = pd.DataFrame([{
        "opened_at": opened_at,
        "symbol": symbol,
        "strategy": strategy,
        "entry": entry,
        "stop": stop,
        "risk_pct": risk_pct,
        "notes": notes,
        "closed_at": "",
        "exit_price": "",
        "pnl": ""
    }])

    journal = pd.read_csv(journal_path)
    journal = pd.concat([journal, new_row], ignore_index=True)
    journal.to_csv(journal_path, index=False)
    st.success(f"✅ Trade added for {symbol}")


def close_trade(symbol, exit_price):
    """Close an existing open trade and compute PnL"""
    journal_path = "outputs/journal.csv"
    if not os.path.exists(journal_path):
        st.error("No journal.csv found yet.")
        return

    journal = pd.read_csv(journal_path)
    if symbol not in journal["symbol"].values:
        st.warning(f"No open trade found for {symbol}.")
        return

    # locate latest open trade
    idx = journal[journal["symbol"] == symbol].last_valid_index()
    row = journal.loc[idx]

    try:
        entry = float(row["entry"])
        pnl = round(((float(exit_price) - entry) / entry) * 100, 2)
        journal.at[idx, "closed_at"] = datetime.now().strftime("%Y-%m-%d")
        journal.at[idx, "exit_price"] = float(exit_price)
        journal.at[idx, "pnl"] = pnl
        journal.to_csv(journal_path, index=False)
        st.success(f"✅ {symbol} trade closed with {pnl}% P/L")
    except Exception as e:
        st.error(f"Error closing trade: {e}")


# ------------------------------------------------------------
# 6️⃣ JOURNAL MANAGEMENT UI
# ------------------------------------------------------------

st.markdown("### 🗒️ Trade Journal Manager")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ➕ Add New Trade")
    symbol = st.text_input("Symbol (e.g. INFY)")
    strategy = st.text_input("Strategy", value="Momentum+PEAD")
    entry = st.number_input("Entry Price", min_value=0.0, step=0.1)
    stop = st.number_input("Stop Loss", min_value=0.0, step=0.1)
    risk_pct = st.number_input("Risk %", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    notes = st.text_area("Notes")
    if st.button("💾 Add Trade"):
        if symbol and entry > 0:
            append_trade_to_journal(datetime.now().strftime("%Y-%m-%d"), symbol, strategy, entry, stop, risk_pct, notes)
        else:
            st.warning("Please enter symbol and entry price.")

with col2:
    st.markdown("#### ✅ Close Trade")
    close_symbol = st.text_input("Symbol to close")
    exit_price = st.number_input("Exit Price", min_value=0.0, step=0.1)
    if st.button("📈 Close Trade"):
        if close_symbol and exit_price > 0:
            close_trade(close_symbol, exit_price)
        else:
            st.warning("Please enter symbol and exit price.")


# ------------------------------------------------------------
# 7️⃣ Display current journal
# ------------------------------------------------------------
journal_path = "outputs/journal.csv"
if os.path.exists(journal_path):
    st.markdown("#### 📘 Current Journal Entries")
    journal = pd.read_csv(journal_path)
    st.dataframe(journal)
else:
    st.info("No journal entries yet.")
