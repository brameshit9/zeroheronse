"""
scanner_core.py
════════════════════════════════════════════════════════════════════════
Zero to Hero — Premium Explosion Scanner (NIFTY 50)
Core logic: NSE session/data client, indicators, and the 18-condition
scorer. Deliberately UI-agnostic — no IPython, no matplotlib .show(),
no print-as-progress — so it can be driven by Streamlit, a CLI, a
notebook, or anything else. See app.py for the Streamlit front-end.

DATA SOURCE — NSE India public endpoints (unofficial, subject to change):
    Historical OHLCV : /api/historical/cm/equity   (max ~365d/call)
    Option chain      : /api/option-chain-equities  (real F&O OI)
    Index history     : /api/historicalOR/indicesHistory
    NIFTY 50 list      : archives.nseindia.com CSV

NSE requires a warmed session (cookies from a normal page hit) and
browser-like headers before /api/* will respond, and it rate-limits
scripted access — see NSESession below.
"""

import io
import random
import time
import datetime
import warnings

import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CONFIG  ← mutate this dict (e.g. from a Streamlit sidebar)
#  before calling run_scan()
# ─────────────────────────────────────────────────────────────
CONFIG = {
    "atr_pct_compress_threshold": 1.8,
    "vol_compress_ratio":         0.65,
    "bb_squeeze_percentile":      20,
    "candle_size_threshold":      0.8,
    "price_range_days":           10,

    "oi_build_chg_pct":   5.0,
    "oi_smart_chg_pct":   8.0,
    "price_flat_pct":     2.0,

    "volume_explosion_x":    2.5,
    "atr_expansion_pct":     15,

    "sr_lookback":       60,
    "sr_proximity_pct":  1.5,

    "rsi_low":  38,
    "rsi_high": 60,

    "ema_fast": 9, "ema_mid": 21, "ema_slow": 50,

    "min_score_show": 5,

    # Polite delay between NSE calls (seconds). NSE rate-limits hard —
    # raise this if you see 403s / empty responses.
    "fetch_delay": 1.2,

    "data_months": 6,
}

# ─────────────────────────────────────────────────────────────
#  NSE SESSION — cookie warm-up, headers, retry/backoff
# ─────────────────────────────────────────────────────────────
class NSESession:
    BASE = "https://www.nseindia.com"
    ARCHIVE = "https://archives.nseindia.com"
    HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/option-chain",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self._warm()

    def _warm(self):
        try:
            self.session.get(self.BASE, timeout=10)
            time.sleep(0.4)
            self.session.get(self.BASE + "/option-chain", timeout=10)
        except Exception:
            pass

    def get_json(self, url, retries=3):
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=12)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        pass
                self._warm()
                time.sleep(1.5 + attempt + random.random())
            except Exception:
                time.sleep(1.5 + attempt + random.random())
        return None

    def get_bytes(self, url, retries=2):
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=12)
                if resp.status_code == 200 and resp.content:
                    return resp.content
                self._warm()
                time.sleep(1.0 + attempt)
            except Exception:
                time.sleep(1.0 + attempt)
        return None

# ─────────────────────────────────────────────────────────────
#  UNIVERSE — NIFTY 50
# ─────────────────────────────────────────────────────────────
NIFTY50_FALLBACK = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","ITC","LT","SBIN",
    "BHARTIARTL","AXISBANK","KOTAKBANK","HINDUNILVR","BAJFINANCE","M&M",
    "MARUTI","SUNPHARMA","NTPC","TATAMOTORS","TITAN","ASIANPAINT",
    "ULTRACEMCO","BAJAJFINSV","WIPRO","ADANIENT","POWERGRID","HCLTECH",
    "JSWSTEEL","TATASTEEL","COALINDIA","GRASIM","NESTLEIND","TECHM",
    "INDUSINDBK","HINDALCO","CIPLA","DRREDDY","EICHERMOT","BPCL","ONGC",
    "SBILIFE","HDFCLIFE","BAJAJ-AUTO","BRITANNIA","APOLLOHOSP","DIVISLAB",
    "TATACONSUM","LTIM","ADANIPORTS","SHRIRAMFIN","TRENT",
]

def get_nifty50_tickers(nse):
    """Live NIFTY 50 constituent list from NSE's archive CSV, with a
    static fallback. Returns plain NSE trading symbols (no suffix)."""
    try:
        url = f"{NSESession.ARCHIVE}/content/indices/ind_nifty50list.csv"
        csv_bytes = nse.get_bytes(url)
        if not csv_bytes:
            raise ValueError("empty response")
        df = pd.read_csv(io.BytesIO(csv_bytes))
        col = "Symbol" if "Symbol" in df.columns else df.columns[2]
        symbols = sorted(set(df[col].astype(str).str.strip().tolist()))
        if len(symbols) >= 45:
            return symbols, True
        raise ValueError("Parsed list looked too short")
    except Exception:
        return list(NIFTY50_FALLBACK), False

# ─────────────────────────────────────────────────────────────
#  CORE 5
# ─────────────────────────────────────────────────────────────
CORE5_KEYS = ["C1_PriceCompress", "C2_VolCompress", "C3_OIBuild",
              "C6_VolExplosion", "C7_ATRExpand"]
CORE5_LABELS = {
    "C1_PriceCompress": "Price Compression",
    "C2_VolCompress":   "Volume Compression",
    "C3_OIBuild":       "OI Build",
    "C6_VolExplosion":  "Volume Explosion",
    "C7_ATRExpand":     "ATR Expansion",
}

COND_LABELS = {
    "C1_PriceCompress":"PC","C2_VolCompress":"VC","C3_OIBuild":"OI",
    "C4_SRLevel":"SR","C5_SmartMoney":"SM","C6_VolExplosion":"VE",
    "C7_ATRExpand":"AE","C8_Breakout":"BO","C9_DeltaRising":"Δ↑",
    "C10_GammaAccel":"Γ↑","C11_VegaExpand":"V↑","C12_ThetaIgnore":"θ✕",
    "C13_UpperCircuit":"UC","C14_RSIReset":"RS","C15_MACDCompress":"MC",
    "C16_BBSqueeze":"BB","C17_EMAStack":"EM","C18_RelStrength":"RS+",
}

KEY_DETAIL_ORDER = [
    "ATR%","10d Rng","Avg Body","Vol Ratio","RSI",
    "5d Move","EMA Stack","S/R","BB Width","BB Sq%",
    "OI Chg","Total OI","PCR","RS vs Bmk","Circuit ~",
]

# ─────────────────────────────────────────────────────────────
#  SAFE COLUMN HELPER
# ─────────────────────────────────────────────────────────────
def find_col(df, prefix):
    for col in df.columns:
        if col.startswith(prefix):
            return col
    return None

# ─────────────────────────────────────────────────────────────
#  DATA FETCH — historical OHLCV from NSE
# ─────────────────────────────────────────────────────────────
def fetch_equity_historical(nse, symbol, months=6):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(months * 30.5))
    frames = []
    cur_end = end
    while cur_end > start:
        cur_start = max(start, cur_end - datetime.timedelta(days=364))
        url = (f"{NSESession.BASE}/api/historical/cm/equity?symbol={symbol}"
               f"&series=[%22EQ%22]&from={cur_start.strftime('%d-%m-%Y')}"
               f"&to={cur_end.strftime('%d-%m-%Y')}")
        data = nse.get_json(url)
        if data and isinstance(data.get("data"), list) and data["data"]:
            frames.append(pd.DataFrame(data["data"]))
        cur_end = cur_start - datetime.timedelta(days=1)
        time.sleep(CONFIG["fetch_delay"])

    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)

    needed = ["CH_TIMESTAMP", "CH_OPENING_PRICE", "CH_TRADE_HIGH_PRICE",
              "CH_TRADE_LOW_PRICE", "CH_CLOSING_PRICE", "CH_TOT_TRADED_QTY"]
    if not all(c in raw.columns for c in needed):
        return None

    raw["Date"] = pd.to_datetime(raw["CH_TIMESTAMP"])
    raw = raw.sort_values("Date")
    df = pd.DataFrame({
        "Open":   pd.to_numeric(raw["CH_OPENING_PRICE"], errors="coerce"),
        "High":   pd.to_numeric(raw["CH_TRADE_HIGH_PRICE"], errors="coerce"),
        "Low":    pd.to_numeric(raw["CH_TRADE_LOW_PRICE"], errors="coerce"),
        "Close":  pd.to_numeric(raw["CH_CLOSING_PRICE"], errors="coerce"),
        "Volume": pd.to_numeric(raw["CH_TOT_TRADED_QTY"], errors="coerce"),
    })
    df.index = raw["Date"].values
    df = df[~df.index.duplicated(keep="last")]
    df.dropna(inplace=True)
    if len(df) < 40:
        return None
    return df

# ─────────────────────────────────────────────────────────────
#  DATA FETCH — real F&O Open Interest from NSE option chain
# ─────────────────────────────────────────────────────────────
def fetch_option_chain(nse, symbol):
    url = f"{NSESession.BASE}/api/option-chain-equities?symbol={symbol}"
    data = nse.get_json(url)
    if not data or "records" not in data:
        return None

    records = data["records"]
    expiries = records.get("expiryDates", [])
    rows = records.get("data", [])
    if not expiries or not rows:
        return None
    nearest = expiries[0]

    tot_ce_oi = tot_pe_oi = tot_ce_chg = tot_pe_chg = 0.0
    tot_ce_vol = tot_pe_vol = 0.0
    for row in rows:
        if row.get("expiryDate") != nearest:
            continue
        ce, pe = row.get("CE"), row.get("PE")
        if ce:
            tot_ce_oi  += ce.get("openInterest", 0) or 0
            tot_ce_chg += ce.get("changeinOpenInterest", 0) or 0
            tot_ce_vol += ce.get("totalTradedVolume", 0) or 0
        if pe:
            tot_pe_oi  += pe.get("openInterest", 0) or 0
            tot_pe_chg += pe.get("changeinOpenInterest", 0) or 0
            tot_pe_vol += pe.get("totalTradedVolume", 0) or 0

    total_oi     = tot_ce_oi + tot_pe_oi
    total_oi_chg = tot_ce_chg + tot_pe_chg
    prev_oi      = total_oi - total_oi_chg
    oi_chg_pct   = (total_oi_chg / prev_oi * 100) if prev_oi > 0 else 0.0
    pcr          = (tot_pe_oi / tot_ce_oi) if tot_ce_oi > 0 else np.nan

    return dict(
        underlying=records.get("underlyingValue"), expiry=nearest,
        total_ce_oi=tot_ce_oi, total_pe_oi=tot_pe_oi,
        chg_ce_oi=tot_ce_chg, chg_pe_oi=tot_pe_chg,
        total_oi=total_oi, total_oi_chg=total_oi_chg, oi_chg_pct=oi_chg_pct,
        ce_volume=tot_ce_vol, pe_volume=tot_pe_vol, pcr=pcr,
    )

# ─────────────────────────────────────────────────────────────
#  BENCHMARK — NIFTY 50 20d return
# ─────────────────────────────────────────────────────────────
def get_nifty_benchmark_return(nse):
    ret = 0.0
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=45)
        url = (f"{NSESession.BASE}/api/historicalOR/indicesHistory?"
               f"indexType=NIFTY%2050&from={start.strftime('%d-%m-%Y')}"
               f"&to={end.strftime('%d-%m-%Y')}")
        data = nse.get_json(url)
        rows = None
        if isinstance(data, dict):
            for key in ("data", "indexCloseOnlineRecords", "records"):
                cand = data.get(key)
                if isinstance(cand, list) and cand:
                    rows = cand
                    break
                if isinstance(cand, dict):
                    subcand = cand.get("data")
                    if isinstance(subcand, list) and subcand:
                        rows = subcand
                        break
        if rows:
            df = pd.DataFrame(rows)
            close_col = next((c for c in df.columns if "close" in c.lower()), None)
            date_col  = next((c for c in df.columns
                              if "date" in c.lower() or "timestamp" in c.lower()), None)
            if close_col and date_col:
                df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
                df = df.dropna(subset=[close_col]).sort_values(date_col)
                closes = df[close_col].values
                if len(closes) >= 21:
                    ret = float((closes[-1] - closes[-21]) / closes[-21] * 100)
    except Exception:
        ret = 0.0
    return ret

# ─────────────────────────────────────────────────────────────
#  INDICATOR COMPUTATION
# ─────────────────────────────────────────────────────────────
def compute_indicators(df):
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["atr"]        = ta.atr(h, l, c, length=14)
    df["atr_pct"]    = (df["atr"] / c) * 100
    df["atr_20min"]  = df["atr_pct"].rolling(20).min()

    bb = ta.bbands(c, length=20, std=2)
    if bb is not None:
        col_u = find_col(bb, "BBU")
        col_l = find_col(bb, "BBL")
        col_m = find_col(bb, "BBM")
        if col_u and col_l and col_m:
            df["bb_width"] = (bb[col_u] - bb[col_l]) / bb[col_m] * 100
        else:
            df["bb_width"] = np.nan
    else:
        df["bb_width"] = np.nan

    df["bb_squeeze"] = df["bb_width"].rolling(252, min_periods=60).rank(pct=True) * 100

    df["vol_20avg"] = v.rolling(20).mean()
    df["vol_ratio"] = v / df["vol_20avg"]

    df["rsi"] = ta.rsi(c, length=14)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        col_h = find_col(macd, "MACDh")
        if col_h:
            df["macd_hist"]     = macd[col_h]
            df["macd_hist_abs"] = df["macd_hist"].abs()
        else:
            df["macd_hist"] = df["macd_hist_abs"] = np.nan
    else:
        df["macd_hist"] = df["macd_hist_abs"] = np.nan

    df["ema9"]  = ta.ema(c, length=9)
    df["ema21"] = ta.ema(c, length=21)
    df["ema50"] = ta.ema(c, length=50)

    df["body_pct"] = ((df["Close"] - df["Open"]).abs() / df["Open"].replace(0, np.nan)) * 100

    n = CONFIG["price_range_days"]
    df["range_n"] = (h.rolling(n).max() - l.rolling(n).min()) / c * 100

    df["price_chg"] = c.pct_change(5) * 100

    return df

# ─────────────────────────────────────────────────────────────
#  S/R PIVOT DETECTION
# ─────────────────────────────────────────────────────────────
def find_sr_levels(df, lookback=60):
    recent = df.tail(lookback)
    h_arr  = recent["High"].values
    l_arr  = recent["Low"].values
    levels = []
    for i in range(2, len(h_arr) - 2):
        if h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i-2] \
                and h_arr[i] > h_arr[i+1] and h_arr[i] > h_arr[i+2]:
            levels.append(float(h_arr[i]))
        if l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i-2] \
                and l_arr[i] < l_arr[i+1] and l_arr[i] < l_arr[i+2]:
            levels.append(float(l_arr[i]))
    return levels

def near_sr(price, levels, pct=1.5):
    for lvl in levels:
        if abs(price - lvl) / price * 100 < pct:
            return True, lvl
    return False, None

# ─────────────────────────────────────────────────────────────
#  CORE SCORER
# ─────────────────────────────────────────────────────────────
def score_ticker(nse, symbol, bmk_ret=0.0):
    df_raw = fetch_equity_historical(nse, symbol, months=CONFIG["data_months"])
    if df_raw is None:
        return None

    df = compute_indicators(df_raw)
    if len(df) < 50:
        return None

    oc = fetch_option_chain(nse, symbol)
    time.sleep(CONFIG["fetch_delay"])

    cur   = "₹"
    row   = df.iloc[-1]
    prev5 = df.iloc[-6:-1]
    price = float(row["Close"])

    scores, details = {}, {}

    # ── CORE 5 ──────────────────────────────────────────────
    atr_low      = float(row["atr_pct"]) < CONFIG["atr_pct_compress_threshold"]
    candle_small = float(prev5["body_pct"].mean()) < CONFIG["candle_size_threshold"]
    range_tight  = float(row["range_n"]) < 4.0
    scores["C1_PriceCompress"] = int(sum([atr_low, candle_small, range_tight]) >= 2)
    details["ATR%"]     = f"{row['atr_pct']:.2f}%"
    details["10d Rng"]  = f"{row['range_n']:.1f}%"
    details["Avg Body"] = f"{prev5['body_pct'].mean():.2f}%"

    vol_compressed = float(row["vol_ratio"]) < CONFIG["vol_compress_ratio"]
    vol_declining  = float(df["vol_ratio"].tail(5).mean()) < 0.80
    scores["C2_VolCompress"] = int(vol_compressed or vol_declining)
    details["Vol Ratio"] = f"{row['vol_ratio']:.2f}x"

    price_flat = abs(float(row["price_chg"])) < CONFIG["price_flat_pct"]
    if oc is not None:
        oi_building = oc["oi_chg_pct"] > CONFIG["oi_build_chg_pct"]
        scores["C3_OIBuild"] = int(price_flat and oi_building)
        details["OI Chg"]   = f"{oc['oi_chg_pct']:+.1f}%"
        details["Total OI"] = f"{oc['total_oi']:,.0f}"
        details["PCR"]      = f"{oc['pcr']:.2f}" if not np.isnan(oc['pcr']) else "N/A"
    else:
        scores["C3_OIBuild"] = 0
        details["OI Chg"] = "N/A (no chain)"
    details["5d ΔPrice"] = f"{row['price_chg']:.1f}%"

    scores["C6_VolExplosion"] = int(float(row["vol_ratio"]) >= CONFIG["volume_explosion_x"])
    details["Vol Explode"] = f"{row['vol_ratio']:.1f}x"

    atr_low_20       = float(df["atr_pct"].tail(20).min())
    atr_expanded_pct = (float(row["atr_pct"]) - atr_low_20) / (atr_low_20 + 1e-6) * 100
    scores["C7_ATRExpand"] = int(atr_expanded_pct >= CONFIG["atr_expansion_pct"])
    details["ATR Expand"] = f"+{atr_expanded_pct:.0f}%"

    # ── REST ────────────────────────────────────────────────
    sr_levels       = find_sr_levels(df)
    is_near, sr_val = near_sr(price, sr_levels, pct=CONFIG["sr_proximity_pct"])
    scores["C4_SRLevel"] = int(is_near)
    details["S/R"] = f"{cur}{sr_val:.2f}" if sr_val else "None"

    if oc is not None:
        smart_money = (oc["oi_chg_pct"] > CONFIG["oi_smart_chg_pct"]) and \
                      (abs(float(row["price_chg"])) < 1.5)
    else:
        smart_money = False
    scores["C5_SmartMoney"] = int(smart_money)

    ema9_ok  = not pd.isna(row["ema9"])  and price > float(row["ema9"])
    ema21_ok = not pd.isna(row["ema21"]) and price > float(row["ema21"])
    big_body = float(row["body_pct"]) > 1.0
    scores["C8_Breakout"] = int(ema9_ok and ema21_ok and big_body
                                and float(row["vol_ratio"]) > 1.5)

    price_5d = float((price - df["Close"].iloc[-6]) / df["Close"].iloc[-6] * 100)
    scores["C9_DeltaRising"] = int(price_5d > 1.5)
    details["5d Move"] = f"{price_5d:.1f}%"

    atr_accel = float(df["atr_pct"].diff().tail(3).mean()) > 0.05
    scores["C10_GammaAccel"] = int(atr_accel and scores["C7_ATRExpand"])

    bb_now  = float(row["bb_width"]) if not pd.isna(row["bb_width"]) else np.nan
    bb_5ago = float(df["bb_width"].iloc[-5]) if not pd.isna(df["bb_width"].iloc[-5]) else np.nan
    bb_exp  = (not np.isnan(bb_now)) and (not np.isnan(bb_5ago)) and (bb_now > bb_5ago * 1.10)
    scores["C11_VegaExpand"] = int(bb_exp)
    details["BB Width"] = f"{bb_now:.2f}%" if not np.isnan(bb_now) else "N/A"

    scores["C12_ThetaIgnore"] = int(abs(price_5d) > 3.0)

    scores["C13_UpperCircuit"] = int(float(row["atr_pct"]) > 1.5 and price_5d > 0)
    circuit_target = price * (1 + float(row["atr_pct"]) * 3 / 100)
    details["Circuit ~"] = f"{cur}{circuit_target:.2f} (+{row['atr_pct']*3:.1f}%)"

    rsi_val = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50.0
    scores["C14_RSIReset"] = int(CONFIG["rsi_low"] <= rsi_val <= CONFIG["rsi_high"])
    details["RSI"] = f"{rsi_val:.1f}"

    macd_shrink = False
    if not pd.isna(row["macd_hist_abs"]):
        m5  = float(df["macd_hist_abs"].tail(5).mean())
        m20 = float(df["macd_hist_abs"].tail(20).mean())
        macd_shrink = (m5 < m20 * 0.60)
    scores["C15_MACDCompress"] = int(macd_shrink)

    bb_sq = float(row["bb_squeeze"]) if not pd.isna(row["bb_squeeze"]) else 50.0
    scores["C16_BBSqueeze"] = int(bb_sq < CONFIG["bb_squeeze_percentile"])
    details["BB Sq%"] = f"{bb_sq:.0f}th pct"

    ema_stack = (not pd.isna(row["ema9"])  and
                 not pd.isna(row["ema21"]) and
                 not pd.isna(row["ema50"]) and
                 float(row["ema9"]) > float(row["ema21"]) > float(row["ema50"]))
    scores["C17_EMAStack"] = int(ema_stack)
    details["EMA Stack"] = "✅" if ema_stack else "❌"

    stock_20d = float((price - df["Close"].iloc[-21]) / df["Close"].iloc[-21] * 100)
    scores["C18_RelStrength"] = int(stock_20d > bmk_ret)
    details["RS vs Bmk"] = f"{stock_20d:.1f}% vs {bmk_ret:.1f}% (NIFTY)"

    # ── TOTALS ──────────────────────────────────────────────
    total = sum(scores.values())
    comp_keys  = ["C1_PriceCompress","C2_VolCompress","C3_OIBuild","C4_SRLevel",
                  "C5_SmartMoney","C14_RSIReset","C15_MACDCompress","C16_BBSqueeze"]
    expl_keys  = ["C6_VolExplosion","C7_ATRExpand","C8_Breakout",
                  "C9_DeltaRising","C10_GammaAccel","C11_VegaExpand"]
    extra_keys = ["C12_ThetaIgnore","C13_UpperCircuit","C17_EMAStack","C18_RelStrength"]

    comp_score  = sum(scores[k] for k in comp_keys)
    expl_score  = sum(scores[k] for k in expl_keys)
    extra_score = sum(scores[k] for k in extra_keys)
    core5_score = sum(scores[k] for k in CORE5_KEYS)

    if   total >= 12: phase, pc = "🔥 EXPLOSION READY", "#ff4444"
    elif total >=  8: phase, pc = "⚡ COMPRESSION",     "#ffaa00"
    elif total >=  5: phase, pc = "👀 EARLY WATCH",     "#44aaff"
    else:             phase, pc = "😴 NEUTRAL",         "#888888"

    is_trigger = bool(total >= 12 and scores["C6_VolExplosion"] and scores["C8_Breakout"])

    return dict(
        ticker=symbol, price=price, market="IN", currency=cur,
        total_score=total, compression_score=comp_score,
        explosion_score=expl_score, extras_score=extra_score,
        core5_score=core5_score,
        phase=phase, phase_color=pc, is_trigger=is_trigger,
        scores=scores, details=details, df=df, option_chain=oc,
    )

# ─────────────────────────────────────────────────────────────
#  SCAN RUNNER — generator-style so a UI can drive its own progress bar
# ─────────────────────────────────────────────────────────────
def run_scan(nse, tickers, progress_callback=None):
    """Scans `tickers` and returns (results, errors).
    progress_callback(i, total, ticker, result_or_none) is called after
    each ticker if provided — wire it to st.progress()/st.empty() etc."""
    bmk_ret = get_nifty_benchmark_return(nse)
    results, errors = [], []
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        res = None
        try:
            res = score_ticker(nse, ticker, bmk_ret=bmk_ret)
            if res and res["total_score"] >= CONFIG["min_score_show"]:
                results.append(res)
        except Exception as e:
            errors.append(f"{ticker}: {e}")
        if progress_callback:
            progress_callback(i, total, ticker, res)
        time.sleep(CONFIG["fetch_delay"])

    results.sort(key=lambda x: (x["is_trigger"], x["core5_score"], x["total_score"]), reverse=True)
    return results, errors

# ─────────────────────────────────────────────────────────────
#  CARD-GRID HTML
# ─────────────────────────────────────────────────────────────
CARD_CSS = """
<style>
  body{margin:0;background:#0d0d0d}
  .zh-grid{display:flex;flex-wrap:wrap;gap:12px;font-family:'Courier New',monospace;padding:10px;background:#0d0d0d}
  .zh-card{background:#111827;border:1px solid #1f2937;border-radius:10px;
           padding:14px;width:270px;color:#e5e7eb;font-size:12px;
           box-shadow:0 2px 8px rgba(0,0,0,.4)}
  .zh-card.fire {border-color:#ff4444;box-shadow:0 0 12px rgba(255,68,68,.3)}
  .zh-card.spark{border-color:#ffaa00;box-shadow:0 0 10px rgba(255,170,0,.25)}
  .zh-card.watch{border-color:#44aaff}
  .zh-card.triggered{border-color:#fbbf24;border-width:2px;
          box-shadow:0 0 18px rgba(251,191,36,.55)}
  .zh-trigger-badge{background:#fbbf24;color:#111827;font-size:10px;
          font-weight:bold;padding:3px 8px;border-radius:4px;
          display:inline-block;margin-bottom:6px;letter-spacing:0.5px}
  .zh-ticker{font-size:18px;font-weight:bold;color:#fff}
  .zh-mkt{font-size:9px;color:#6b7280;margin-left:6px}
  .zh-price {font-size:13px;color:#9ca3af}
  .zh-phase {font-size:10px;font-weight:bold;padding:3px 7px;border-radius:4px;
             display:inline-block;margin:5px 0}
  .zh-scores{display:flex;justify-content:space-between;margin:6px 0}
  .zh-sb{text-align:center}
  .zh-sb-val{font-size:15px;font-weight:bold}
  .zh-sb-lbl{font-size:9px;color:#6b7280}
  .zh-bar-bg{background:#1f2937;border-radius:3px;height:5px;margin:5px 0}
  .zh-bar-fg{height:5px;border-radius:3px}
  .zh-dots{display:flex;flex-wrap:wrap;gap:3px;margin-top:7px}
  .zh-dot{width:22px;height:22px;border-radius:4px;font-size:8px;
          display:flex;align-items:center;justify-content:center;font-weight:bold}
  .zh-dot.on {background:#14532d;color:#4ade80}
  .zh-dot.off{background:#1f2937;color:#374151}
  .zh-detail{color:#6b7280;margin-top:7px;font-size:10px;line-height:1.75}
  .zh-detail span{color:#d1d5db}
  .zh-core5{margin:8px 0;padding:8px;background:#0b1220;border-radius:6px;
          border:1px solid #1f2937}
  .zh-core5-hdr{display:flex;justify-content:space-between;align-items:center;
          margin-bottom:5px}
  .zh-core5-title{font-size:9px;color:#9ca3af;letter-spacing:0.5px;font-weight:bold}
  .zh-core5-score{font-size:12px;font-weight:bold;color:#fbbf24}
  .zh-core5-row{display:flex;justify-content:space-between;font-size:9px;padding:1px 0}
  .zh-core5-row.on  span.lbl{color:#4ade80}
  .zh-core5-row.off span.lbl{color:#6b7280}
  .zh-core5-row span.mark{font-size:10px}
</style>
"""

def render_cards(results):
    filtered = [r for r in results if r["total_score"] >= CONFIG["min_score_show"]]
    cards = []
    for r in filtered:
        if r["is_trigger"]:
            cls = "triggered"
        elif "EXPLOSION" in r["phase"]:
            cls = "fire"
        elif "COMPRESSION" in r["phase"]:
            cls = "spark"
        else:
            cls = "watch"
        bar_pct = int(r["total_score"] / 18 * 100)
        bar_col = "#fbbf24" if r["is_trigger"] else \
                  ("#ff4444" if bar_pct >= 67 else ("#ffaa00" if bar_pct >= 44 else "#44aaff"))

        dots = "".join(
            f'<div class="zh-dot {"on" if r["scores"].get(k,0) else "off"}" title="{k}">{lbl}</div>'
            for k, lbl in COND_LABELS.items()
        )
        det_lines = "".join(
            f'{k}: <span>{r["details"][k]}</span><br>'
            for k in KEY_DETAIL_ORDER if k in r["details"]
        )
        trigger_badge = '<div class="zh-trigger-badge">🎯 BUY TRIGGER</div>' if r["is_trigger"] else ""
        core5_rows = "".join(
            f'<div class="zh-core5-row {"on" if r["scores"].get(k,0) else "off"}">'
            f'<span class="lbl">{i}. {CORE5_LABELS[k]}</span>'
            f'<span class="mark">{"✅" if r["scores"].get(k,0) else "❌"}</span></div>'
            for i, k in enumerate(CORE5_KEYS, start=1)
        )

        cards.append(f"""
        <div class="zh-card {cls}">
          {trigger_badge}
          <div style="display:flex;justify-content:space-between;align-items:start">
            <div>
              <div class="zh-ticker">{r['ticker']}<span class="zh-mkt">{r['market']}</span></div>
              <div class="zh-price">{r['currency']}{r['price']:.2f}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:24px;font-weight:bold;color:{bar_col}">
                {r['total_score']}<span style="font-size:11px;color:#6b7280">/18</span>
              </div>
            </div>
          </div>
          <div class="zh-phase" style="background:{r['phase_color']}22;color:{r['phase_color']}">{r['phase']}</div>
          <div class="zh-bar-bg"><div class="zh-bar-fg" style="width:{bar_pct}%;background:{bar_col}"></div></div>
          <div class="zh-core5">
            <div class="zh-core5-hdr">
              <div class="zh-core5-title">CORE 5</div>
              <div class="zh-core5-score">{r['core5_score']}/5</div>
            </div>
            {core5_rows}
          </div>
          <div class="zh-scores">
            <div class="zh-sb"><div class="zh-sb-val" style="color:#44aaff">{r['compression_score']}/8</div><div class="zh-sb-lbl">COMPRESS</div></div>
            <div class="zh-sb"><div class="zh-sb-val" style="color:#ff4444">{r['explosion_score']}/6</div><div class="zh-sb-lbl">EXPLODE</div></div>
            <div class="zh-sb"><div class="zh-sb-val" style="color:#a855f7">{r['extras_score']}/4</div><div class="zh-sb-lbl">EXTRAS</div></div>
          </div>
          <div class="zh-dots">{dots}</div>
          <div class="zh-detail">{det_lines}</div>
        </div>""")

    return CARD_CSS + '<div class="zh-grid">' + "".join(cards) + "</div>"

# ─────────────────────────────────────────────────────────────
#  DETAIL CHART — returns a matplotlib Figure (caller does st.pyplot(fig))
# ─────────────────────────────────────────────────────────────
def build_detail_chart(result, n=60):
    r  = result
    df = r["df"].tail(n).copy()
    cur = r["currency"]

    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
    gs  = GridSpec(4, 1, figure=fig, hspace=0.05, height_ratios=[3,1,1,1])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    for ax in axes:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#6b7280", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1f2937")

    x   = list(range(len(df)))
    c   = df["Close"].values
    o   = df["Open"].values
    h   = df["High"].values
    l   = df["Low"].values
    v   = df["Volume"].values
    vma = df["vol_20avg"].values

    ax = axes[0]
    for i in x:
        col = "#22c55e" if c[i] >= o[i] else "#ef4444"
        ax.plot([i,i],[l[i],h[i]], color=col, lw=0.8, alpha=0.6)
        ax.add_patch(plt.Rectangle((i-0.3, min(o[i],c[i])), 0.6,
                                    abs(c[i]-o[i]), color=col, alpha=0.9))
    for ema, col, lbl in [("ema9","#facc15","EMA9"),
                           ("ema21","#60a5fa","EMA21"),
                           ("ema50","#a78bfa","EMA50")]:
        ax.plot(x, df[ema].values, color=col, lw=1.0, label=lbl, alpha=0.85)

    for lvl in find_sr_levels(df)[:4]:
        ax.axhline(lvl, color="#f97316", lw=0.7, ls="--", alpha=0.6)

    ax.set_title(
        f"{r['ticker']} ({r['market']})  |  Score {r['total_score']}/18  |  {r['phase']}  |  {cur}{r['price']:.2f}",
        color="#e5e7eb", fontsize=11, pad=8, fontfamily="monospace"
    )
    ax.legend(loc="upper left", fontsize=7, facecolor="#1f2937",
              labelcolor="#e5e7eb", framealpha=0.7)
    ax.set_ylabel("Price", color="#6b7280", fontsize=8)
    ax.tick_params(labelbottom=False)

    ax = axes[1]
    for i in x:
        col = "#22c55e" if v[i] >= vma[i]*CONFIG["volume_explosion_x"] else \
              ("#facc15" if v[i] >= vma[i] else "#374151")
        ax.bar(i, v[i], color=col, width=0.8, alpha=0.8)
    ax.plot(x, vma, color="#60a5fa", lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("Volume", color="#6b7280", fontsize=8)
    ax.tick_params(labelbottom=False)

    ax   = axes[2]
    atrv = df["atr_pct"].values
    ax.plot(x, atrv, color="#f59e0b", lw=1.2, label="ATR%")
    ax.axhline(CONFIG["atr_pct_compress_threshold"], color="#44aaff", lw=0.8,
               ls="--", alpha=0.7, label=f"Compress <{CONFIG['atr_pct_compress_threshold']}%")
    ax.fill_between(x, 0, atrv,
                    where=np.array(atrv) < CONFIG["atr_pct_compress_threshold"],
                    color="#44aaff", alpha=0.08)
    ax.legend(loc="upper right", fontsize=7, facecolor="#1f2937",
              labelcolor="#e5e7eb", framealpha=0.7)
    ax.set_ylabel("ATR%", color="#6b7280", fontsize=8)
    ax.tick_params(labelbottom=False)

    ax    = axes[3]
    rsiv  = df["rsi"].values
    ax.plot(x, rsiv, color="#a78bfa", lw=1.2, label="RSI")
    ax.axhline(CONFIG["rsi_high"], color="#6b7280", lw=0.7, ls="--")
    ax.axhline(CONFIG["rsi_low"],  color="#6b7280", lw=0.7, ls="--")
    ax.fill_between(x, CONFIG["rsi_low"], CONFIG["rsi_high"],
                    alpha=0.08, color="#44aaff", label="Reset Zone")
    ax.set_ylim(10, 90)
    ax.set_ylabel("RSI", color="#6b7280", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, facecolor="#1f2937",
              labelcolor="#e5e7eb", framealpha=0.7)

    cond_str = "  ".join(
        f"{k.split('_')[0]}:{'✅' if v else '❌'}"
        for k, v in r["scores"].items()
    )
    oi_line = ""
    if r.get("option_chain") and not np.isnan(r["option_chain"]["pcr"]):
        oc = r["option_chain"]
        oi_line = f"  CE OI:{oc['total_ce_oi']:,.0f}  PE OI:{oc['total_pe_oi']:,.0f}  PCR:{oc['pcr']:.2f}"
    fig.text(0.01, 0.005, cond_str + oi_line, fontsize=6, color="#4b5563",
             fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    return fig
