"""
app.py — Streamlit front-end for the Zero to Hero Premium Explosion Scanner.
Run locally:   streamlit run app.py
Deploy:        push to GitHub, then deploy on share.streamlit.io pointing
               at this file. See README.md for the important caveat about
               NSE blocking datacenter IPs (incl. Streamlit Cloud's).
"""

import time
import pandas as pd
import streamlit as st

import scanner_core as sc

st.set_page_config(
    page_title="Zero to Hero — Explosion Scanner",
    page_icon="⚡",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
#  ONE-TIME SESSION STATE
# ─────────────────────────────────────────────────────────────
if "nse" not in st.session_state:
    st.session_state.nse = None          # NSESession, created lazily
if "results" not in st.session_state:
    st.session_state.results = []
if "errors" not in st.session_state:
    st.session_state.errors = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None

def get_session():
    if st.session_state.nse is None:
        st.session_state.nse = sc.NSESession()
    return st.session_state.nse

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_universe():
    nse = sc.NSESession()  # short-lived, just for this one fetch
    tickers, live = sc.get_nifty50_tickers(nse)
    return tickers, live

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — controls
# ─────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Scanner Controls")

tickers_all, is_live = cached_universe()
st.sidebar.caption(
    ("✅ Live NIFTY 50 list" if is_live else "⚠️ Using fallback NIFTY 50 list")
    + f" — {len(tickers_all)} symbols"
)

universe_mode = st.sidebar.radio("Universe", ["Full NIFTY 50", "Custom subset"])
if universe_mode == "Custom subset":
    chosen = st.sidebar.multiselect("Pick symbols", tickers_all, default=tickers_all[:10])
else:
    chosen = tickers_all

st.sidebar.subheader("Thresholds")
sc.CONFIG["min_score_show"]        = st.sidebar.slider("Min score to show", 0, 18, sc.CONFIG["min_score_show"])
sc.CONFIG["volume_explosion_x"]    = st.sidebar.slider("Volume explosion (x avg)", 1.5, 5.0, sc.CONFIG["volume_explosion_x"], 0.1)
sc.CONFIG["atr_expansion_pct"]     = st.sidebar.slider("ATR expansion (%)", 5, 50, sc.CONFIG["atr_expansion_pct"])
sc.CONFIG["oi_build_chg_pct"]      = st.sidebar.slider("OI build threshold (%)", 1.0, 20.0, sc.CONFIG["oi_build_chg_pct"], 0.5)
sc.CONFIG["data_months"]           = st.sidebar.slider("Historical lookback (months)", 3, 12, sc.CONFIG["data_months"])

st.sidebar.subheader("Rate limiting")
sc.CONFIG["fetch_delay"] = st.sidebar.slider(
    "Delay between NSE calls (s)", 0.5, 3.0, sc.CONFIG["fetch_delay"], 0.1,
    help="NSE rate-limits scripted access hard. Raise this if you see 403s or empty data.",
)
st.sidebar.caption(
    f"Estimated time: ~{len(chosen) * (2 * sc.CONFIG['fetch_delay'] + 1):.0f}s "
    f"for {len(chosen)} symbol(s)."
)

run_clicked = st.sidebar.button("🚀 Run Scanner", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.title("⚡ Zero to Hero — Premium Explosion Scanner")
st.caption("NIFTY 50 · Compression → **real NSE Open Interest** build → Volume/ATR Explosion")

with st.expander("How this works / data source notes", expanded=False):
    st.markdown("""
- **Data source:** NSE India's own public endpoints — historical OHLCV
  (`/api/historical/cm/equity`) and live F&O option chains
  (`/api/option-chain-equities`) for real Open Interest. No third-party
  data vendor.
- **OI Build (C3)** and **Smart Money (C5)** use real change-in-OI from
  the option chain, not a price/volume proxy.
- NSE **rate-limits scripted access** and can be picky about cloud IPs.
  If a scan comes back mostly empty, try again, raise the delay slider,
  or run this app locally instead of on a cloud host.
- 🎯 **Buy Trigger** = score ≥ 12 **and** Volume Explosion (C6) **and**
  Breakout (C8) both fired. Everything else is a setup, not a signal.
- This is pattern-matching, not financial advice — confirm price action
  and manage your own risk before acting on anything shown here.
    """)

# ─────────────────────────────────────────────────────────────
#  RUN SCAN
# ─────────────────────────────────────────────────────────────
if run_clicked:
    if not chosen:
        st.warning("Pick at least one symbol.")
    else:
        nse = get_session()
        progress = st.progress(0.0)
        status = st.empty()

        def on_progress(i, total, ticker, res):
            pct = (i + 1) / total
            progress.progress(pct)
            tag = "✅" if res else "—"
            status.text(f"[{i+1}/{total}] {tag} {ticker}")

        t0 = time.time()
        results, errors = sc.run_scan(nse, chosen, progress_callback=on_progress)
        elapsed = time.time() - t0

        st.session_state.results = results
        st.session_state.errors = errors
        st.session_state.last_run = elapsed

        progress.empty()
        status.empty()

# ─────────────────────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────────────────────
results = st.session_state.results
errors = st.session_state.errors

if st.session_state.last_run is not None:
    st.caption(f"Last scan: {len(results)} results, {len(errors)} errors, "
               f"took {st.session_state.last_run:.0f}s.")

if not results:
    st.info("Set your controls in the sidebar and hit **Run Scanner**.")
else:
    triggers = [r for r in results if r["is_trigger"]]

    if triggers:
        st.success(f"🎯 {len(triggers)} BUY TRIGGER(S) confirmed (score ≥12, "
                    "Volume Explosion + Breakout both fired)")
        trig_df = pd.DataFrame([{
            "Ticker": r["ticker"], "Price": f"₹{r['price']:.2f}",
            "Score": f"{r['total_score']}/18", "Core5": f"{r['core5_score']}/5",
            "5d Move": r["details"].get("5d Move", "—"),
            "Vol": r["details"].get("Vol Explode", "—"),
            "OI Chg": r["details"].get("OI Chg", "—"),
        } for r in triggers])
        st.dataframe(trig_df, use_container_width=True, hide_index=True)
    else:
        st.info("🎯 No confirmed Buy Triggers right now — everything below is "
                "still Compression/Watch, not a confirmed entry.")

    st.subheader("Summary")
    summary_df = pd.DataFrame([{
        "🎯": "🎯" if r["is_trigger"] else "",
        "Ticker": r["ticker"], "Price": f"₹{r['price']:.2f}",
        "Core5": f"{r['core5_score']}/5", "Score": f"{r['total_score']}/18",
        "Compress": f"{r['compression_score']}/8",
        "Explode": f"{r['explosion_score']}/6",
        "Phase": r["phase"],
    } for r in results])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Cards")
    cards_html = sc.render_cards(results)
    # rough height estimate so the iframe doesn't clip or leave dead space
    n_shown = len([r for r in results if r["total_score"] >= sc.CONFIG["min_score_show"]])
    rows = max(1, -(-n_shown // 4))  # ceil div, ~4 cards per row on a wide screen
    st.components.v1.html(cards_html, height=rows * 340 + 60, scrolling=True)

    st.subheader("Detail chart")
    pick = st.selectbox("Pick a symbol", [r["ticker"] for r in results])
    picked_result = next(r for r in results if r["ticker"] == pick)
    fig = sc.build_detail_chart(picked_result)
    st.pyplot(fig)

    if errors:
        with st.expander(f"⚠️ {len(errors)} error(s)"):
            for e in errors:
                st.text(e)
