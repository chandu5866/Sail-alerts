"""
SAIL intraday alert bot (NSE) - runs every 5 minutes on GitHub Actions.

Strategy: Opening Range Breakout (ORB) with VWAP and volume filters.
  - Opening range = high/low of the first 15 minutes (9:15-9:30 IST).
  - BUY  call: a COMPLETED 5-min candle closes above the range high,
               above VWAP, with volume above average.
  - SELL call: the mirror image (intraday short).
  - SL = other side of the opening range, limited to 0.3%-1.0% of price.
  - Targets = 1x and 2x the risk. After Target 1, SL moves to cost.
  - Max one call per day. No new calls after 1:30 PM. Square-off reminder
    at 3:05 PM.

Safety checks:
  - Only completed candles are used (never a half-formed one).
  - No new call if the data looks stale (feed lagging too far behind).
  - No call if the opening range is too narrow (noise) or too wide (risky).
  - Trade management is replayed candle by candle, so a late or missed run
    cannot skip a target or stop-loss. If SL and a target fall in the same
    candle, SL is assumed first (the conservative choice).
  - Prices are rounded to the NSE tick size (Rs 0.05).

Educational tool, not financial advice. Always confirm the live price in
your broker app before placing an order.
"""

import json
import os
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# ------------------------------ CONFIG --------------------------------
NAME = "SAIL"
GROWW_SYMBOL = "NSE-SAIL"      # Groww symbol: exchange-tradingsymbol
YAHOO_SYMBOL = "SAIL.NS"        # fallback data source (free, delayed)
IST = ZoneInfo("Asia/Kolkata")

OPEN_T = dtime(9, 15)
RANGE_END = dtime(9, 30)        # opening range = 9:15 to 9:30
NO_NEW_AFTER = dtime(13, 30)    # no fresh calls after this time
SQUAREOFF_T = dtime(15, 5)      # square-off reminder
STOP_T = dtime(15, 15)          # bot stops working at this time

BAR_MIN = 5
TICK = 0.05
MIN_RANGE_PCT, MAX_RANGE_PCT = 0.25, 1.5   # opening range size filter
MIN_RISK_PCT, MAX_RISK_PCT = 0.3, 1.0      # SL distance limits
TARGET_MULTIPLES = (1.0, 2.0)
VOLUME_MULT = 1.2               # breakout candle volume vs average
MAX_CHASE_R = 0.25              # don't enter if price ran > 0.25R past entry
MAX_DATA_LAG_MIN = 25           # data older than this = no new calls

STATE_FILE = "state.json"
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
GROWW_TOTP_TOKEN = os.getenv("GROWW_TOTP_TOKEN", "")
GROWW_TOTP_SECRET = os.getenv("GROWW_TOTP_SECRET", "")
# -----------------------------------------------------------------------


def tick(x: float) -> float:
    return round(round(x / TICK) * TICK, 2)


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def send(text: str) -> None:
    print(text)
    if not TG_TOKEN or not TG_CHAT_ID:
        print("(Telegram not configured)")
        return
    import requests
    for attempt in range(3):          # retry so an alert is not lost
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT_ID, "text": text},
                timeout=20,
            )
            if r.ok:
                return
            print("Telegram HTTP", r.status_code, r.text[:200])
        except Exception as e:
            print("Telegram error:", e)


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_bars_yahoo() -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(YAHOO_SYMBOL, period="5d", interval=f"{BAR_MIN}m",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    df.index = idx.tz_convert(IST)
    return df.dropna(subset=["High", "Low", "Close"])


def candles_to_df(rows) -> pd.DataFrame:
    """Groww candle rows -> DataFrame. Row = [time, o, h, l, c, volume, oi]."""
    recs = []
    for c in rows or []:
        ts = c[0]
        if isinstance(ts, (int, float)):               # epoch seconds / millis
            unit = "ms" if ts > 1e11 else "s"
            t = pd.Timestamp(ts, unit=unit, tz="UTC").tz_convert(IST)
        else:                                          # "2026-11-09T09:15:00"
            t = pd.Timestamp(ts)
            t = t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)
        recs.append((t, c[1], c[2], c[3], c[4], c[5] or 0))
    if not recs:
        return pd.DataFrame()
    df = (pd.DataFrame(recs, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
          .set_index("ts").sort_index())
    return df[~df.index.duplicated(keep="last")]


def get_bars_groww() -> pd.DataFrame:
    """5-minute candles for today from the Groww Trading API (TOTP login)."""
    from growwapi import GrowwAPI
    import pyotp
    totp = pyotp.TOTP(GROWW_TOTP_SECRET).now()
    token = GrowwAPI.get_access_token(api_key=GROWW_TOTP_TOKEN, totp=totp)
    groww = GrowwAPI(token)
    now = datetime.now(IST)
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    resp = groww.get_historical_candles(
        exchange=groww.EXCHANGE_NSE,
        segment=groww.SEGMENT_CASH,
        groww_symbol=GROWW_SYMBOL,
        start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=now.strftime("%Y-%m-%d %H:%M:%S"),
        candle_interval=groww.CANDLE_INTERVAL_MIN_5,
    )
    return candles_to_df(resp.get("candles"))


def get_bars():
    """Returns (dataframe, source_name, groww_error_or_None)."""
    err = None
    if GROWW_TOTP_TOKEN and GROWW_TOTP_SECRET:
        try:
            df = get_bars_groww()
            if not df.empty:
                return df, "Groww", None
            err = "Groww returned no candles"
        except Exception as e:  # network, auth, plan, rate limit...
            err = f"{type(e).__name__}: {e}"
        print("Groww data failed:", err)
    return get_bars_yahoo(), "Yahoo (delayed)", err


def process(df: pd.DataFrame, state: dict, now: datetime):
    """Pure logic: returns (new_state, [messages]). No network, easy to test."""
    msgs = []
    today = now.date()
    if state.get("date") != str(today):
        state = {"date": str(today), "status": "none"}

    day = df[df.index.date == today]
    # Only completed candles
    done = day[day.index + timedelta(minutes=BAR_MIN) <= now]
    orb = done[(done.index.time >= OPEN_T) & (done.index.time < RANGE_END)]
    if len(orb) < 3:
        return state, msgs

    or_high, or_low = float(orb["High"].max()), float(orb["Low"].min())
    last_end = done.index[-1] + timedelta(minutes=BAR_MIN)
    lag_min = (now - last_end).total_seconds() / 60
    stale = lag_min > MAX_DATA_LAG_MIN

    # ---------------- Manage an open call ----------------
    if state["status"] == "open":
        side = state["side"]
        long_ = side == "BUY"
        new_bars = done[done.index > pd.Timestamp(state["last_processed"])]
        for ts, b in new_bars.iterrows():
            state["last_processed"] = ts.isoformat()
            hi, lo = float(b["High"]), float(b["Low"])
            sl, t1, t2 = state["sl"], state["t1"], state["t2"]
            sl_hit = lo <= sl if long_ else hi >= sl
            t1_hit = hi >= t1 if long_ else lo <= t1
            t2_hit = hi >= t2 if long_ else lo <= t2
            if sl_hit:  # conservative: SL first if both touched in one candle
                if state["t1_hit"]:
                    msgs.append(f"⚪ {NAME}: price came back to cost "
                                f"({fmt(sl)}). Exit the rest. Trade closed.")
                else:
                    msgs.append(f"❌ {NAME}: SL hit at {fmt(sl)}. "
                                f"Exit the {side} trade.")
                state["status"] = "closed"
                break
            if t2_hit:
                msgs.append(f"🎯🎯 {NAME}: Target 2 ({fmt(t2)}) reached. "
                            f"Book full profit.")
                state["status"] = "closed"
                break
            if t1_hit and not state["t1_hit"]:
                msgs.append(f"🎯 {NAME}: Target 1 ({fmt(t1)}) reached. Book "
                            f"part profit and move SL to cost "
                            f"({fmt(state['entry'])}).")
                state["t1_hit"] = True
                state["sl"] = state["entry"]

        if state["status"] == "open" and now.time() >= SQUAREOFF_T:
            msgs.append(f"⏰ {NAME}: market closing soon. Square off your "
                        f"intraday position now.")
            state["status"] = "closed"
        return state, msgs

    # ---------------- Look for a new call ----------------
    if state["status"] != "none":
        return state, msgs
    if now.time() > NO_NEW_AFTER:
        return state, msgs
    if stale:
        if not state.get("stale_warned"):
            msgs.append(f"⚠️ {NAME}: price data is about {lag_min:.0f} min behind. "
                        f"No new calls until it catches up.")
            state["stale_warned"] = True
        return state, msgs

    range_pct = (or_high - or_low) / or_low * 100
    if not (MIN_RANGE_PCT <= range_pct <= MAX_RANGE_PCT):
        why = "too narrow" if range_pct < MIN_RANGE_PCT else "too wide"
        msgs.append(f"ℹ️ {NAME}: no call today. Opening range is {why} "
                    f"({range_pct:.2f}%).")
        state["status"] = "skipped"
        return state, msgs

    after_range = done[done.index.time >= RANGE_END]
    if after_range.empty:
        return state, msgs
    bar = after_range.iloc[-1]
    price = float(bar["Close"])

    typical = (done["High"] + done["Low"] + done["Close"]) / 3
    vol = done["Volume"].astype(float)
    vwap = float((typical * vol).sum() / vol.sum()) if vol.sum() > 0 \
        else float(typical.mean())

    prior_vol = after_range["Volume"].astype(float).iloc[:-1]
    avg_vol = float(prior_vol.mean()) if len(prior_vol) else 0.0
    vol_ok = avg_vol == 0 or float(bar["Volume"]) >= VOLUME_MULT * avg_vol

    side = None
    if price > or_high and price > vwap and vol_ok:
        side = "BUY"
    elif price < or_low and price < vwap and vol_ok:
        side = "SELL"
    if not side:
        return state, msgs

    sign = 1 if side == "BUY" else -1
    raw = (price - or_low) if side == "BUY" else (or_high - price)
    risk = min(max(raw, price * MIN_RISK_PCT / 100), price * MAX_RISK_PCT / 100)
    entry = tick(price)
    sl = tick(entry - sign * risk)
    t1 = tick(entry + sign * risk * TARGET_MULTIPLES[0])
    t2 = tick(entry + sign * risk * TARGET_MULTIPLES[1])
    chase = tick(entry + sign * risk * MAX_CHASE_R)

    state.update(status="open", side=side, entry=entry, sl=sl, t1=t1, t2=t2,
                 t1_hit=False, last_processed=done.index[-1].isoformat())
    icon = "🟢" if side == "BUY" else "🔴"
    verb = "Buy" if side == "BUY" else "Sell"
    way = "above" if side == "BUY" else "below"
    msgs.append(
        f"{icon} {NAME} INTRADAY {side}\n"
        f"{verb} at - {fmt(entry)} (skip if price is already {way} {fmt(chase)})\n"
        f"SL - {fmt(sl)}\n"
        f"Tg - {fmt(t1)}, {fmt(t2)}\n"
        f"Opening range {fmt(or_low)}-{fmt(or_high)}, VWAP {fmt(vwap)}\n"
        f"Signal candle closed {last_end:%H:%M} IST. Data can be delayed: "
        f"check the live price before ordering."
    )
    return state, msgs


def main() -> None:
    now = datetime.now(IST)

    # Manual run from the Actions tab = connection test
    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
        df, source, err = get_bars()
        if df.empty:
            last = "No data received right now."
        else:
            last = (f"Data source: {source}\n"
                    f"Latest candle: {fmt(float(df['Close'].iloc[-1]))} "
                    f"(candle starting {df.index[-1]:%d %b %H:%M} IST)")
        if err:
            last += f"\n⚠️ Groww problem: {err}"
        send(f"✅ {NAME} alert bot is running.\n{last}")
        return

    if now.weekday() >= 5 or not (RANGE_END <= now.time() <= STOP_T):
        return
    df, source, err = get_bars()
    if df.empty:
        print("No data (holiday or feed problem).")
        return

    state, msgs = process(df, load_state(), now)
    if err and not state.get("groww_err_sent"):
        msgs.append(f"⚠️ {NAME} bot: Groww data failed ({err}). "
                    f"Using delayed Yahoo data instead.")
        state["groww_err_sent"] = True
    for m in msgs:
        send(m)
    save_state(state)


if __name__ == "__main__":
    main()
