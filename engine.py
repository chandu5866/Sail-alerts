"""
Intraday trading engine (v2, multi-timeframe) shared by trainer.py (the weekly
study) and live_bot.py (the live Telegram calls).

Both run the SAME functions on completed 5-minute candles, and a signal on
candle i only uses data finished by the end of candle i, so what the trainer
tests is exactly what the live bot sends. 15-minute and daily candles are
built from the same 5-minute data (no extra downloads).

Trend filters (the bigger picture decides the direction)
  m15    : 15-min EMA 20 above EMA 50 -> only BUYs; below -> only SELLs
  day    : yesterday's close above its 20-day average -> only BUYs; below -> SELLs
  m15day : both must agree

Setups (5-min entries unless stated; each can BUY or SELL-short, intraday only)
  bb     : Bollinger band (20, 2) + RSI(7) reversal: price stretched, candle turns back
  rsipb  : RSI(7) pullback: RSI dips below 30 then turns back up (mirror for shorts)
  vwap   : pullback to VWAP that closes back on the trend side
  orb    : first close beyond the 9:15-9:30 range, right side of VWAP, with volume
  ema15  : 15-min EMA 9 crosses EMA 21 (checked when each 15-min candle closes)
  st15   : 15-min Supertrend (10, 3) flips
  dip    : yesterday was a sharp dip (daily RSI(2) under 15) in a daily uptrend;
           buy at 9:35 if the day is holding above its open (mirror for shorts)
  (orb/ema/vwap/bb/st without a trend filter are the v1 rules, kept so an old
   live_config.json still works.)
Trade rules
  Entry at the close of the signal candle. Signals only from 9:30 to 2:30 PM.
  Stop = ATR(14 x 5-min) x am (1.5, or 2.5 for 15-min setups), kept between
  0.25% and 1.2% of price. Target = R x stop distance. Stop assumed first if
  both are hit in one candle. One position per stock at a time, at most 3
  trades per stock per day, squared off at the close of the 3:10-3:15 candle.

Past results do not predict future results. Not financial advice.
"""

import time
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from backtest import SLIP_PCT, leg_cost
from sail_intraday_bot import (BAR_MIN, GROWW_TOTP_SECRET, GROWW_TOTP_TOKEN,
                               IST, candles_to_df, tick)
from sail_lab3 import rsi_series

FIRST_SIGNAL = dtime(9, 30)
LAST_SIGNAL = dtime(14, 25)       # last candle allowed to give a signal (ends 2:30)
LAST_BAR = dtime(14, 55)          # square off at the close of this candle
ATR_MULT, MIN_SL_PCT, MAX_SL_PCT = 1.5, 0.25, 1.2
MAX_TRADES_PER_DAY = 3
HISTORY_DAYS_NEEDED = 45          # calendar days of 5-min data for the daily trend

NAMES = {
    "bb": "5-min Bollinger + RSI reversal",
    "rsipb": "5-min RSI pullback",
    "vwap": "Pullback to VWAP",
    "orb": "Opening-range breakout",
    "ema15": "15-min EMA 9/21 crossover",
    "st15": "15-min Supertrend flip",
    "dip": "Buy the dip at the open",
    "ema": "5-min EMA 9/21 crossover with the EMA 50 trend",
    "st": "5-min Supertrend flip",
}
TREND_TEXT = {None: "", "none": "", "m15": " in the 15-min trend",
              "day": " in the daily trend", "m15day": " in the 15-min + daily trend"}


def combos():
    """Every rule set the trainer tests (30). Kept small on purpose:
    the more combinations tested, the easier it is to find a lucky one."""
    out = []
    for R in (1.0, 1.5):
        for tf in ("none", "m15", "m15day"):
            out.append(dict(kind="bb", rsi=25, tf=tf, R=R))
            out.append(dict(kind="rsipb", tf=tf, R=R))
        for tf in ("m15", "m15day"):
            out.append(dict(kind="vwap", tf=tf, R=R))
        out.append(dict(kind="dip", tf="day", R=R))
    for R in (1.5, 2.0):
        for tf in ("m15", "m15day"):
            out.append(dict(kind="orb", vol=1.5, tf=tf, R=R))
        for tf in ("none", "day"):
            out.append(dict(kind="ema15", tf=tf, am=2.5, R=R))
            out.append(dict(kind="st15", tf=tf, am=2.5, R=R))
    return out


def label(p):
    tf = p.get("tf")
    if p["kind"] == "dip":
        tf = None
    return f"{NAMES[p['kind']]}{TREND_TEXT.get(tf, '')}, target {p['R']:g}x risk"


# ------------------------------- data -----------------------------------
def login():
    from growwapi import GrowwAPI
    import pyotp
    totp = pyotp.TOTP(GROWW_TOTP_SECRET).now()
    return GrowwAPI(GrowwAPI.get_access_token(api_key=GROWW_TOTP_TOKEN, totp=totp))


def fetch_5min(g, symbol, start_date, end_dt, pause=0.35):
    """5-min candles from start_date 9:15 to end_dt, in 29-day requests."""
    cur = datetime.combine(start_date, dtime(9, 15), tzinfo=IST)
    frames = []
    while cur < end_dt:
        nxt = min(cur + timedelta(days=29), end_dt)
        resp = None
        for attempt in range(3):
            try:
                resp = g.get_historical_candles(
                    exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH,
                    groww_symbol=f"NSE-{symbol}",
                    start_time=cur.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time=nxt.strftime("%Y-%m-%d %H:%M:%S"),
                    candle_interval=g.CANDLE_INTERVAL_MIN_5, timeout=20)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        frames.append(candles_to_df((resp or {}).get("candles")))
        cur = nxt
        if pause:
            time.sleep(pause)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[[dtime(9, 15) <= t <= dtime(15, 25) for t in df.index.time]]


# ---------------------------- indicators --------------------------------
def _supertrend(H, L, C, atr, mult):
    hl2 = (H + L) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    n = len(C)
    fub, flb, d = ub.copy(), lb.copy(), np.ones(n)
    for i in range(1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or C[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or C[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if C[i] < flb[i] else 1
        else:
            d[i] = 1 if C[i] > fub[i] else -1
    return d


def _true_range(H, L, C):
    prev_c = C.shift(1)
    return pd.concat([H - L, (H - prev_c).abs(), (L - prev_c).abs()], axis=1).max(axis=1)


def indicators(df):
    """Continuous multi-day 5-min candles -> dict of numpy arrays (5-min,
    15-min mapped to 5-min, and yesterday's daily values)."""
    O, H, L, C = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    V = df["Volume"].astype(float)
    idx = df.index
    day_arr = np.array(idx.date)
    day = pd.Series(day_arr, index=idx)
    tod = np.array(idx.time)
    step = pd.Timedelta(minutes=BAR_MIN)

    # ---- 5-min ----
    tp = (H + L + C) / 3
    vwap = ((tp * V).groupby(day).cumsum()
            / V.groupby(day).cumsum().replace(0, np.nan)).fillna(tp)
    in_or = pd.Series([dtime(9, 15) <= t < dtime(9, 30) for t in tod], index=idx)
    orh = H.where(in_or).groupby(day).transform("max")
    orl = L.where(in_or).groupby(day).transform("min")
    tr = _true_range(H, L, C)
    bbm, bbs = C.rolling(20).mean(), C.rolling(20).std()
    ind = dict(
        index=idx, tod=tod, day=day_arr,
        O=O.to_numpy(), H=H.to_numpy(), L=L.to_numpy(), C=C.to_numpy(), V=V.to_numpy(),
        Cp=C.shift(1).to_numpy(), vwap=vwap.to_numpy(), orh=orh.to_numpy(), orl=orl.to_numpy(),
        dopen=O.groupby(day).transform("first").to_numpy(),
        e9=C.ewm(span=9, adjust=False).mean().to_numpy(),
        e21=C.ewm(span=21, adjust=False).mean().to_numpy(),
        e50=C.ewm(span=50, adjust=False).mean().to_numpy(),
        va=V.rolling(20, min_periods=5).mean().shift(1).to_numpy(),
        atr14=tr.rolling(14, min_periods=5).mean().to_numpy(),
        atr10=tr.ewm(alpha=1 / 10, adjust=False).mean().to_numpy(),
        rsi7=rsi_series(C, 7).to_numpy(),
        bb_lo=(bbm - 2 * bbs).to_numpy(), bb_hi=(bbm + 2 * bbs).to_numpy(),
    )
    ind["e9p"] = np.r_[np.nan, ind["e9"][:-1]]
    ind["e21p"] = np.r_[np.nan, ind["e21"][:-1]]
    ind["rsi7p"] = np.r_[np.nan, ind["rsi7"][:-1]]

    # ---- 15-min, built from 5-min; a 5-min candle only sees 15-min candles
    #      that had fully closed by the end of that 5-min candle ----
    mins = np.array([(t.hour * 60 + t.minute) - (9 * 60 + 15) for t in tod])
    bucket_start = idx - pd.to_timedelta(mins % 15, unit="m")
    g15 = df.groupby(bucket_start)
    m = pd.DataFrame({"O": g15["Open"].first(), "H": g15["High"].max(),
                      "L": g15["Low"].min(), "C": g15["Close"].last()}).astype(float)
    m_end = m.index + pd.Timedelta(minutes=15)
    m_end = np.minimum(m_end.values,                              # last bucket of the day
                       (m.index.normalize() + pd.Timedelta(hours=15, minutes=30)).values)
    m_tr = _true_range(m["H"], m["L"], m["C"])
    m_e9 = m["C"].ewm(span=9, adjust=False).mean().to_numpy()
    m_e21 = m["C"].ewm(span=21, adjust=False).mean().to_numpy()
    m_e20 = m["C"].ewm(span=20, adjust=False).mean().to_numpy()
    m_e50 = m["C"].ewm(span=50, adjust=False).mean().to_numpy()
    m_st = _supertrend(m["H"].to_numpy(), m["L"].to_numpy(), m["C"].to_numpy(),
                       m_tr.ewm(alpha=1 / 10, adjust=False).mean().to_numpy(), 3.0)
    bar_end = (idx + step).values
    j = np.searchsorted(m_end, bar_end, side="right") - 1        # last closed 15-min candle
    ok = j >= 0
    jj = np.where(ok, j, 0)

    def mapped(a):
        out = np.where(ok, a[jj], np.nan)
        return out

    ind["m_new"] = ok & (np.r_[-1, j[:-1]] != j)                  # a 15-min candle just closed
    ind["m_e9"], ind["m_e21"] = mapped(m_e9), mapped(m_e21)
    ind["m_e20"], ind["m_e50"] = mapped(m_e20), mapped(m_e50)
    jp = np.where(ok & (j >= 1), j - 1, 0)
    ind["m_e9p"] = np.where(ok & (j >= 1), m_e9[jp], np.nan)
    ind["m_e21p"] = np.where(ok & (j >= 1), m_e21[jp], np.nan)
    ind["m_st"] = np.where(ok, m_st[jj], 0)
    ind["m_stp"] = np.where(ok & (j >= 1), m_st[jp], 0)

    # ---- daily, built from 5-min; today only sees yesterday's values ----
    dclose = C.groupby(day).last()
    dsma20 = dclose.rolling(20).mean()
    drsi2 = rsi_series(dclose, 2)
    prev = pd.DataFrame({"c": dclose.shift(1), "sma": dsma20.shift(1), "rsi2": drsi2.shift(1)})
    prev.loc[dclose.index[:20], "rsi2"] = np.nan                  # RSI needs a few days to settle
    ind["d_c"] = prev["c"].reindex(day_arr).to_numpy()
    ind["d_sma"] = prev["sma"].reindex(day_arr).to_numpy()
    ind["d_rsi2"] = prev["rsi2"].reindex(day_arr).to_numpy()

    ind["ok_time"] = np.array([FIRST_SIGNAL <= t <= LAST_SIGNAL for t in tod])
    days, starts = np.unique(day_arr, return_index=True)
    bounds = list(starts) + [len(tod)]
    ind["days"] = [(d, bounds[k], bounds[k + 1]) for k, d in enumerate(days)]
    ind["_st"] = {}
    return ind


def supertrend_dir(ind, mult):
    if mult not in ind["_st"]:
        ind["_st"][mult] = _supertrend(ind["H"], ind["L"], ind["C"], ind["atr10"], mult)
    return ind["_st"][mult]


def trend(ind, tf):
    """(buy_allowed, sell_allowed) arrays for a trend filter."""
    n = len(ind["C"])
    up, dn = np.ones(n, bool), np.ones(n, bool)
    if tf in ("m15", "m15day"):
        up &= ind["m_e20"] > ind["m_e50"]
        dn &= ind["m_e20"] < ind["m_e50"]
    if tf in ("day", "m15day"):
        up &= ind["d_c"] > ind["d_sma"]
        dn &= ind["d_c"] < ind["d_sma"]
    return up, dn


def signals(ind, p):
    """+1 = BUY, -1 = SELL, 0 = nothing, for every candle (uses data up to it)."""
    O, H, L, C, V, va = (ind[k] for k in ("O", "H", "L", "C", "V", "va"))
    k, tf = p["kind"], p.get("tf")
    with np.errstate(invalid="ignore"):
        vol = (np.isnan(va) | (va == 0) | (V >= p.get("vol", 1.0) * va))
        t_up, t_dn = trend(ind, tf)
        if k == "orb":
            orh, orl, Cp, vw = ind["orh"], ind["orl"], ind["Cp"], ind["vwap"]
            up = (C > orh) & (Cp <= orh) & (C > vw) & vol
            dn = (C < orl) & (Cp >= orl) & (C < vw) & vol
        elif k == "ema":
            e9, e21, e50, e9p, e21p = (ind[x] for x in ("e9", "e21", "e50", "e9p", "e21p"))
            up = (e9 > e21) & (e9p <= e21p) & (C > e50) & vol
            dn = (e9 < e21) & (e9p >= e21p) & (C < e50) & vol
        elif k == "vwap":
            vw = ind["vwap"]
            up = (L <= vw * 1.0005) & (C > vw) & (C > O)
            dn = (H >= vw * 0.9995) & (C < vw) & (C < O)
            if tf is None:                                  # v1 rule: own EMA trend
                up &= ind["e21"] > ind["e50"]
                dn &= ind["e21"] < ind["e50"]
        elif k == "bb":
            r = ind["rsi7"]
            up = (C < ind["bb_lo"]) & (r < p["rsi"]) & (C > O)
            dn = (C > ind["bb_hi"]) & (r > 100 - p["rsi"]) & (C < O)
        elif k == "rsipb":
            r, rp = ind["rsi7"], ind["rsi7p"]
            up = (rp < 30) & (r >= 30) & (C > O)
            dn = (rp > 70) & (r <= 70) & (C < O)
        elif k == "ema15":
            new = ind["m_new"]
            up = new & (ind["m_e9"] > ind["m_e21"]) & (ind["m_e9p"] <= ind["m_e21p"])
            dn = new & (ind["m_e9"] < ind["m_e21"]) & (ind["m_e9p"] >= ind["m_e21p"])
        elif k == "st15":
            new = ind["m_new"]
            up = new & (ind["m_st"] == 1) & (ind["m_stp"] == -1)
            dn = new & (ind["m_st"] == -1) & (ind["m_stp"] == 1)
        elif k == "dip":
            at = np.array([t == FIRST_SIGNAL for t in ind["tod"]])    # the 9:30-9:35 candle
            r2 = ind["d_rsi2"]
            up = at & (r2 < 15) & (C > ind["dopen"])
            dn = at & (r2 > 85) & (C < ind["dopen"])
            t_up, t_dn = trend(ind, "day")
        else:  # "st": v1 5-min supertrend
            d = supertrend_dir(ind, p["mult"])
            dp = np.r_[d[0], d[:-1]]
            up, dn = (d == 1) & (dp == -1), (d == -1) & (dp == 1)
        up &= t_up
        dn &= t_dn
    s = np.zeros(len(C), dtype=int)
    s[up] = 1
    s[dn] = -1
    s[~ind["ok_time"]] = 0
    return s


# ------------------------------ simulation ------------------------------
def run_day(ind, s, p, i0, i1):
    """Trades for one day (candles i0..i1-1). Each trade: dict with side, entry,
    sl, tgt, entry_i, and exit/why/exit_i once closed (status open/closed)."""
    H, L, C, atr, tod = ind["H"], ind["L"], ind["C"], ind["atr14"], ind["tod"]
    am = p.get("am", ATR_MULT)
    trades, i = [], i0
    sig_idx = np.flatnonzero(s[i0:i1]) + i0
    while len(trades) < MAX_TRADES_PER_DAY:
        nxt = sig_idx[sig_idx >= i]
        if not len(nxt):
            break
        j = int(nxt[0])
        side = int(s[j])
        entry = tick(C[j])
        a = atr[j] if not np.isnan(atr[j]) else entry * MIN_SL_PCT / 100
        dist = min(max(am * a, entry * MIN_SL_PCT / 100), entry * MAX_SL_PCT / 100)
        t = dict(side="BUY" if side == 1 else "SELL", entry=entry,
                 sl=tick(entry - side * dist), tgt=tick(entry + side * p["R"] * dist),
                 entry_i=j, status="open", exit=None, why=None, exit_i=None)
        trades.append(t)
        k = j + 1
        while k < i1:
            hi, lo = H[k], L[k]
            if (lo <= t["sl"]) if side == 1 else (hi >= t["sl"]):
                t.update(exit=t["sl"], why="STOP")
            elif (hi >= t["tgt"]) if side == 1 else (lo <= t["tgt"]):
                t.update(exit=t["tgt"], why="TARGET")
            elif tod[k] >= LAST_BAR:
                t.update(exit=tick(C[k]), why="CLOSE")
            if t["exit"] is not None:
                t.update(status="closed", exit_i=k)
                break
            k += 1
        if t["status"] == "open":
            break                   # still running (live) or data ended
        i = k + 1
    return trades


def net_pnl(side, entry, exit_px, capital):
    """Net Rs for one intraday trade after Groww charges and slippage."""
    qty = max(1, int(capital // entry))
    s = SLIP_PCT / 100
    sign = 1 if side == "BUY" else -1
    e_fill, x_fill = entry * (1 + sign * s), exit_px * (1 - sign * s)
    es, xs = ("buy", "sell") if side == "BUY" else ("sell", "buy")
    return (sign * qty * (x_fill - e_fill)
            - leg_cost(qty * e_fill, es) - leg_cost(qty * x_fill, xs))


def backtest(ind, p, capital, day_from=None):
    """All closed trades of rule set p: list of (day, net, side)."""
    s = signals(ind, p)
    C = ind["C"]
    out = []
    for d, i0, i1 in ind["days"]:
        if day_from is not None and d < day_from:
            continue
        for t in run_day(ind, s, p, i0, i1):
            if t["status"] == "open":          # data gap: exit at the day's last close
                t.update(exit=tick(C[i1 - 1]), why="CLOSE")
            out.append((d, net_pnl(t["side"], t["entry"], t["exit"], capital), t["side"]))
    return out
