"""
Intraday trading engine shared by trainer.py (the weekly study) and
live_bot.py (the live Telegram calls).

Both run the SAME functions on completed 5-minute candles, and a signal on
candle i only uses candles up to i, so what the trainer tests is exactly what
the live bot sends.

Strategies (each can BUY or SELL-short, intraday only)
  orb  : opening-range breakout. Price closes beyond the 9:15-9:30 high/low
         for the first time, on the right side of VWAP, with volume.
  ema  : EMA 9 crosses EMA 21 in the direction of EMA 50, with volume.
  vwap : trend pullback. EMA 21 above EMA 50 (uptrend), price dips to VWAP and
         closes back above it with a green candle (mirror for shorts).
  bb   : mean reversion. Close outside the Bollinger band (20, 2), RSI(7)
         extreme, and the candle turns back (green below the band = BUY).
  st   : Supertrend (ATR 10) flips direction.
Trade rules
  Entry at the close of the signal candle. Signals only from 9:30 to 2:30 PM.
  Stop = 1.5 x ATR(14), kept between 0.25% and 1.2% of price.
  Target = R x stop distance. Stop assumed first if both are hit in a candle.
  One position per stock at a time, at most 3 trades per stock per day.
  Anything still open is squared off at the close of the 3:10-3:15 candle.

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
LAST_BAR = dtime(15, 10)          # square off at the close of this candle
ATR_MULT, MIN_SL_PCT, MAX_SL_PCT = 1.5, 0.25, 1.2
MAX_TRADES_PER_DAY = 3

NAMES = {
    "orb": "Opening-range breakout",
    "ema": "EMA 9/21 crossover with the EMA 50 trend",
    "vwap": "Trend pullback to VWAP",
    "bb": "Bollinger band + RSI reversal",
    "st": "Supertrend flip",
}


def combos():
    """Every rule set the trainer tests (27). Kept small on purpose:
    the more combinations tested, the easier it is to find a lucky one."""
    out = []
    for R in (1.0, 1.5, 2.0):
        for k in (1.0, 1.5):
            out.append(dict(kind="orb", vol=k, R=R))
            out.append(dict(kind="ema", vol=k, R=R))
        out.append(dict(kind="vwap", R=R))
        for m in (2.0, 3.0):
            out.append(dict(kind="st", mult=m, R=R))
    for R in (0.75, 1.0, 1.5):
        for lvl in (25, 15):
            out.append(dict(kind="bb", rsi=lvl, R=R))
    return out


def label(p):
    extra = {"orb": f", volume {p.get('vol', 1)}x", "ema": f", volume {p.get('vol', 1)}x",
             "st": f" ({p.get('mult')}x ATR)", "bb": f", RSI {p.get('rsi')}/{100 - p.get('rsi', 0)}",
             "vwap": ""}[p["kind"]]
    return f"{NAMES[p['kind']]}{extra}, target {p['R']:g}x risk"


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
def indicators(df):
    """Continuous multi-day 5-min candles -> dict of numpy arrays."""
    O, H, L, C = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    V = df["Volume"].astype(float)
    day = pd.Series(df.index.date, index=df.index)
    tod = np.array(df.index.time)
    tp = (H + L + C) / 3
    vwap = ((tp * V).groupby(day).cumsum()
            / V.groupby(day).cumsum().replace(0, np.nan)).fillna(tp)
    in_or = pd.Series([dtime(9, 15) <= t < dtime(9, 30) for t in tod], index=df.index)
    orh = H.where(in_or).groupby(day).transform("max")
    orl = L.where(in_or).groupby(day).transform("min")
    prev_c = C.shift(1)
    tr = pd.concat([H - L, (H - prev_c).abs(), (L - prev_c).abs()], axis=1).max(axis=1)
    bbm, bbs = C.rolling(20).mean(), C.rolling(20).std()
    ind = dict(
        index=df.index, tod=tod, day=np.array(df.index.date),
        O=O.to_numpy(), H=H.to_numpy(), L=L.to_numpy(), C=C.to_numpy(), V=V.to_numpy(),
        Cp=prev_c.to_numpy(), vwap=vwap.to_numpy(), orh=orh.to_numpy(), orl=orl.to_numpy(),
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
    ind["ok_time"] = np.array([FIRST_SIGNAL <= t <= LAST_SIGNAL for t in tod])
    days, starts = np.unique(ind["day"], return_index=True)
    bounds = list(starts) + [len(tod)]
    ind["days"] = [(d, bounds[k], bounds[k + 1]) for k, d in enumerate(days)]
    ind["_st"] = {}
    return ind


def supertrend_dir(ind, mult):
    if mult in ind["_st"]:
        return ind["_st"][mult]
    H, L, C, atr = ind["H"], ind["L"], ind["C"], ind["atr10"]
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
    ind["_st"][mult] = d
    return d


def signals(ind, p):
    """+1 = BUY, -1 = SELL, 0 = nothing, for every candle (uses data up to it)."""
    O, H, L, C, V, va = (ind[k] for k in ("O", "H", "L", "C", "V", "va"))
    with np.errstate(invalid="ignore"):
        vol = (np.isnan(va) | (va == 0) | (V >= p.get("vol", 1.0) * va))
        k = p["kind"]
        if k == "orb":
            orh, orl, Cp, vw = ind["orh"], ind["orl"], ind["Cp"], ind["vwap"]
            up = (C > orh) & (Cp <= orh) & (C > vw) & vol
            dn = (C < orl) & (Cp >= orl) & (C < vw) & vol
        elif k == "ema":
            e9, e21, e50, e9p, e21p = (ind[x] for x in ("e9", "e21", "e50", "e9p", "e21p"))
            up = (e9 > e21) & (e9p <= e21p) & (C > e50) & vol
            dn = (e9 < e21) & (e9p >= e21p) & (C < e50) & vol
        elif k == "vwap":
            vw, e21, e50 = ind["vwap"], ind["e21"], ind["e50"]
            up = (e21 > e50) & (L <= vw * 1.0005) & (C > vw) & (C > O)
            dn = (e21 < e50) & (H >= vw * 0.9995) & (C < vw) & (C < O)
        elif k == "bb":
            r = ind["rsi7"]
            up = (C < ind["bb_lo"]) & (r < p["rsi"]) & (C > O)
            dn = (C > ind["bb_hi"]) & (r > 100 - p["rsi"]) & (C < O)
        else:  # supertrend
            d = supertrend_dir(ind, p["mult"])
            dp = np.r_[d[0], d[:-1]]
            up, dn = (d == 1) & (dp == -1), (d == -1) & (dp == 1)
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
        dist = min(max(ATR_MULT * a, entry * MIN_SL_PCT / 100), entry * MAX_SL_PCT / 100)
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
