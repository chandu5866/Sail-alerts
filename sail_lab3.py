"""
SAIL lab 3: high win-rate ideas and slower (swing) trades.

PART 1 - intraday "small target, wide stop" ideas (mean reversion). These are
         built to win often (70-85% of trades) and the lab shows whether that
         still earns money after real charges.
PART 2 - swing trades on the daily chart (hold a few days to weeks, long
         only), with delivery-trade charges.

Everything is charged Groww costs and slippage. An idea only counts as
promising if it is profitable in ALL THREE equal time slices (oldest,
middle, newest). Win rate is shown next to profit so you can see both.

Past results do not predict future results. Not financial advice.
"""

import os
from datetime import time as dtime, timedelta

import numpy as np
import pandas as pd

import backtest as bt
import sail_intraday_bot as bot

DAYS = int(os.getenv("LAB_DAYS", "1800") or "1800")
SYMBOL = "SAIL"
INTRA_MIN_CALLS, INTRA_MIN_SLICE = 60, 12
SWING_MIN_CALLS, SWING_MIN_SLICE = 20, 4
WARMUP = 210


# --------------------------- shared helpers ---------------------------
def rsi_series(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def delivery_leg(value: float, side: str) -> float:
    """Approximate Groww charges for one delivery order (verify in the app)."""
    brokerage = min(20.0, 0.001 * value)
    exch = value * 0.0000297
    sebi = value * 0.000001
    ipft = value * 0.000001
    gst = 0.18 * (brokerage + exch)
    stt = 0.001 * value                      # 0.1% on both buy and sell
    stamp = 0.00015 * value if side == "buy" else 0.0
    dp = 16.5 if side == "sell" else 0.0     # depository charge per sell
    return brokerage + exch + sebi + ipft + gst + stt + stamp + dp


def net_single(side, entry, exit_price, kind):
    """Net Rs for one trade with a single exit, after charges and slippage."""
    qty = max(1, int(bt.CAPITAL / entry))
    s = bt.SLIP_PCT / 100
    sign = 1 if side == "BUY" else -1
    e_fill = entry * (1 + sign * s)
    x_fill = exit_price * (1 - sign * s)
    gross = sign * qty * (x_fill - e_fill)
    leg = bt.leg_cost if kind == "intraday" else delivery_leg
    es, xs = ("buy", "sell") if side == "BUY" else ("sell", "buy")
    return gross - leg(qty * e_fill, es) - leg(qty * x_fill, xs)


def thirds(days):
    ds = sorted(set(days))
    a, b = ds[len(ds) // 3], ds[2 * len(ds) // 3]
    return a, b


def slice_stats(trades, a, b):
    parts = [[t for t in trades if t["day"] < a],
             [t for t in trades if a <= t["day"] < b],
             [t for t in trades if t["day"] >= b]]
    return [(len(p), sum(t["net"] for t in p)) for p in parts]


# ------------------------- PART 1: intraday ---------------------------
INTRA_BASE = dict(kind="bb", rsi=20, bbk=2.0, dev=0.6, tgt=0.3, sl=1.0, hold=12,
                  start=dtime(9, 45), last=dtime(14, 30), max_trades=2)


def IV(name, **kw):
    d = dict(INTRA_BASE)
    d.update(kw)
    d["name"] = name
    return d


INTRA_VARIANTS = [
    IV("Band+RSI fade, target 0.25% / stop 0.8%", tgt=0.25, sl=0.8),
    IV("Band+RSI fade, target 0.3% / stop 1.0%", tgt=0.3, sl=1.0),
    IV("Band+RSI fade, target 0.4% / stop 1.2%", tgt=0.4, sl=1.2),
    IV("Band+RSI fade, target 0.5% / stop 1.5%", tgt=0.5, sl=1.5),
    IV("Band+RSI(15) fade, target 0.4% / stop 1.2%", rsi=15, tgt=0.4, sl=1.2),
    IV("Band+RSI(25) fade, target 0.4% / stop 1.2%", rsi=25, tgt=0.4, sl=1.2),
    IV("Band(1.5) + RSI(30) fade, target 0.4% / stop 1.2%", bbk=1.5, rsi=30, tgt=0.4, sl=1.2),
    IV("Band(1.5) + RSI(30) fade, target 0.5% / stop 1.5%", bbk=1.5, rsi=30, tgt=0.5, sl=1.5),
    IV("VWAP stretch 0.6% fade, target 0.3% / stop 1.0%", kind="vwapdev", dev=0.6, tgt=0.3, sl=1.0),
    IV("VWAP stretch 0.8% fade, target 0.4% / stop 1.2%", kind="vwapdev", dev=0.8, tgt=0.4, sl=1.2),
    IV("VWAP stretch 1.0% fade, target 0.5% / stop 1.5%", kind="vwapdev", dev=1.0, tgt=0.5, sl=1.5),
    IV("VWAP stretch 0.6% fade, target 0.5% / stop 2.0%", kind="vwapdev", dev=0.6, tgt=0.5, sl=2.0),
]


def prepare_intraday(df):
    d = df.copy()
    c = d["Close"]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["bbm"], d["bbs"] = mid, sd
    d["rsi"] = rsi_series(c, 7)
    tp = (d["High"] + d["Low"] + c) / 3
    day = d.index.date
    pv = (tp * d["Volume"]).groupby(day).cumsum()
    vv = d["Volume"].groupby(day).cumsum()
    d["vwap"] = (pv / vv.replace(0, np.nan)).fillna(tp)
    out = []
    for k, (dt, g) in enumerate(d.groupby(day)):
        if k < 6 or len(g) < 60:
            continue
        out.append(dict(
            day=dt, idx=g.index, times=[t.time() for t in g.index],
            O=g["Open"].to_numpy(float), H=g["High"].to_numpy(float),
            L=g["Low"].to_numpy(float), C=g["Close"].to_numpy(float),
            rsi=g["rsi"].to_numpy(float), bbm=g["bbm"].to_numpy(float),
            bbs=g["bbs"].to_numpy(float), vw=g["vwap"].to_numpy(float)))
    return out


def intraday_day(c, P):
    idx, times = c["idx"], c["times"]
    H, L, C, O = c["H"], c["L"], c["C"], c["O"]
    n = len(C)
    last_bar = max((i for i in range(n) if times[i] <= dtime(15, 0)), default=None)
    if last_bar is None:
        return []
    trades, i = [], 0
    while i < n and len(trades) < P["max_trades"]:
        if times[i] < P["start"]:
            i += 1
            continue
        if (idx[i] + timedelta(minutes=bot.BAR_MIN)).time() > P["last"]:
            break
        side = None
        if P["kind"] == "bb":
            lo_b = c["bbm"][i] - P["bbk"] * c["bbs"][i]
            hi_b = c["bbm"][i] + P["bbk"] * c["bbs"][i]
            if C[i] < lo_b and c["rsi"][i] < P["rsi"] and C[i] > O[i]:
                side = "BUY"
            elif C[i] > hi_b and c["rsi"][i] > 100 - P["rsi"] and C[i] < O[i]:
                side = "SELL"
        else:
            dev = (C[i] - c["vw"][i]) / c["vw"][i] * 100
            if dev <= -P["dev"] and C[i] > O[i]:
                side = "BUY"
            elif dev >= P["dev"] and C[i] < O[i]:
                side = "SELL"
        if side is None:
            i += 1
            continue
        sign = 1 if side == "BUY" else -1
        entry = bot.tick(C[i])
        sl = bot.tick(entry - sign * entry * P["sl"] / 100)
        tg = bot.tick(entry + sign * entry * P["tgt"] / 100)
        end = min(last_bar, i + P["hold"])
        why, exit_price, exit_j = None, None, end
        for j in range(i + 1, end + 1):
            sl_hit = L[j] <= sl if side == "BUY" else H[j] >= sl
            tg_hit = H[j] >= tg if side == "BUY" else L[j] <= tg
            if sl_hit:                      # conservative: stop first
                why, exit_price, exit_j = "SL", sl, j
                break
            if tg_hit:
                why, exit_price, exit_j = "TGT", tg, j
                break
        if why is None:
            why, exit_price = "time", float(C[end])
        net = net_single(side, entry, exit_price, "intraday")
        trades.append(dict(day=c["day"], side=side, why=why, net=net,
                           hit=(why == "TGT")))
        i = exit_j + 1
    return trades


def run_intraday(days):
    a, b = thirds([c["day"] for c in days])
    rows = []
    for P in INTRA_VARIANTS:
        tr = []
        for c in days:
            tr += intraday_day(c, P)
        wins = sum(1 for t in tr if t["net"] > 0)
        hits = sum(1 for t in tr if t["hit"])
        rows.append(dict(name=P["name"], calls=len(tr),
                         win=(wins / len(tr) * 100 if tr else 0.0),
                         hit=(hits / len(tr) * 100 if tr else 0.0),
                         sl=slice_stats(tr, a, b)))
    return rows


# --------------------------- PART 2: swing ----------------------------
SWING_VARIANTS = [
    dict(name="RSI(2) dip in uptrend, stop 6%", kind="rsi2", rsi=10, sl=6.0, hold=7),
    dict(name="RSI(2) deep dip (<5), stop 6%", kind="rsi2", rsi=5, sl=6.0, hold=7),
    dict(name="RSI(2) dip, stop 4%", kind="rsi2", rsi=10, sl=4.0, hold=7),
    dict(name="Pullback to 20-day average, stop 5%", kind="pull", sl=5.0, hold=12),
    dict(name="Pullback to 20-day average, stop 8%", kind="pull", sl=8.0, hold=15),
    dict(name="20-day breakout, 10-day low exit, stop 8%", kind="brk", sl=8.0, hold=60),
    dict(name="20-day breakout, 10-day low exit, stop 5%", kind="brk", sl=5.0, hold=60),
]


def daily_frame(df):
    g = df.groupby(df.index.date)
    d = pd.DataFrame({"O": g["Open"].first(), "H": g["High"].max(),
                      "L": g["Low"].min(), "C": g["Close"].last()})
    d = d[g.size() >= 60]
    d["sma20"] = d["C"].rolling(20).mean()
    d["sma50"] = d["C"].rolling(50).mean()
    d["sma200"] = d["C"].rolling(200).mean()
    d["rsi2"] = rsi_series(d["C"], 2)
    d["rsi14"] = rsi_series(d["C"], 14)
    d["hh20"] = d["H"].rolling(20).max().shift(1)
    d["ll10"] = d["L"].rolling(10).min().shift(1)
    return d


def swing_signal(d, t, S):
    C, L = d["C"], d["L"]
    if S["kind"] == "rsi2":
        return C[t] > d["sma200"][t] and d["rsi2"][t] < S["rsi"]
    if S["kind"] == "pull":
        return (C[t] > d["sma50"][t] and d["sma20"][t] > d["sma50"][t]
                and min(L[t - 1], L[t]) <= d["sma20"][t] * 1.005
                and d["rsi14"][t] < 50 and C[t] > C[t - 1])
    return C[t] > d["hh20"][t] and C[t] > d["sma50"][t]


def swing_exit_rule(d, k, S):
    if S["kind"] == "rsi2":
        return d["C"][k] > d["H"][k - 1]
    if S["kind"] == "pull":
        return d["rsi14"][k] > 60
    return d["C"][k] < d["ll10"][k]


def swing_trades(dframe, S):
    arr = {k: dframe[k].to_numpy(float) for k in dframe.columns}
    dates = list(dframe.index)
    n = len(dates)
    O, H, L, C = arr["O"], arr["H"], arr["L"], arr["C"]
    trades, t = [], WARMUP
    while t < n - 2:
        if not swing_signal(arr, t, S):
            t += 1
            continue
        e = t + 1
        entry = O[e]
        stop = entry * (1 - S["sl"] / 100)
        exit_price, exit_k, why = None, None, None
        for k in range(e, n):
            if L[k] <= stop:
                exit_price, exit_k, why = min(O[k], stop), k, "SL"
                break
            if k > e and swing_exit_rule(arr, k, S):
                exit_price = O[k + 1] if k + 1 < n else C[k]
                exit_k, why = min(k + 1, n - 1), "rule"
                break
            if k - e + 1 >= S["hold"]:
                exit_price = O[k + 1] if k + 1 < n else C[k]
                exit_k, why = min(k + 1, n - 1), "time"
                break
        if exit_price is None:
            exit_price, exit_k, why = C[n - 1], n - 1, "end"
        trades.append(dict(day=dates[e], why=why, hold=exit_k - e,
                           hit=bool(exit_price > entry),
                           net=net_single("BUY", entry, exit_price, "delivery")))
        t = max(exit_k, t + 1)
    return trades


def run_swing(dframe):
    a, b = thirds(list(dframe.index)[WARMUP:])
    rows = []
    for S in SWING_VARIANTS:
        tr = swing_trades(dframe, S)
        wins = sum(1 for t in tr if t["net"] > 0)
        hits = sum(1 for t in tr if t["hit"])
        avg_hold = (sum(t["hold"] for t in tr) / len(tr)) if tr else 0
        rows.append(dict(name=S["name"], calls=len(tr),
                         win=(wins / len(tr) * 100 if tr else 0.0),
                         hit=(hits / len(tr) * 100 if tr else 0.0),
                         sl=slice_stats(tr, a, b), hold=avg_hold))
    return rows


# ----------------------------- reporting ------------------------------
def line(r, extra=""):
    s = r["sl"]
    return (f"{r['name']}: {r['calls']} trades, hit {r['hit']:.0f}%, "
            f"profit-win {r['win']:.0f}%{extra}, "
            f"old {s[0][1]:+,.0f}, mid {s[1][1]:+,.0f}, new {s[2][1]:+,.0f}")


def passes(r, min_calls, min_slice):
    return (r["calls"] >= min_calls and all(n >= min_slice for n, _ in r["sl"])
            and all(p > 0 for _, p in r["sl"]))


def report(intra, swing, ndays, hold_pct):
    msgs = []
    head = [f"🧪 SAIL lab 3 ({ndays} trading days), Rs {bt.CAPITAL:,.0f} per trade,",
            "after charges + slippage. Slices: oldest / middle / newest third.",
            f"For reference, just holding SAIL moved {hold_pct:+.0f}% over this period.", "",
            "PART 1 - intraday small-target ideas:"]
    msgs.append("\n".join(head + [line(r) for r in
                                  sorted(intra, key=lambda r: -sum(p for _, p in r["sl"]))]))
    s_lines = ["PART 2 - swing trades (daily chart, long only):"]
    s_lines += [line(r, f", avg hold {r['hold']:.0f}d") for r in
                sorted(swing, key=lambda r: -sum(p for _, p in r["sl"]))]
    msgs.append("\n".join(s_lines))
    good_i = [r for r in intra if passes(r, INTRA_MIN_CALLS, INTRA_MIN_SLICE)]
    good_s = [r for r in swing if passes(r, SWING_MIN_CALLS, SWING_MIN_SLICE)]
    out = []
    for tag, good in (("intraday", good_i), ("swing", good_s)):
        for r in good[:3]:
            out.append(f"✅ {tag}: {r['name']} - hit {r['hit']:.0f}%, "
                       f"total {sum(p for _, p in r['sl']):+,.0f}")
    if out:
        out += ["", "Testing many ideas can find lucky ones. Candidates to "
                "watch, not proof."]
        msgs.append("\n".join(out))
    else:
        msgs.append("❌ Nothing was profitable in all three slices after costs.")
    high = [r for r in intra + swing if r["hit"] >= 75 and r["calls"] >= 20]
    msgs.append("'hit' = trades that reached their target (or closed in "
                "profit before charges); 'profit-win' = trades that still "
                "made money after charges.")
    if high:
        msgs.append("Ideas with 75%+ hit rate: " + "; ".join(
            f"{r['name']} (hit {r['hit']:.0f}%, after-charges total "
            f"{sum(p for _, p in r['sl']):+,.0f})" for r in high))
    return msgs


def main() -> None:
    g = bt.login()
    df = bt.fetch_symbol(g, SYMBOL, DAYS)
    if df.empty:
        bot.send("🧪 SAIL lab 3: Groww returned no candles.")
        return
    days = prepare_intraday(df)
    dframe = daily_frame(df)
    hold_pct = (dframe["C"].iloc[-1] / dframe["C"].iloc[0] - 1) * 100
    for m in report(run_intraday(days), run_swing(dframe), len(dframe), hold_pct):
        bot.send(m)


if __name__ == "__main__":
    main()
