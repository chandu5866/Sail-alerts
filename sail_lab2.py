"""
SAIL strategy lab 2: trend and pullback ideas.

Tests the kind of setup on your chart (EMA 9 / 21 / 50 plus volume) and two
other classic ideas (VWAP pullback, opening gap fade) on about 3 years of
SAIL 5-minute candles from Groww.

Every idea is charged real Groww costs and slippage. An idea only counts as
"promising" if it is profitable in ALL THREE periods:
  A = oldest part, B = middle part, new = newest LAB_TEST_DAYS days.
Up to 2 trades a day are allowed, the same exits as the live bot (half off
at Target 1, stop moves to cost, square-off at 3:05 PM).

Past results do not predict future results. Not financial advice.
"""

import os
from datetime import time as dtime, timedelta

import numpy as np
import pandas as pd

import backtest as bt
import sail_intraday_bot as bot

DAYS = int(os.getenv("LAB_DAYS", "1000") or "1000")
TEST_DAYS = int(os.getenv("LAB_TEST_DAYS", "180") or "180")
SYMBOL = "SAIL"
MIN_TOTAL_CALLS = 40
MIN_PERIOD_CALLS = 8
WARMUP_DAYS = 4

BASE = dict(kind="stack", start=dtime(9, 45), last=dtime(14, 30),
            vol_mult=1.0, t=(1.5, 3.0), rmin=0.4, rmax=1.0, max_trades=2,
            use_vwap=False, gap=0.5)


def V(name, **kw):
    d = dict(BASE)
    d.update(kw)
    d["name"] = name
    return d


MORNING = dtime(11, 30)
VARIANTS = [
    V("EMA stack pullback", kind="stack"),
    V("EMA stack pullback, volume 1.5x", kind="stack", vol_mult=1.5),
    V("EMA stack pullback + above/below VWAP", kind="stack", use_vwap=True),
    V("EMA stack pullback, targets 1x/2x", kind="stack", t=(1.0, 2.0)),
    V("EMA stack pullback, morning only", kind="stack", last=MORNING),
    V("EMA stack pullback, wider SL 0.6-1.2%", kind="stack", rmin=0.6, rmax=1.2),
    V("EMA stack + VWAP + volume 1.5x", kind="stack", use_vwap=True, vol_mult=1.5),
    V("EMA 9/21 cross + EMA50 filter, volume 1.5x", kind="cross", vol_mult=1.5),
    V("EMA 9/21 cross + EMA50 filter, volume 2x", kind="cross", vol_mult=2.0),
    V("EMA cross, volume 1.5x, targets 2x/4x", kind="cross", vol_mult=1.5, t=(2.0, 4.0)),
    V("VWAP pullback in trend", kind="vwap"),
    V("VWAP pullback, volume 1.2x", kind="vwap", vol_mult=1.2),
    V("VWAP pullback, targets 1x/2x", kind="vwap", t=(1.0, 2.0)),
    V("VWAP pullback, morning only", kind="vwap", last=MORNING),
    V("Gap fade 0.5%", kind="gap", gap=0.5, max_trades=1),
    V("Gap fade 0.8%", kind="gap", gap=0.8, max_trades=1),
    V("Gap fade 1.0%", kind="gap", gap=1.0, max_trades=1),
]


# --------------------------- preparation ------------------------------
def prepare(df):
    d = df.copy()
    d["ema9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["Close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["volavg"] = d["Volume"].rolling(20, min_periods=5).mean().shift(1)
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    day = d.index.date
    pv = (tp * d["Volume"]).groupby(day).cumsum()
    vv = d["Volume"].groupby(day).cumsum()
    d["vwap"] = (pv / vv.replace(0, np.nan)).fillna(tp)
    days, prev = [], None
    for k, (dt, g) in enumerate(d.groupby(day)):
        times = [t.time() for t in g.index]
        ok = len([t for t in times if bot.OPEN_T <= t < bot.RANGE_END]) >= 3
        if ok and prev is not None and k >= WARMUP_DAYS:
            days.append(dict(
                day=dt, idx=g.index, times=times, prev_close=prev,
                O=g["Open"].to_numpy(float), H=g["High"].to_numpy(float),
                L=g["Low"].to_numpy(float), C=g["Close"].to_numpy(float),
                Vo=g["Volume"].to_numpy(float), e9=g["ema9"].to_numpy(float),
                e21=g["ema21"].to_numpy(float), e50=g["ema50"].to_numpy(float),
                va=g["volavg"].to_numpy(float), vw=g["vwap"].to_numpy(float)))
        prev = float(g["Close"].iloc[-1])
    return days


# ----------------------------- signals ---------------------------------
def vol_ok(c, i, mult):
    va = c["va"][i]
    return np.isnan(va) or va == 0 or c["Vo"][i] >= mult * va


def signal(c, i, P):
    """Returns (side, raw_sl_price, custom_targets_or_None) or None."""
    H, L, C, O = c["H"], c["L"], c["C"], c["O"]
    e9, e21, e50, vw = c["e9"], c["e21"], c["e50"], c["vw"]
    kind, tol = P["kind"], 0.0005
    lo4, hi4 = i - 3, i + 1

    if kind == "stack" and i >= 4:
        if not vol_ok(c, i, P["vol_mult"]):
            return None
        if e9[i] > e21[i] > e50[i] and L[lo4:hi4].min() <= e21[i] * (1 + tol) \
                and C[i] > e9[i] and C[i] > O[i] and (not P["use_vwap"] or C[i] > vw[i]):
            return "BUY", float(L[i - 4:hi4].min()), None
        if e9[i] < e21[i] < e50[i] and H[lo4:hi4].max() >= e21[i] * (1 - tol) \
                and C[i] < e9[i] and C[i] < O[i] and (not P["use_vwap"] or C[i] < vw[i]):
            return "SELL", float(H[i - 4:hi4].max()), None

    elif kind == "cross" and i >= 5:
        if not vol_ok(c, i, P["vol_mult"]):
            return None
        if e9[i - 1] <= e21[i - 1] and e9[i] > e21[i] and C[i] > e50[i]:
            return "BUY", float(L[i - 4:hi4].min()), None
        if e9[i - 1] >= e21[i - 1] and e9[i] < e21[i] and C[i] < e50[i]:
            return "SELL", float(H[i - 4:hi4].max()), None

    elif kind == "vwap" and i >= 4:
        if not vol_ok(c, i, P["vol_mult"]):
            return None
        if e21[i] > e50[i] and L[i] <= vw[i] * (1 + tol) and C[i] > vw[i] and C[i] > O[i]:
            return "BUY", float(L[i - 4:hi4].min()), None
        if e21[i] < e50[i] and H[i] >= vw[i] * (1 - tol) and C[i] < vw[i] and C[i] < O[i]:
            return "SELL", float(H[i - 4:hi4].max()), None

    elif kind == "gap" and c["times"][i] == dtime(9, 30):
        day_open, pc = O[0], c["prev_close"]
        gap = (day_open - pc) / pc * 100
        if gap <= -P["gap"] and C[i] > day_open:
            return "BUY", float(L[: i + 1].min()), ("gap", pc)
        if gap >= P["gap"] and C[i] < day_open:
            return "SELL", float(H[: i + 1].max()), ("gap", pc)
    return None


# ------------------------------ engine ----------------------------------
def run_day(c, P):
    idx, times = c["idx"], c["times"]
    H, L, C = c["H"], c["L"], c["C"]
    n = len(C)
    last_bar = max((i for i in range(n) if times[i] <= dtime(15, 0)), default=None)
    if last_bar is None:
        return []
    trades, i = [], 0
    while i < n and len(trades) < P["max_trades"]:
        if times[i] < P["start"] and P["kind"] != "gap":
            i += 1
            continue
        bar_end = idx[i] + timedelta(minutes=bot.BAR_MIN)
        limit = dtime(9, 40) if P["kind"] == "gap" else P["last"]
        if bar_end.time() > limit:
            break
        if P["kind"] == "gap" and times[i] != dtime(9, 30):
            i += 1
            continue
        s = signal(c, i, P)
        if s is None:
            i += 1
            continue
        side, raw_sl, custom = s
        price = float(C[i])
        sign = 1 if side == "BUY" else -1
        risk = min(max(abs(price - raw_sl), price * P["rmin"] / 100),
                   price * P["rmax"] / 100)
        entry = bot.tick(price)
        sl = bot.tick(entry - sign * risk)
        if custom:
            pc = custom[1]
            dist = sign * (pc - entry)
            if dist < 1.5 * risk:
                i += 1
                continue
            t1 = bot.tick(entry + sign * dist * 0.5)
            t2 = bot.tick(entry + sign * dist)
        else:
            t1 = bot.tick(entry + sign * risk * P["t"][0])
            t2 = bot.tick(entry + sign * risk * P["t"][1])
        trade = dict(day=c["day"], side=side, entry=entry, sl=sl, t1=t1, t2=t2,
                     t1_hit=False)
        cur_sl, exit_j = sl, None
        for j in range(i + 1, last_bar + 1):
            hi, lo = H[j], L[j]
            sl_hit = lo <= cur_sl if side == "BUY" else hi >= cur_sl
            t1_hit = hi >= t1 if side == "BUY" else lo <= t1
            t2_hit = hi >= t2 if side == "BUY" else lo <= t2
            if sl_hit:
                trade["why"] = "T1 then cost" if trade["t1_hit"] else "SL"
                exit_j = j
                break
            if t2_hit:
                trade["why"] = "T2"
                exit_j = j
                break
            if t1_hit and not trade["t1_hit"]:
                trade["t1_hit"] = True
                cur_sl = entry
        if exit_j is None:
            trade["why"] = "square-off"
            trade["last"] = float(C[last_bar])
            exit_j = last_bar
        trade["net"] = bt.evaluate(trade)
        trades.append(trade)
        i = exit_j + 1
    return trades


def run_variant(days, P):
    out = []
    for c in days:
        out += run_day(c, P)
    return out


# ----------------------------- reporting --------------------------------
def summarize(days):
    all_days = [c["day"] for c in days]
    new_cut = all_days[-1] - timedelta(days=TEST_DAYS)
    older = [d for d in all_days if d < new_cut]
    mid = older[len(older) // 2]
    rows = []
    for P in VARIANTS:
        tr = run_variant(days, P)
        a = [t for t in tr if t["day"] < mid]
        b = [t for t in tr if mid <= t["day"] < new_cut]
        n = [t for t in tr if t["day"] >= new_cut]
        wins = sum(1 for t in tr if t["net"] > 0)
        rows.append(dict(
            name=P["name"], calls=len(tr), win=(wins / len(tr) * 100 if tr else 0),
            A=sum(t["net"] for t in a), B=sum(t["net"] for t in b),
            N=sum(t["net"] for t in n), ca=len(a), cb=len(b), cn=len(n)))
    return rows, len(days)


def report(rows, ndays):
    rows_s = sorted(rows, key=lambda r: r["A"] + r["B"], reverse=True)
    head = [f"🧪 SAIL lab 2: {ndays} trading days, Rs {bt.CAPITAL:,.0f} per trade,",
            "after Groww charges + slippage.",
            f"A = oldest part, B = middle, new = newest {TEST_DAYS} days.", ""]
    lines = [f"{r['name']}: {r['calls']} calls, win {r['win']:.0f}%, "
             f"A {r['A']:+,.0f}, B {r['B']:+,.0f}, new {r['N']:+,.0f}" for r in rows_s]
    out = ["\n".join(head + lines)]
    good = [r for r in rows if r["A"] > 0 and r["B"] > 0 and r["N"] > 0
            and r["calls"] >= MIN_TOTAL_CALLS
            and min(r["ca"], r["cb"], r["cn"]) >= MIN_PERIOD_CALLS]
    good.sort(key=lambda r: r["A"] + r["B"] + r["N"], reverse=True)
    if good:
        txt = ["✅ Profitable in ALL THREE periods:"]
        for r in good[:3]:
            txt.append(f"{r['name']}: win {r['win']:.0f}%, "
                       f"total {r['A'] + r['B'] + r['N']:+,.0f}")
        txt += ["", "Testing many ideas can find lucky ones, so these are "
                "candidates to watch, not proof."]
        out.append("\n".join(txt))
    else:
        out.append("❌ No idea was profitable in all three periods after costs.")
    return out


def main() -> None:
    g = bt.login()
    df = bt.fetch_symbol(g, SYMBOL, DAYS)
    if df.empty:
        bot.send("🧪 SAIL lab 2: Groww returned no candles.")
        return
    rows, nd = summarize(prepare(df))
    for m in report(rows, nd):
        bot.send(m)


if __name__ == "__main__":
    main()
