"""Stocks in Play study: 5-minute opening-range breakout, only on shares with unusual opening volume.

Based on Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading Strategy For The U.S. Equity Market".
Rules, tested on NSE with Groww 5-min data, after charges:
  1. In play: the 9:15-9:20 candle's volume divided by its average over the last 14 days
     (relative volume). Keep shares with relative volume >= 1 and take the top N.
  2. Direction: first candle closed up -> only BUY; closed down -> only SELL.
  3. Entry: BUY when price breaks the first candle's high (SELL: breaks its low), any time until 14:55.
  4. Stop: a small fraction of the share's daily ATR(14) away from entry. No target - winners run.
  5. Exit: stop, or square-off at 15:00.
Chosen on everything before the newest LOCK_DAYS days, then checked on those 🔒 locked days.

Run:  python sip.py      (sends the report to Telegram)
"""
import os
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

import engine as E
from backtest import SLIP_PCT, leg_cost
from ml import POOL, fetch_long
from sail_intraday_bot import IST, send

EXTRA = ["M&M", "ULTRACEMCO", "TRENT", "HAL", "RECLTD", "PFC", "IRCTC", "IDFCFIRSTB", "INDUSINDBK",
         "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "CIPLA", "DRREDDY", "DIVISLAB", "GRASIM",
         "SHRIRAMFIN", "CHOLAFIN", "LUPIN", "AUROPHARMA"]
SYMBOLS = list(dict.fromkeys(POOL + EXTRA))
DAYS = 1100                      # ~3 years if Groww has it
LOCK_DAYS = 40
CAPITAL = float(os.getenv("BACKTEST_CAPITAL", "200000"))   # total, split across the day's positions
RISK = 1000.0                    # Rs lost per day if every stop hits (split across positions)
TOP_N = [1, 3, 5]
STOP_ATR = [0.05, 0.10, 0.20]    # stop distance as a fraction of daily ATR(14)
OR_BAR, FIRST, LAST = dtime(9, 15), dtime(9, 20), dtime(14, 55)


def candidates(sym, df):
    """One row per day: relative volume, direction, opening range and the rest of the day's candles."""
    if df.empty:
        return []
    df = df.copy()
    df["day"] = df.index.date
    d = df.groupby("day").agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    pc = d["C"].shift(1)
    tr = pd.concat([d["H"] - d["L"], (d["H"] - pc).abs(), (d["L"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)                          # only past days
    first = df[df.index.time == OR_BAR].set_index("day")
    rv = first["Volume"] / first["Volume"].rolling(14).mean().shift(1)
    out = []
    for day, g in df.groupby("day"):
        if day not in first.index or pd.isna(rv.get(day)) or pd.isna(atr.get(day)) or atr[day] <= 0:
            continue
        f = first.loc[day]
        side = int(np.sign(f["Close"] - f["Open"]))
        rest = g[[FIRST <= t <= LAST for t in g.index.time]]
        if side == 0 or len(rest) < 20 or rest.index[-1].time() != LAST:
            continue
        out.append(dict(day=day, sym=sym, rv=float(rv[day]), side=side, orh=float(f["High"]),
                        orl=float(f["Low"]), atr=float(atr[day]), o=rest["Open"].to_numpy(),
                        h=rest["High"].to_numpy(), l=rest["Low"].to_numpy(), c=rest["Close"].to_numpy()))
    return out


def simulate(c, mult, cap, risk):
    side, lvl = c["side"], c["orh"] if c["side"] == 1 else c["orl"]
    o, h, l = c["o"], c["h"], c["l"]
    hit = np.flatnonzero(h >= lvl) if side == 1 else np.flatnonzero(l <= lvl)
    if len(hit) == 0:
        return None                                               # never broke out
    i0 = hit[0]
    entry = max(lvl, o[i0]) if side == 1 else min(lvl, o[i0])
    dist = mult * c["atr"]
    stop = entry - side * dist
    qty = max(1, int(min(cap // entry, risk // dist)))
    exit_ = c["c"][-1]                                            # square-off at 15:00
    for i in range(i0, len(o)):
        if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
            exit_ = stop if i == i0 else (min(stop, o[i]) if side == 1 else max(stop, o[i]))
            break
    s = SLIP_PCT / 100
    e, x = entry * (1 + side * s), exit_ * (1 - side * s)
    es, xs = ("buy", "sell") if side == 1 else ("sell", "buy")
    return side * qty * (x - e) - leg_cost(qty * e, es) - leg_cost(qty * x, xs)


def run_variant(cands, n, mult):
    rows = []
    for day, g in cands.groupby("day"):
        pick = g[g["rv"] >= 1].nlargest(n, "rv")
        for c in pick.to_dict("records"):
            r = simulate(c, mult, CAPITAL / n, RISK / n)
            if r is not None:
                rows.append((day, c["sym"], r))
    return pd.DataFrame(rows, columns=["day", "sym", "net"])


def summary(tr):
    if tr.empty:
        return None
    m = tr.groupby(pd.to_datetime(tr["day"]).dt.to_period("M"))["net"].sum()
    w, ls = tr[tr["net"] > 0]["net"], tr[tr["net"] <= 0]["net"]
    return dict(n=len(tr), acc=len(w) / len(tr) * 100, net=tr["net"].sum(),
                avg_w=w.mean() if len(w) else 0.0, avg_l=ls.mean() if len(ls) else 0.0,
                best=tr["net"].max(), good=int((m > 0).sum()), months=len(m))


def line(name, sm):
    return (f"{name}: {sm['n']} trades, win {sm['acc']:.0f}%, avg win Rs {sm['avg_w']:+,.0f} / "
            f"avg loss Rs {sm['avg_l']:+,.0f}, best Rs {sm['best']:+,.0f}, net Rs {sm['net']:+,.0f}, "
            f"winning months {sm['good']}/{sm['months']}")


def study():
    now = datetime.now(IST)
    start = (now - timedelta(days=DAYS)).date()
    g = E.login()
    allc, failed = [], []
    for sym in SYMBOLS:
        try:
            allc += candidates(sym, fetch_long(g, sym, start, now))
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
            failed.append(sym)
    cands = pd.DataFrame(allc)
    days = sorted(cands["day"].unique())
    lock_from = days[-LOCK_DAYS]
    pre, lock = cands[cands["day"] < lock_from], cands[cands["day"] >= lock_from]
    lines = [f"🔥 Stocks in Play study (5-min opening-range breakout on unusual-volume shares)",
             f"{len(SYMBOLS) - len(failed)} shares, {len(days)} trading days, Rs {CAPITAL:,.0f} capital, "
             f"max loss Rs {RISK:,.0f} a day if every stop hits, after charges.",
             "Each morning: rank shares by 9:15-9:20 volume vs their 14-day average, take the top N (min 1x). "
             "First candle up -> BUY above its high; down -> SELL below its low. Stop = part of daily ATR. "
             "No target: winners run to 15:00.", "",
             f"Before the newest {LOCK_DAYS} days ({days[0]:%b %Y} - {pre['day'].max():%d %b %Y}):"]
    scored = []
    for n in TOP_N:
        for mult in STOP_ATR:
            sm = summary(run_variant(pre, n, mult))
            if sm:
                scored.append(((sm["good"] / sm["months"], sm["net"]), (n, mult), sm))
    scored.sort(key=lambda x: x[0], reverse=True)
    for k, (_, (n, mult), sm) in enumerate(scored):
        lines.append(("👉 " if k == 0 else "   ") + line(f"top {n}, stop {mult:g} ATR", sm))
    n, mult = scored[0][1]
    lk = summary(run_variant(lock, n, mult))
    lines += ["", f"🔒 Locked {LOCK_DAYS} days ({lock_from:%d %b} - {days[-1]:%d %b}), never used for any choice, with 👉:",
              "   " + (line(f"top {n}, stop {mult:g} ATR", lk) if lk else "no trades")]
    top = pre[pre["rv"] >= 1].sort_values("rv", ascending=False).groupby("day").head(n)
    lines += ["", "Most often in play: " + ", ".join(top["sym"].value_counts().head(8).index)]
    if failed:
        lines.append("No data: " + ", ".join(failed))
    lines += ["", "Worth paper-trading only if 👉 wins in most months AND the 🔒 locked line is positive. "
                  "Expect a LOW win rate: the money comes from a few big winners."]
    chunk = ""
    for ln in lines:                                   # Telegram limit is 4,096 characters
        if len(chunk) + len(ln) > 3500:
            send(chunk)
            chunk = ""
        chunk += ln + "\n"
    if chunk.strip():
        send(chunk)


if __name__ == "__main__":
    study()
