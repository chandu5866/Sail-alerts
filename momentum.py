"""Intraday momentum study: does the first half hour predict the last half hour?

Published research on index markets: the return from the previous close to the first
half hour (and the second-last half hour) tends to point the same way as the last half
hour. This tests that on NSE with Groww 5-min data, after charges, month by month,
plus a locked test on the newest days that is never used for any choice.

Run:  python momentum.py      (sends the report to Telegram)
"""
import os
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

import engine as E
from backtest import SLIP_PCT, leg_cost
from ml import fetch_long
from sail_intraday_bot import IST, send

SYMBOLS = ["NIFTYBEES", "BANKBEES", "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "TCS"]
DAYS = 1100                      # ~3 years of history if Groww has it
LOCK_DAYS = 40                   # newest days, never used for any choice
CAPITAL = float(os.getenv("BACKTEST_CAPITAL", "200000"))
ENTRIES = {"14:30": dtime(14, 25), "15:00": dtime(14, 55)}   # candle whose close is the entry
EXITS = {"15:15": dtime(15, 10), "15:25": dtime(15, 20)}      # candle whose close is the exit
FILTERS = ["all", "agree", "big"]


def day_table(df):
    """One row per day: first-half-hour move, 14:30-15:00 move, entry and exit prices."""
    df = df.copy()
    df["day"] = df.index.date
    rows, prev_close = [], None
    for d, g in df.groupby("day"):
        t = {ts.time(): c for ts, c in zip(g.index, g["Close"])}
        if prev_close and dtime(9, 40) in t and all(x in t for x in
                                                    list(ENTRIES.values()) + list(EXITS.values())):
            rows.append(dict(day=d, r1=t[dtime(9, 40)] / prev_close - 1,
                             rmid=t[dtime(14, 55)] / t[dtime(14, 25)] - 1,
                             **{"in" + k.replace(":", ""): t[v] for k, v in ENTRIES.items()},
                             **{"out" + k.replace(":", ""): t[v] for k, v in EXITS.items()}))
        if dtime(15, 25) in t:
            prev_close = t[dtime(15, 25)]
    out = pd.DataFrame(rows)
    if not out.empty:     # "big" = first-half-hour move larger than the median of the past 60 days
        out["big"] = out["r1"].abs() > out["r1"].abs().shift(1).rolling(60, min_periods=20).median()
    return out


def trade_net(side, entry, exit_):
    q = max(1, int(CAPITAL // entry))
    s = SLIP_PCT / 100
    e = entry * (1 + side * s)
    x = exit_ * (1 - side * s)
    es, xs = ("buy", "sell") if side == 1 else ("sell", "buy")
    return side * q * (x - e) - leg_cost(q * e, es) - leg_cost(q * x, xs)


def variant_trades(tab, ent, ext, flt, direction):
    """direction +1 = follow the first half hour (momentum), -1 = fade it."""
    if flt == "agree" and ent != "15:00":
        return None
    res = []
    for r in tab.itertuples():
        if r.r1 == 0 or (flt == "big" and not r.big):
            continue
        if flt == "agree" and np.sign(r.rmid) != np.sign(r.r1):
            continue
        side = int(np.sign(r.r1)) * direction
        res.append((r.day, trade_net(side, getattr(r, "in" + ent.replace(":", "")), getattr(r, "out" + ext.replace(":", "")))))
    return pd.DataFrame(res, columns=["day", "net"])


def summary(tr):
    if tr is None or tr.empty:
        return None
    m = tr.groupby(pd.to_datetime(tr["day"]).dt.to_period("M"))["net"].sum()
    return dict(n=len(tr), acc=(tr["net"] > 0).mean() * 100, net=tr["net"].sum(),
                per=tr["net"].mean(), good=int((m > 0).sum()), months=len(m))


def study():
    now = datetime.now(IST)
    start = (now - timedelta(days=DAYS)).date()
    g = E.login()
    lines = [f"⏱ Intraday momentum study: {len(SYMBOLS)} symbols, ~{DAYS // 365} years of 5-min data, "
             f"Rs {CAPITAL:,.0f} per trade, after charges. One trade a day per symbol.",
             "Rule: trade in the direction of the move from yesterday's close to 9:45 "
             "('fade' = the opposite way), enter 14:30 or 15:00, exit 15:15 or 15:25.",
             "Filters: all days | 'agree' = 14:30-15:00 moved the same way | 'big' = 9:45 move above its 60-day median.",
             f"Chosen on everything before the newest {LOCK_DAYS} days, then checked on those 🔒 locked days.", ""]
    for sym in SYMBOLS:
        try:
            tab = day_table(fetch_long(g, sym, start, now))
        except Exception as e:
            lines.append(f"{sym}: data failed ({type(e).__name__})")
            continue
        if len(tab) < LOCK_DAYS + 120:
            lines.append(f"{sym}: not enough data ({len(tab)} days)")
            continue
        lock_from = tab["day"].iloc[-LOCK_DAYS]
        pre, lock = tab[tab["day"] < lock_from], tab[tab["day"] >= lock_from]
        scored = []
        for ent in ENTRIES:
            for ext in EXITS:
                for flt in FILTERS:
                    for direction, dname in ((1, "follow"), (-1, "fade")):
                        sm = summary(variant_trades(pre, ent, ext, flt, direction))
                        if sm and sm["n"] >= 40:
                            scored.append(((sm["good"] / sm["months"], sm["net"]),
                                           (ent, ext, flt, direction, dname), sm))
        if not scored:
            lines.append(f"{sym}: no variant with enough trades")
            continue
        scored.sort(key=lambda x: x[0], reverse=True)
        base = summary(variant_trades(pre, "15:00", "15:25", "all", 1))
        lines.append(f"📈 {sym} ({len(pre)} days + {len(lock)} locked)")
        if base:
            lines.append(f"   Plain rule (follow, 15:00-15:25, all days): {base['n']} trades, "
                         f"{base['acc']:.0f}%, Rs {base['net']:+,.0f}, winning months {base['good']}/{base['months']}")
        for k, (_, v, sm) in enumerate(scored[:2]):
            lines.append(f"   {'👉' if k == 0 else '  '} {v[4]} {v[0]}-{v[1]} {v[2]}: {sm['n']} trades, {sm['acc']:.0f}%, "
                         f"Rs {sm['net']:+,.0f} (Rs {sm['per']:+,.0f}/trade), winning months {sm['good']}/{sm['months']}")
        v = scored[0][1]
        lk = summary(variant_trades(lock, v[0], v[1], v[2], v[3]))
        lines.append(f"   🔒 Locked {LOCK_DAYS} days with 👉: " + (
            f"{lk['n']} trades, {lk['acc']:.0f}%, Rs {lk['net']:+,.0f}" if lk else "no trades"))
    lines += ["", "A rule is worth paper-trading only if it wins in most months AND the 🔒 locked days are positive."]
    msg, chunk = "\n".join(lines), ""
    for line in msg.split("\n"):                      # Telegram limit is 4,096 characters
        if len(chunk) + len(line) > 3500:
            send(chunk)
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        send(chunk)


if __name__ == "__main__":
    study()
