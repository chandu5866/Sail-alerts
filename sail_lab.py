"""
SAIL strategy lab.

Tests many rule variations on SAIL's past 5-minute candles (about 3 years
from Groww) and reports, for each one: number of calls, win rate, and profit
after real Groww charges and slippage.

A variation only counts if it is profitable BOTH in the older period and in
the newest LAB_TEST_DAYS days, which the ranking never looked at. The
"current rules" row is the live bot, so you can see what each change does.

Past results do not predict future results. Not financial advice.
"""

import os
from datetime import datetime, time as dtime, timedelta

import pandas as pd

import backtest as bt
import sail_intraday_bot as bot

DAYS = int(os.getenv("LAB_DAYS", "1000") or "1000")
TEST_DAYS = int(os.getenv("LAB_TEST_DAYS", "180") or "180")
SYMBOL = "SAIL"
MIN_TOTAL_CALLS = 30
MIN_NEW_CALLS = 8

BASE = dict(range_min=0.25, range_max=1.5, vol_mult=1.2, t=(1.0, 2.0),
            rmin=0.3, rmax=1.0, last=dtime(13, 25), strength=0.0, ext=99.0,
            sides="both", trend=False)


def V(name, **kw):
    d = dict(BASE)
    d.update(kw)
    d["name"] = name
    return d


MORNING = dtime(11, 30)
EARLY = dtime(10, 30)
VARIANTS = [
    V("current rules"),
    V("targets 1.5x/3x", t=(1.5, 3.0)),
    V("targets 2x/4x", t=(2.0, 4.0)),
    V("volume 1.5x", vol_mult=1.5),
    V("volume 2x", vol_mult=2.0),
    V("range 0.4-1.2%", range_min=0.4, range_max=1.2),
    V("morning only (to 11:30)", last=MORNING),
    V("early only (to 10:30)", last=EARLY),
    V("strong candle 70%", strength=0.7),
    V("no chasing (max 0.3% past range)", ext=0.3),
    V("buy only", sides="buy"),
    V("sell only", sides="sell"),
    V("with daily trend", trend=True),
    V("wider SL 0.6-1.2%", rmin=0.6, rmax=1.2),
    V("wider SL 0.8-1.5%", rmin=0.8, rmax=1.5),
    V("trend + volume 1.5x", trend=True, vol_mult=1.5),
    V("trend + morning", trend=True, last=MORNING),
    V("trend + strong candle + morning", trend=True, strength=0.7, last=MORNING),
    V("trend + volume 1.5x + targets 1.5x/3x", trend=True, vol_mult=1.5, t=(1.5, 3.0)),
    V("trend + wider SL 0.6-1.2%", trend=True, rmin=0.6, rmax=1.2),
    V("strict: trend+vol1.5+strong+range+morning", trend=True, vol_mult=1.5,
      strength=0.7, range_min=0.4, range_max=1.2, last=MORNING),
    V("strict + wider SL + targets 1.5x/3x", trend=True, vol_mult=1.5, strength=0.7,
      range_min=0.4, range_max=1.2, last=MORNING, rmin=0.6, rmax=1.2, t=(1.5, 3.0)),
]


def run_day(day, g, prev_close, P):
    """One day under rule set P. Returns a trade dict or None."""
    idx = g.index
    times = [t.time() for t in idx]
    n = len(g)
    H = g["High"].to_numpy(float)
    L = g["Low"].to_numpy(float)
    C = g["Close"].to_numpy(float)
    Vo = g["Volume"].to_numpy(float)

    orb = [i for i in range(n) if bot.OPEN_T <= times[i] < bot.RANGE_END]
    or_high, or_low = H[orb].max(), L[orb].min()
    range_pct = (or_high - or_low) / or_low * 100
    if not (P["range_min"] <= range_pct <= P["range_max"]):
        return None
    after = [i for i in range(n) if times[i] >= bot.RANGE_END]
    if not after:
        return None
    typ = (H + L + C) / 3
    cum_pv = (typ * Vo).cumsum()
    cum_v = Vo.cumsum()
    last_bar = max((i for i in range(n) if times[i] <= dtime(15, 0)), default=None)
    if last_bar is None:
        return None
    first_after = after[0]

    for i in after:
        bar_end = idx[i] + timedelta(minutes=bot.BAR_MIN)
        if bar_end.time() > P["last"]:
            break
        price = C[i]
        vwap = cum_pv[i] / cum_v[i] if cum_v[i] > 0 else typ[: i + 1].mean()
        prior = Vo[first_after:i]
        avg = prior.mean() if len(prior) else 0.0
        vol_ok = avg == 0 or Vo[i] >= P["vol_mult"] * avg

        side = None
        if price > or_high and price > vwap and vol_ok:
            side = "BUY"
        elif price < or_low and price < vwap and vol_ok:
            side = "SELL"
        if side is None:
            continue
        if P["sides"] == "buy" and side != "BUY":
            continue
        if P["sides"] == "sell" and side != "SELL":
            continue
        rng = H[i] - L[i]
        if P["strength"] > 0:
            if rng <= 0:
                continue
            pos = (C[i] - L[i]) / rng if side == "BUY" else (H[i] - C[i]) / rng
            if pos < P["strength"]:
                continue
        ext = ((price - or_high) / or_high * 100 if side == "BUY"
               else (or_low - price) / or_low * 100)
        if ext > P["ext"]:
            continue
        if P["trend"]:
            if prev_close is None:
                continue
            if (side == "BUY" and price <= prev_close) or \
               (side == "SELL" and price >= prev_close):
                continue

        # ---- trade found: build levels exactly like the live bot ----
        sign = 1 if side == "BUY" else -1
        raw = (price - or_low) if side == "BUY" else (or_high - price)
        risk = min(max(raw, price * P["rmin"] / 100), price * P["rmax"] / 100)
        entry = bot.tick(price)
        sl = bot.tick(entry - sign * risk)
        t1 = bot.tick(entry + sign * risk * P["t"][0])
        t2 = bot.tick(entry + sign * risk * P["t"][1])
        trade = dict(day=day, side=side, entry=entry, sl=sl, t1=t1, t2=t2,
                     t1_hit=False)
        cur_sl = sl
        for j in range(i + 1, last_bar + 1):
            hi, lo = H[j], L[j]
            sl_hit = lo <= cur_sl if side == "BUY" else hi >= cur_sl
            t1_hit = hi >= t1 if side == "BUY" else lo <= t1
            t2_hit = hi >= t2 if side == "BUY" else lo <= t2
            if sl_hit:
                trade["why"] = "T1 then cost" if trade["t1_hit"] else "SL"
                return trade
            if t2_hit:
                trade["why"] = "T2"
                return trade
            if t1_hit and not trade["t1_hit"]:
                trade["t1_hit"] = True
                cur_sl = entry
        trade["why"] = "square-off"
        trade["last"] = float(C[last_bar])
        return trade
    return None


def run_variant(df, P):
    days_list = sorted(set(df.index.date))
    closes = {}
    for d in days_list:
        closes[d] = float(df[df.index.date == d]["Close"].iloc[-1])
    trades, prev = [], None
    for k, d in enumerate(days_list):
        g = df[df.index.date == d]
        prev_close = closes[days_list[k - 1]] if k > 0 else None
        if len(g[(g.index.time >= bot.OPEN_T) & (g.index.time < bot.RANGE_END)]) < 3:
            continue
        t = run_day(d, g, prev_close, P)
        if t:
            t["net"] = bt.evaluate(t)
            trades.append(t)
    return trades


def summarize_variants(df):
    days_all = sorted(set(df.index.date))
    cutoff = days_all[-1] - timedelta(days=TEST_DAYS)
    rows = []
    for P in VARIANTS:
        tr = run_variant(df, P)
        old_t = [t for t in tr if t["day"] < cutoff]
        new_t = [t for t in tr if t["day"] >= cutoff]
        wins = sum(1 for t in tr if t["net"] > 0)
        rows.append(dict(name=P["name"], calls=len(tr), wins=wins,
                         win_pct=(wins / len(tr) * 100 if tr else 0.0),
                         old=sum(t["net"] for t in old_t),
                         new=sum(t["net"] for t in new_t),
                         new_calls=len(new_t)))
    return rows, len(days_all)


def report(rows, ndays):
    rows_sorted = sorted(rows, key=lambda r: r["old"], reverse=True)
    head = [f"🧪 SAIL lab: {ndays} trading days, Rs {bt.CAPITAL:,.0f} per trade,",
            f"after Groww charges + slippage. 'new' = newest {TEST_DAYS} days (unseen).",
            ""]
    lines = [f"{r['name']}: {r['calls']} calls, win {r['win_pct']:.0f}%, "
             f"old {r['old']:+,.0f}, new {r['new']:+,.0f}" for r in rows_sorted]
    out = ["\n".join(head + lines)]
    good = [r for r in rows if r["old"] > 0 and r["new"] > 0
            and r["calls"] >= MIN_TOTAL_CALLS and r["new_calls"] >= MIN_NEW_CALLS]
    good.sort(key=lambda r: r["old"] + r["new"], reverse=True)
    if good:
        txt = ["✅ Profitable in BOTH periods:"]
        for r in good[:3]:
            txt.append(f"{r['name']}: win {r['win_pct']:.0f}%, "
                       f"total {r['old'] + r['new']:+,.0f}")
        txt.append("")
        txt.append("Trying many variations can find lucky ones. Treat these as "
                   "candidates to watch, not proof.")
        out.append("\n".join(txt))
    else:
        out.append("❌ No variation was profitable in both periods after costs.")
    return out


def main() -> None:
    g = bt.login()
    df = bt.fetch_symbol(g, SYMBOL, DAYS)
    if df.empty:
        bot.send("🧪 SAIL lab: Groww returned no candles.")
        return
    rows, ndays = summarize_variants(df)
    for m in report(rows, ndays):
        bot.send(m)


if __name__ == "__main__":
    main()
