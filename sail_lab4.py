"""
Lab 4: the RSI(2) dip idea across many stocks, plus a next-day intraday version.

Why: on SAIL alone the "RSI(2) deep dip" idea made money in all three time
slices but traded only ~4 times a year. The same rule on ~40 liquid stocks
gives many more trades AND shows whether the edge is real or just SAIL luck.

Data: daily candles from Groww (about 5 years), long trades only.
  SWING versions : buy the next open after the dip, hold a few days.
  INTRADAY versions: buy the next open after the dip, sell the SAME day
                    (target / stop / close). Uses only daily highs and lows,
                    and assumes the stop was hit first if both were touched.

An idea only counts if it is profitable in all three equal time slices AND in
most of the stocks. Charges and slippage are included.

Past results do not predict future results. Not financial advice.
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import backtest as bt
import sail_lab3 as l3
import sail_intraday_bot as bot

DAYS = int(os.getenv("LAB_DAYS", "1850") or "1850")
WARMUP = 210
MIN_SHARE_PROFITABLE = 0.6

POOL = ["SAIL", "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN",
        "BHARTIARTL", "ITC", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR",
        "MARUTI", "SUNPHARMA", "TATASTEEL", "NTPC", "ONGC", "POWERGRID",
        "COALINDIA", "ADANIPORTS", "BAJFINANCE", "ASIANPAINT", "HCLTECH",
        "WIPRO", "TECHM", "JSWSTEEL", "HINDALCO", "VEDL", "BANKBARODA",
        "PNB", "CANBK", "IOC", "BPCL", "GAIL", "NMDC", "TATAPOWER", "DLF",
        "NESTLEIND", "TITAN"]

SWING = [
    dict(name="Swing: RSI(2)<5 dip, stop 6%, hold up to 7d", kind="rsi2", rsi=5, sl=6.0, hold=7),
    dict(name="Swing: RSI(2)<10 dip, stop 6%, hold up to 7d", kind="rsi2", rsi=10, sl=6.0, hold=7),
    dict(name="Swing: RSI(2)<15 dip, stop 6%, hold up to 7d", kind="rsi2", rsi=15, sl=6.0, hold=7),
    dict(name="Swing: RSI(2)<5 dip, stop 8%, hold up to 10d", kind="rsi2", rsi=5, sl=8.0, hold=10),
]
INTRADAY = [
    dict(name="Next-day intraday: RSI(2)<5, target 1% / stop 1%", rsi=5, tgt=1.0, sl=1.0),
    dict(name="Next-day intraday: RSI(2)<5, target 1.5% / stop 1.5%", rsi=5, tgt=1.5, sl=1.5),
    dict(name="Next-day intraday: RSI(2)<5, sell at close, stop 2%", rsi=5, tgt=99.0, sl=2.0),
    dict(name="Next-day intraday: RSI(2)<10, target 1% / stop 1%", rsi=10, tgt=1.0, sl=1.0),
    dict(name="Next-day intraday: RSI(2)<10, target 1.5% / stop 1.5%", rsi=10, tgt=1.5, sl=1.5),
    dict(name="Next-day intraday: RSI(2)<10, sell at close, stop 2%", rsi=10, tgt=99.0, sl=2.0),
    dict(name="Next-day intraday: RSI(2)<15, target 1% / stop 1%", rsi=15, tgt=1.0, sl=1.0),
]


# ------------------------------- data ---------------------------------
def fetch_daily(g, symbol: str, days: int) -> pd.DataFrame:
    end = datetime.now(bot.IST).date()
    cur = end - timedelta(days=days)
    frames = []
    while cur <= end:
        nxt = min(cur + timedelta(days=170), end)   # API limit: 180 days of daily candles
        for attempt in range(3):
            try:
                resp = g.get_historical_candles(
                    exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH,
                    groww_symbol=f"NSE-{symbol}",
                    start_time=f"{cur} 00:00:00", end_time=f"{nxt} 23:59:59",
                    candle_interval=g.CANDLE_INTERVAL_DAY)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        frames.append(bot.candles_to_df(resp.get("candles")))
        cur = nxt + timedelta(days=1)
        time.sleep(0.4)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def make_frame(candles: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame({"O": candles["Open"].to_numpy(float), "H": candles["High"].to_numpy(float),
                      "L": candles["Low"].to_numpy(float), "C": candles["Close"].to_numpy(float)},
                     index=[t.date() for t in candles.index])
    d = d[~d.index.duplicated(keep="last")]
    d["sma20"] = d["C"].rolling(20).mean()
    d["sma50"] = d["C"].rolling(50).mean()
    d["sma200"] = d["C"].rolling(200).mean()
    d["rsi2"] = l3.rsi_series(d["C"], 2)
    d["rsi14"] = l3.rsi_series(d["C"], 14)
    d["hh20"] = d["H"].rolling(20).max().shift(1)
    d["ll10"] = d["L"].rolling(10).min().shift(1)
    return d


# ----------------------------- strategies ------------------------------
def swing_run(d, S):
    trades = l3.swing_trades(d, S)
    dates = list(d.index)
    pos = {x: i for i, x in enumerate(dates)}
    for t in trades:
        t["exit_day"] = dates[min(pos[t["day"]] + t["hold"], len(dates) - 1)]
    return trades


def intraday_run(d, S):
    arr = {k: d[k].to_numpy(float) for k in d.columns}
    dates = list(d.index)
    O, H, L, C = arr["O"], arr["H"], arr["L"], arr["C"]
    n, trades = len(dates), []
    for t in range(WARMUP, n - 1):
        if not (C[t] > arr["sma200"][t] and arr["rsi2"][t] < S["rsi"]):
            continue
        e = t + 1
        entry = O[e]
        stop = entry * (1 - S["sl"] / 100)
        target = entry * (1 + S["tgt"] / 100)
        if L[e] <= stop:                      # conservative: stop first
            exit_price = stop
        elif H[e] >= target:
            exit_price = target
        else:
            exit_price = C[e]
        trades.append(dict(day=dates[e], exit_day=dates[e], hold=0,
                           hit=bool(exit_price > entry),
                           net=l3.net_single("BUY", entry, exit_price, "intraday")))
    return trades


# ------------------------------ reporting ------------------------------
def max_drawdown(trades):
    cum = peak = dd = 0.0
    for t in sorted(trades, key=lambda x: x["exit_day"]):
        cum += t["net"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def max_open(trades):
    ev = []
    for t in trades:
        ev.append((t["day"], 1))
        ev.append((t["exit_day"], -1))
    cur = best = 0
    for _, v in sorted(ev, key=lambda x: (x[0], x[1] > 0)):
        cur += v
        best = max(best, cur)
    return best


def summarize(name, per_stock, a, b, months):
    allt = [t for tr in per_stock.values() for t in tr]
    if not allt:
        return dict(name=name, n=0, ok=False)
    s = [[t for t in allt if t["day"] < a],
         [t for t in allt if a <= t["day"] < b],
         [t for t in allt if t["day"] >= b]]
    sl = [(len(p), sum(t["net"] for t in p)) for p in s]
    traded = {k: v for k, v in per_stock.items() if v}
    profitable = sum(1 for v in traded.values() if sum(t["net"] for t in v) > 0)
    wins = sum(1 for t in allt if t["net"] > 0)
    hits = sum(1 for t in allt if t["hit"])
    row = dict(name=name, n=len(allt), per_month=len(allt) / months,
               hit=hits / len(allt) * 100, win=wins / len(allt) * 100,
               sl=sl, stocks=len(traded), prof=profitable,
               dd=max_drawdown(allt), mx=max_open(allt),
               total=sum(t["net"] for t in allt), per_stock=per_stock)
    row["ok"] = (all(p > 0 for _, p in sl) and all(c >= 8 for c, _ in sl)
                 and profitable >= MIN_SHARE_PROFITABLE * len(traded))
    return row


def fmt(r):
    s = r["sl"]
    return (f"{r['name']}\n  {r['n']} trades ({r['per_month']:.1f}/month), "
            f"hit {r['hit']:.0f}%, profit-win {r['win']:.0f}%\n"
            f"  old {s[0][1]:+,.0f}, mid {s[1][1]:+,.0f}, new {s[2][1]:+,.0f}; "
            f"profitable in {r['prof']}/{r['stocks']} stocks\n"
            f"  worst drawdown {r['dd']:,.0f}, up to {r['mx']} trades open at once")


def main() -> None:
    g = bt.login()
    frames, failed = {}, []
    for sym in POOL:
        try:
            c = fetch_daily(g, sym, DAYS)
            if len(c) < WARMUP + 100:
                failed.append(sym)
                continue
            frames[sym] = make_frame(c)
            print(sym, "daily bars:", len(c))
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
            failed.append(sym)
    if not frames:
        bot.send("🧪 Lab 4: no daily data received from Groww.")
        return
    all_dates = sorted({x for f in frames.values() for x in f.index[WARMUP:]})
    a, b = all_dates[len(all_dates) // 3], all_dates[2 * len(all_dates) // 3]
    months = max((all_dates[-1] - all_dates[0]).days / 30.4, 1)
    rows = []
    for S in SWING:
        rows.append(summarize(S["name"], {k: swing_run(f, S) for k, f in frames.items()}, a, b, months))
    for S in INTRADAY:
        rows.append(summarize(S["name"], {k: intraday_run(f, S) for k, f in frames.items()}, a, b, months))
    rows = [r for r in rows if r["n"] > 0]
    rows.sort(key=lambda r: -r["total"])
    head = (f"🧪 Lab 4: RSI(2) dip on {len(frames)} stocks, about {months / 12:.1f} years, "
            f"Rs {bt.CAPITAL:,.0f} per trade, after charges + slippage.")
    bot.send(head + "\n\n" + "\n\n".join(fmt(r) for r in rows[:6]))
    if len(rows) > 6:
        bot.send("\n\n".join(fmt(r) for r in rows[6:]))
    good = [r for r in rows if r["ok"]]
    if failed:
        bot.send("No data for: " + ", ".join(failed))
    if good:
        best = good[0]
        ps = sorted(((k, len(v), sum(t["net"] for t in v)) for k, v in best["per_stock"].items() if v),
                    key=lambda x: -x[2])
        txt = [f"✅ Profitable in all 3 slices and most stocks: {len(good)} idea(s).",
               f"Best: {best['name']}", "Top stocks: " + ", ".join(f"{k} {n}tr {p:+,.0f}" for k, n, p in ps[:5]),
               "Weakest: " + ", ".join(f"{k} {n}tr {p:+,.0f}" for k, n, p in ps[-5:]),
               "", "Candidates to watch, not proof."]
        bot.send("\n".join(txt))
    else:
        bot.send("❌ No idea was profitable in all 3 slices and most stocks after charges.")


if __name__ == "__main__":
    main()
