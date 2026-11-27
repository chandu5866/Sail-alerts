"""Monthly momentum study (positional, 1-6 month holds) for Indian stocks, after costs.

Method (like NSE's Nifty200 Momentum 30 index):
  score = average of z-scores of (6-month return / volatility) and (12-month return / volatility)
  hold the top N, equal weight, rebalance monthly or quarterly at the next day's open.
Optional trend filter: hold cash while NIFTYBEES is below its 200-day average.
Benchmark: buying and holding NIFTYBEES. Variants are chosen on everything before the
newest LOCK_MONTHS months, then checked on those 🔒 locked months.

Run:  python momo.py      (sends the report to Telegram)
"""
import functools
import io
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

import engine as E
from ml import POOL
from sail_intraday_bot import IST, candles_to_df, send

print = functools.partial(print, flush=True)     # show progress in the log as it happens

INDEX = "NIFTYBEES"
START = datetime(2010, 1, 1, tzinfo=IST)       # Yahoo has adjusted NSE prices this far back
CHUNK = [1000]                                  # days per request; set by a quick probe
LOCK_MONTHS = 12
TOP_N = [10, 20]
REBAL = {"monthly": 1, "quarterly": 3}
COST_SIDE = 0.002            # per side: STT 0.1% + charges + slippage on delivery trades
FALLBACK = POOL + ["M&M", "ULTRACEMCO", "TRENT", "HAL", "RECLTD", "PFC", "IRCTC", "INDUSINDBK",
                   "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "CIPLA", "DRREDDY", "DIVISLAB", "GRASIM",
                   "SHRIRAMFIN", "CHOLAFIN", "LUPIN", "AUROPHARMA", "TCS", "NESTLEIND", "BRITANNIA",
                   "PIDILITIND", "HAVELLS", "SIEMENS", "ABB", "DMART", "POLYCAB", "PERSISTENT", "COFORGE"]


def universe():
    """Today's Nifty 200 list from niftyindices.com (falls back to a built-in list)."""
    try:
        import requests
        r = requests.get("https://niftyindices.com/IndexConstituent/ind_nifty200list.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].dropna().astype(str).str.strip().tolist()
        if len(syms) > 150:
            return syms, "Nifty 200 (today's list)"
    except Exception as e:
        print("nifty200 list failed:", type(e).__name__, e)
    return list(dict.fromkeys(FALLBACK)), f"{len(set(FALLBACK))} large F&O shares (Nifty 200 list not reachable)"


def _get(g, symbol, a, b, iv):
    resp = g.get_historical_candles(
        exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_CASH, groww_symbol=f"NSE-{symbol}",
        start_time=a.strftime("%Y-%m-%d %H:%M:%S"), end_time=b.strftime("%Y-%m-%d %H:%M:%S"),
        candle_interval=iv, timeout=15)
    E.time.sleep(0.3)
    return candles_to_df((resp or {}).get("candles"))


def fetch_daily(g, symbol, start, end):
    """Daily candles in ~3-year requests (falls back to 6-month requests if Groww refuses)."""
    frames, cur = [], start
    iv = getattr(g, "CANDLE_INTERVAL_DAY", "1day")
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK[0]), end)
        try:
            frames.append(_get(g, symbol, cur, nxt, iv))
        except Exception:
            if CHUNK[0] == 180:
                if (end - cur).days < 400:
                    raise
                print(symbol, "no data from", cur.date())
                cur = nxt
                continue
            a = cur
            while a < nxt:
                b = min(a + timedelta(days=180), nxt)
                try:
                    frames.append(_get(g, symbol, a, b, iv))
                except Exception as e:
                    if (end - a).days < 400:
                        raise
                    print(symbol, "no data from", a.date(), type(e).__name__)
                a = b
        cur = nxt
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index = pd.to_datetime(df.index.date)
    df = df[~df.index.duplicated(keep="last")]
    return df[["Open", "Close"]]


def load_yahoo(syms):
    """Split/bonus-adjusted daily prices for all shares in one download (Yahoo Finance)."""
    import yfinance as yf
    tick = [s + ".NS" for s in syms]
    data = yf.download(tick, start=START.strftime("%Y-%m-%d"), auto_adjust=True,
                       progress=False, threads=True, group_by="column")
    C, O = data["Close"].copy(), data["Open"].copy()
    C.columns = [c[:-3] for c in C.columns]
    O.columns = [c[:-3] for c in O.columns]
    C.index = pd.to_datetime(C.index.date)
    O.index = C.index
    keep = [c for c in C.columns if C[c].notna().sum() > 300]
    return C[keep], O[keep], [s for s in syms if s not in keep]


def scores(C, i):
    """Normalised momentum score for every stock with a full year of history before day i."""
    if i < 253:
        return pd.Series(dtype=float)
    win = C.iloc[i - 253:i]
    ok = win.notna().all()
    win = win.loc[:, ok]
    lr = np.log(win / win.shift(1)).iloc[1:]
    vol = lr.std() * np.sqrt(252)
    last = win.iloc[-1]
    m6 = (last / win.iloc[-127] - 1) / vol
    m12 = (last / win.iloc[0] - 1) / vol
    z = lambda s: (s - s.mean()) / s.std()
    return ((z(m6) + z(m12)) / 2).dropna()


def backtest(C, O, idx_close, n, step, trend, start_i, end_i):
    """Equity curve (start 1.0) from day start_i to end_i, rebalancing at month starts."""
    dates = C.index
    months = pd.Series(dates.to_period("M"), index=dates)
    first = [k for k in range(start_i, end_i) if k == 0 or months.iloc[k] != months.iloc[k - 1]]
    rebal = [k for k in first if (dates[k].month - 1) % step == 0]
    sma = idx_close.rolling(200).mean()
    eq = pd.Series(np.nan, index=dates[start_i:end_i])
    value, held, wts, k_prev = 1.0, [], None, start_i
    picks_log = []
    for r, k in enumerate(rebal + [end_i]):
        # mark the current holdings from k_prev up to day k (exit at day k's open)
        if held and k > k_prev:
            px = pd.concat([O.iloc[[k_prev]][held], C.iloc[k_prev:k][held]])
            if k < len(dates):
                px = pd.concat([px, O.iloc[[k]][held]])
            px = px.ffill()
            path = (px / px.iloc[0]).mul(wts, axis=1).sum(axis=1)
            seg = path.iloc[1:1 + (k - k_prev)] * value
            eq.iloc[k_prev - start_i:k - start_i] = seg.values[:k - k_prev]
            end_val = path.iloc[-1] * value
            drift = (px.iloc[-1] / px.iloc[0]) * wts
            old_w = drift / drift.sum()
        else:
            if k > k_prev:
                eq.iloc[k_prev - start_i:k - start_i] = value
            end_val, old_w = value, pd.Series(dtype=float)
        value = end_val
        if k >= end_i:
            break
        risk_on = (not trend) or (idx_close.iloc[k - 1] > sma.iloc[k - 1])
        sc = scores(C, k) if risk_on else pd.Series(dtype=float)
        new = list(sc.nlargest(n).index) if len(sc) >= n else []
        new_w = pd.Series(1.0 / len(new), index=new) if new else pd.Series(dtype=float)
        turnover = new_w.sub(old_w, fill_value=0).abs().sum()
        value *= 1 - COST_SIDE * turnover
        held, wts, k_prev = new, new_w, k
        picks_log.append((dates[k], new))
    return eq.ffill().fillna(1.0), picks_log


def stats(eq, bench):
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    y = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
    yb = bench.groupby(bench.index.year).last() / bench.groupby(bench.index.year).first() - 1
    beat = int((y > yb.reindex(y.index)).sum())
    return dict(cagr=cagr * 100, dd=dd * 100, total=(eq.iloc[-1] - 1) * 100, beat=beat, years=len(y), yearly=y, yb=yb)


def study():
    now = datetime.now(IST)
    syms, uname = universe()
    try:
        C, O, failed = load_yahoo([INDEX] + syms)
        src = "Yahoo Finance (split/bonus adjusted)"
        print("yahoo:", C.shape, C.index[0], C.index[-1])
    except Exception as e:
        print("yahoo failed:", type(e).__name__, e)
        C = None
    if C is None or INDEX not in C or len(C) < 800:
        send("⚠️ Momentum study: Yahoo Finance data not available from GitHub right now "
             "(Groww's daily history is too short and not split-adjusted). Try again later.")
        return
    t0 = datetime.now()
    C = C[C[INDEX].notna()].sort_index()
    O = O.reindex(C.index)
    idx = C[INDEX]
    stocks = [c for c in C.columns if c != INDEX]
    Cs, Os = C[stocks], O[stocks].fillna(C[stocks])
    lock_from = C.index.searchsorted(C.index[-1] - pd.DateOffset(months=LOCK_MONTHS))
    start_i = 260
    bench_pre = idx.iloc[start_i:lock_from] / idx.iloc[start_i]
    lines = [f"📈 Monthly momentum study: {uname}, {len(stocks)} shares with data ({src}), "
             f"{C.index[0]:%b %Y} - {C.index[-1]:%b %Y}, after costs ({COST_SIDE * 100:.1f}% per side).",
             "Score = 6- and 12-month return adjusted for volatility (NSE's method). Hold the top N, equal weight.",
             "Trend filter = hold cash while NIFTYBEES is below its 200-day average.",
             f"⚠️ Uses today's index list, so past results look a bit better than reality (survivorship).", "",
             f"Before the newest {LOCK_MONTHS} months ({C.index[start_i]:%b %Y} - {C.index[lock_from - 1]:%b %Y}):"]
    bs = stats(bench_pre, bench_pre)
    lines.append(f"   NIFTYBEES buy & hold: {bs['cagr']:.1f}%/yr, worst fall {bs['dd']:.0f}%")
    scored = []
    for n in TOP_N:
        for rname, step in REBAL.items():
            for trend in (False, True):
                eq, _ = backtest(Cs, Os, idx, n, step, trend, start_i, lock_from)
                st = stats(eq, bench_pre)
                name = f"top {n}, {rname}, {'trend filter' if trend else 'always invested'}"
                scored.append(((st["beat"] / st["years"], st["cagr"]), name, (n, step, trend), st))
    scored.sort(key=lambda x: x[0], reverse=True)
    for k, (_, name, _, st) in enumerate(scored):
        lines.append(f"{'👉' if k == 0 else '  '} {name}: {st['cagr']:.1f}%/yr, worst fall {st['dd']:.0f}%, "
                     f"beat NIFTYBEES in {st['beat']}/{st['years']} years")
    name, (n, step, trend), st = scored[0][1], scored[0][2], scored[0][3]
    lines.append("   Year by year 👉 vs NIFTYBEES: " + " | ".join(
        f"{y}: {v * 100:+.0f}% vs {st['yb'].get(y, np.nan) * 100:+.0f}%" for y, v in st["yearly"].items()))
    eq_l, picks = backtest(Cs, Os, idx, n, step, trend, lock_from, len(C))
    bl = idx.iloc[lock_from:] / idx.iloc[lock_from]
    lines += ["", f"🔒 Locked {LOCK_MONTHS} months ({C.index[lock_from]:%b %Y} - {C.index[-1]:%b %Y}), never used for any choice:",
              f"   👉 {name}: {(eq_l.iloc[-1] - 1) * 100:+.1f}% vs NIFTYBEES {(bl.iloc[-1] - 1) * 100:+.1f}%, "
              f"worst fall {(eq_l / eq_l.cummax() - 1).min() * 100:.0f}%"]
    risk_on = (not trend) or idx.iloc[-1] > idx.rolling(200).mean().iloc[-1]
    now_list = list(scores(Cs, len(Cs)).nlargest(n).index) if risk_on else []
    lines += ["", f"📋 If 👉 were run today: " + (", ".join(now_list) if now_list else "hold cash (NIFTY below its 200-day average)"),
              "", "Worth paper-trading only if 👉 beats NIFTYBEES in most years AND in the 🔒 locked months. "
                  "Otherwise a Nifty200 Momentum 30 ETF or a plain NIFTYBEES is the simpler choice."]
    if failed:
        lines.append(f"No data ({len(failed)}): " + ", ".join(failed[:25]) + (" ..." if len(failed) > 25 else ""))
    chunk = ""
    for ln in lines:
        if len(chunk) + len(ln) > 3500:
            send(chunk)
            chunk = ""
        chunk += ln + "\n"
    if chunk.strip():
        send(chunk)


if __name__ == "__main__":
    study()
