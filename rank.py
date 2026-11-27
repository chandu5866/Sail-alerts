"""Ranking study (Qlib-style): each morning rank ~60 shares by predicted move and trade only the top one.

At 9:45 every share gets features from its first 30 minutes and recent days. A ridge model,
trained walk-forward (only on older days), predicts each share's move from 9:45 to 15:00
relative to the other shares. Each day we trade the top-ranked share (BUY) and/or the
bottom-ranked share (SELL). Baseline without learning: rank by relative strength vs NIFTY.
Chosen on everything before the newest LOCK_DAYS days, then checked on those 🔒 locked days.

Run:  python rank.py      (sends the report to Telegram)
"""
import functools
import os
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

import engine as E
from backtest import SLIP_PCT, leg_cost
from ml import fetch_long
from sail_intraday_bot import IST, send
from sip import SYMBOLS

print = functools.partial(print, flush=True)

INDEX = "NIFTYBEES"
DAYS = 1100
LOCK_DAYS = 40
BLOCK = 20                       # walk-forward block = ~1 month
MIN_TRAIN = 120                  # days of history before the first prediction
CAPITAL = float(os.getenv("BACKTEST_CAPITAL", "200000"))
RIDGE = 10.0
T_DEC, T_EXIT = dtime(9, 40), dtime(14, 55)          # decide on the 9:40 candle's close, exit on 14:55's
FEATS = ["gap", "r30", "rs30", "rvol30", "ret1", "ret5", "pos30", "atrp", "vs_prev_hi", "vs_prev_lo"]
FEAT_NAME = {"gap": "opening gap", "r30": "first-30-min move", "rs30": "first-30-min move vs NIFTY",
             "rvol30": "first-30-min volume vs usual", "ret1": "yesterday's move", "ret5": "last 5 days' move",
             "pos30": "place in first-30-min range", "atrp": "how jumpy (ATR %)",
             "vs_prev_hi": "distance above yesterday's high", "vs_prev_lo": "distance below yesterday's low"}


def day_rows(sym, df):
    if df.empty:
        return []
    df = df.copy()
    df["day"] = df.index.date
    d = df.groupby("day").agg(O=("Open", "first"), H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    pc = d["C"].shift(1)
    tr = pd.concat([d["H"] - d["L"], (d["H"] - pc).abs(), (d["L"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)
    first = df[df.index.time <= T_DEC]
    v30 = first.groupby("day")["Volume"].sum()
    rv = v30 / v30.rolling(14).mean().shift(1)
    out = []
    for day, g in df.groupby("day"):
        if pd.isna(atr.get(day)) or pd.isna(rv.get(day)) or pd.isna(pc.get(day)):
            continue
        t = g.index.time
        f, rest = g[t <= T_DEC], g[(t > T_DEC) & (t <= T_EXIT)]
        if len(f) < 6 or len(rest) < 50 or rest.index[-1].time() != T_EXIT:
            continue
        o, c = f["Open"].iloc[0], f["Close"].iloc[-1]
        hi, lo = f["High"].max(), f["Low"].min()
        k = d.index.get_loc(day)
        a = atr[day]
        out.append(dict(day=day, sym=sym, gap=o / pc[day] - 1, r30=c / o - 1, rvol30=np.log(max(rv[day], 1e-3)),
                        ret1=d["C"].iloc[k - 1] / d["C"].iloc[k - 2] - 1 if k >= 2 else 0.0,
                        ret5=d["C"].iloc[k - 1] / d["C"].iloc[k - 6] - 1 if k >= 6 else 0.0,
                        pos30=(c - lo) / (hi - lo) if hi > lo else 0.5, atrp=a / c,
                        vs_prev_hi=(c - d["H"].iloc[k - 1]) / a, vs_prev_lo=(d["L"].iloc[k - 1] - c) / a,
                        entry=c, exit=rest["Close"].iloc[-1], atr=a,
                        hi=rest["High"].to_numpy(), lo=rest["Low"].to_numpy(), op=rest["Open"].to_numpy()))
    return out


def trade(r, side, cap, stop_k):
    entry, exit_ = r["entry"], r["exit"]
    if stop_k:
        stop = entry - side * stop_k * r["atr"]
        for i in range(len(r["hi"])):
            if (side == 1 and r["lo"][i] <= stop) or (side == -1 and r["hi"][i] >= stop):
                exit_ = min(stop, r["op"][i]) if side == 1 else max(stop, r["op"][i])
                break
    q = max(1, int(cap // entry))
    s = SLIP_PCT / 100
    e, x = entry * (1 + side * s), exit_ * (1 - side * s)
    es, xs = ("buy", "sell") if side == 1 else ("sell", "buy")
    return side * q * (x - e) - leg_cost(q * e, es) - leg_cost(q * x, xs)


def ridge_fit(X, y):
    A = np.c_[np.ones(len(X)), X]
    reg = np.eye(A.shape[1]) * RIDGE
    reg[0, 0] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def walk_forward(df, days):
    """Predicted score for every row; each block predicted by a model trained only on older days."""
    pred = pd.Series(np.nan, index=df.index)
    coefs = []
    for b0 in range(MIN_TRAIN, len(days), BLOCK):
        blk = days[b0:b0 + BLOCK]
        tr = df[df["day"] < blk[0]]
        te = df[df["day"].isin(blk)]
        mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
        X = ((tr[FEATS] - mu) / sd).clip(-5, 5).to_numpy()
        w = ridge_fit(X, tr["fwd_dm"].to_numpy())
        coefs.append(w[1:])
        pred[te.index] = np.c_[np.ones(len(te)), ((te[FEATS] - mu) / sd).clip(-5, 5).to_numpy()] @ w
    return pred, np.mean(coefs, axis=0) if coefs else np.zeros(len(FEATS))


def run(df, score, mode, stop_k):
    rows = []
    for day, g in df.groupby("day"):
        g = g[g[score].notna()]
        if len(g) < 10:
            continue
        picks = []
        if mode in ("long", "both"):
            picks.append((g.loc[g[score].idxmax()], 1))
        if mode in ("short", "both"):
            picks.append((g.loc[g[score].idxmin()], -1))
        for r, side in picks:
            rows.append((day, r["sym"], side, trade(r, side, CAPITAL / len(picks), stop_k)))
    return pd.DataFrame(rows, columns=["day", "sym", "side", "net"])


def summary(tr):
    if tr.empty:
        return None
    m = tr.groupby(pd.to_datetime(tr["day"]).dt.to_period("M"))["net"].sum()
    return dict(n=len(tr), acc=(tr["net"] > 0).mean() * 100, net=tr["net"].sum(), per=tr["net"].mean(),
                good=int((m > 0).sum()), months=len(m))


def fmt(name, sm):
    return (f"{name}: {sm['n']} trades, win {sm['acc']:.0f}%, Rs {sm['net']:+,.0f} "
            f"(Rs {sm['per']:+,.0f}/trade), winning months {sm['good']}/{sm['months']}")


def study():
    now = datetime.now(IST)
    start = (now - timedelta(days=DAYS)).date()
    g = E.login()
    rows, failed, t0 = [], [], datetime.now()
    syms = [INDEX] + [s for s in SYMBOLS if s != INDEX]
    for j, sym in enumerate(syms):
        if j % 10 == 0:
            print(f"{j}/{len(syms)} shares downloaded, {(datetime.now() - t0).seconds // 60} min")
        try:
            rows += day_rows(sym, fetch_long(g, sym, start, now))
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
            failed.append(sym)
    df = pd.DataFrame(rows)
    mkt = df[df["sym"] == INDEX].set_index("day")["r30"]
    df = df[df["sym"] != INDEX].copy()
    df["rs30"] = df["r30"] - df["day"].map(mkt).fillna(0)
    df["fwd"] = df["exit"] / df["entry"] - 1
    df["fwd_dm"] = df["fwd"] - df.groupby("day")["fwd"].transform("mean")    # move vs the other shares
    df = df.reset_index(drop=True)
    days = sorted(df["day"].unique())
    df["pred"], coef = walk_forward(df, days)
    df["rs_rank"] = df["rs30"]                                               # baseline: no learning
    lock_from = days[-LOCK_DAYS]
    first_pred = days[MIN_TRAIN]
    pre = df[(df["day"] >= first_pred) & (df["day"] < lock_from)]
    lock = df[df["day"] >= lock_from]
    lines = [f"🏆 Ranking study: {df['sym'].nunique()} shares, {len(days)} trading days, Rs {CAPITAL:,.0f} a day, after charges.",
             "At 9:45 a model (trained only on older days, retrained monthly) ranks every share by its expected "
             "move to 15:00 vs the others. BUY the top one and/or SELL the bottom one, exit at 15:00.",
             f"Baseline without learning: rank by first-30-min strength vs NIFTY.", "",
             f"Walk-forward, {first_pred:%b %Y} - {pre['day'].max():%d %b %Y}:"]
    scored = []
    for score, sname in (("pred", "model"), ("rs_rank", "baseline")):
        for mode in ("long", "short", "both"):
            for stop_k in (0, 0.5):
                sm = summary(run(pre, score, mode, stop_k))
                if sm:
                    name = f"{sname}, {mode}, {'stop 0.5 ATR' if stop_k else 'no stop'}"
                    scored.append(((sm["good"] / sm["months"], sm["net"]), name, (score, mode, stop_k), sm))
    scored.sort(key=lambda x: x[0], reverse=True)
    for k, (_, name, _, sm) in enumerate(scored):
        lines.append(("👉 " if k == 0 else "   ") + fmt(name, sm))
    name, (score, mode, stop_k) = scored[0][1], scored[0][2]
    lk = summary(run(lock, score, mode, stop_k))
    lines += ["", f"🔒 Locked {LOCK_DAYS} days ({lock_from:%d %b} - {days[-1]:%d %b}), never used for any choice:",
              "   " + (fmt(name, lk) if lk else "no trades")]
    top = sorted(zip(FEATS, coef), key=lambda x: -abs(x[1]))[:4]
    lines += ["", "What the model leans on most: " + "; ".join(
        f"{FEAT_NAME[f]} ({'+' if c > 0 else '-'})" for f, c in top)]
    if failed:
        lines.append("No data: " + ", ".join(failed))
    lines += ["", "Worth paper-trading only if 👉 wins in most months AND the 🔒 locked line is positive."]
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
