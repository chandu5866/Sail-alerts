"""
Learning bot: rules suggest trades, a model learns which ones usually fail.

How it works
  1. Seven setups from engine.py (Bollinger reversal, RSI pullback, VWAP
     pullback, buy-the-dip at the open, opening-range breakout, 15-min EMA
     cross, 15-min Supertrend) produce candidate trades on 40 liquid shares.
  2. For every candidate it records 13 facts known at that moment (time of
     day, 15-min and daily trend, distance from VWAP, RSI, volume surge, gap,
     how far the share already moved today...) and whether the trade made
     money after Groww charges and slippage.
  3. One small model per setup (regularised logistic regression) learns which
     conditions lead to losses and gives each new signal a confidence score.
  4. Live, it sends only calls above the confidence cut-off, ONE TRADE AT A
     TIME (the highest-confidence signal wins), at most 4 a day.
  5. Every Saturday it re-learns from the latest year, including last week.

Honest testing: model fitted on the oldest half, cut-off tuned on the next
quarter, then scored on the newest quarter it never saw. ✅ only if that
unseen quarter made money.

Modes:  python ml.py learn | run | test | replay   (see learn.yml, ml_live.yml)

Past results do not predict future results. Not financial advice.
"""

import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

import engine as E
from sail_intraday_bot import IST, fmt, send

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_FILE = "ml_model.json"
STATE_FILE = "ml_state.json"
LOG_FILE = "ml_trades.csv"
LEARN_DAYS = 380             # calendar days of history to learn from
WARMUP_DAYS = 25             # first days only warm up the indicators
HISTORY_DAYS = 45            # live: calendar days of history for the indicators
MAX_TRADES_PER_DAY = 4
MIN_CALLS_PER_DAY = 0.3      # a cut-off must give at least this many calls a day
MIN_TEST_TRADES = 15
RIDGE = 10.0                 # model regularisation (higher = simpler model)
HANDOFF = dtime(12, 0, 30)
DAY_END = dtime(15, 31)
SUMMARY_AT = dtime(15, 0, 20)
LATE_MIN = 8
GRACE_S = 25
STEP = timedelta(minutes=E.BAR_MIN)

POOL = ["SAIL", "TATASTEEL", "SBIN", "ICICIBANK", "HDFCBANK", "AXISBANK",
        "RELIANCE", "INFY", "TATAPOWER", "BANKBARODA", "PNB", "ONGC", "NTPC",
        "COALINDIA", "VEDL", "HINDALCO", "JSWSTEEL", "BEL", "BHEL", "NMDC",
        "LT", "KOTAKBANK", "MARUTI", "SUNPHARMA", "BHARTIARTL", "ITC",
        "HINDUNILVR", "BAJFINANCE", "TITAN", "ASIANPAINT", "ADANIPORTS",
        "ADANIENT", "TECHM", "WIPRO", "HCLTECH", "DLF", "GAIL", "IOC", "BPCL",
        "CANBK"]

# Loosest version of each setup: the model learns when trend filters help.
SETUPS = {
    "bb": dict(kind="bb", rsi=25, tf="none", R=1.5),
    "rsipb": dict(kind="rsipb", tf="none", R=1.5),
    "vwap": dict(kind="vwap", tf="none", R=1.5),
    "dip": dict(kind="dip", tf="day", R=1.5),
    "orb": dict(kind="orb", vol=1.0, tf="none", R=2.0),
    "ema15": dict(kind="ema15", tf="none", am=2.5, R=2.0),
    "st15": dict(kind="st15", tf="none", am=2.5, R=2.0),
}
SETUP_NAME = {"bb": "Bollinger reversal", "rsipb": "RSI pullback", "vwap": "VWAP pullback",
              "dip": "Dip at the open", "orb": "Opening-range breakout",
              "ema15": "15-min EMA cross", "st15": "15-min Supertrend"}
FEATURES = ["time", "atr", "trend15", "trend_day", "vwap_dist", "rsi", "volume",
            "day_move", "gap", "or_width", "mom30", "st_agree", "bb_pos"]
PLAIN = {"time": "later in the day", "atr": "in jumpier shares",
         "trend15": "with the 15-min trend", "trend_day": "with the daily trend",
         "vwap_dist": "when price is already far past VWAP",
         "rsi": "when RSI is already stretched", "volume": "on a volume surge",
         "day_move": "when the share already moved a lot today",
         "gap": "after a gap in the trade direction", "or_width": "after a wide opening range",
         "mom30": "after a strong last-30-min move", "st_agree": "when the 15-min Supertrend agrees",
         "bb_pos": "when price is stretched outside its band"}


# ------------------------------ candidates -------------------------------
def features(ind, ii, sd):
    """13 facts known at the close of each signal candle, oriented so that
    positive = 'in the trade direction'."""
    C = ind["C"][ii]
    s = sd.astype(float)
    mins = np.array([(t.hour * 60 + t.minute) - 555 for t in ind["tod"][ii]], float)
    with np.errstate(all="ignore"):
        va, V = ind["va"][ii], ind["V"][ii]
        vol = np.log(np.where(np.isfinite(va) & (va > 0), V / va, 1.0))
        mid = (ind["bb_lo"][ii] + ind["bb_hi"][ii]) / 2
        sdv = (ind["bb_hi"][ii] - ind["bb_lo"][ii]) / 4
        c6 = ind["C"][np.maximum(ii - 6, 0)]
        X = np.column_stack([
            mins / 375,
            ind["atr14"][ii] / C * 100,
            (ind["m_e20"][ii] - ind["m_e50"][ii]) / C * 100 * s,
            (ind["d_c"][ii] - ind["d_sma"][ii]) / ind["d_sma"][ii] * 100 * s,
            (C - ind["vwap"][ii]) / C * 100 * s,
            (ind["rsi7"][ii] - 50) / 50 * s,
            vol,
            (C - ind["dopen"][ii]) / ind["dopen"][ii] * 100 * s,
            (ind["dopen"][ii] - ind["d_c"][ii]) / ind["d_c"][ii] * 100 * s,
            (ind["orh"][ii] - ind["orl"][ii]) / C * 100,
            (C - c6) / C * 100 * s,
            ind["m_st"][ii] * s,
            np.where(sdv > 0, (C - mid) / sdv, 0.0) * s,
        ])
    X[~np.isfinite(X)] = np.nan
    return X


def sim_trade(ind, j, side, p, i1):
    """One trade from signal candle j, same exits as engine.run_day."""
    H, L, C, atr, tod = ind["H"], ind["L"], ind["C"], ind["atr14"], ind["tod"]
    entry = E.tick(C[j])
    a = atr[j] if np.isfinite(atr[j]) else entry * E.MIN_SL_PCT / 100
    dist = min(max(p.get("am", E.ATR_MULT) * a, entry * E.MIN_SL_PCT / 100),
               entry * E.MAX_SL_PCT / 100)
    sl, tgt = E.tick(entry - side * dist), E.tick(entry + side * p["R"] * dist)
    for k in range(j + 1, i1):
        if (L[k] <= sl) if side == 1 else (H[k] >= sl):
            return entry, sl, tgt, k, "STOP", sl
        if (H[k] >= tgt) if side == 1 else (L[k] <= tgt):
            return entry, sl, tgt, k, "TARGET", tgt
        if tod[k] >= E.LAST_BAR:
            return entry, sl, tgt, k, "CLOSE", E.tick(C[k])
    return entry, sl, tgt, None, None, None        # still open / data ended


def stock_candidates(sym, ind, capital, day_from=None, only_day=None):
    """Every signal of every setup for one share, with features and outcome."""
    rows, js, sides = [], [], []
    for name, p in SETUPS.items():
        s = E.signals(ind, p)
        for d, i0, i1 in ind["days"]:
            if (day_from is not None and d < day_from) or (only_day is not None and d != only_day):
                continue
            if ind["tod"][i0] != dtime(9, 15):      # skip special short sessions (Muhurat)
                continue
            for j in np.flatnonzero(s[i0:i1]) + i0:
                side = int(s[j])
                entry, sl, tgt, k, why, px = sim_trade(ind, j, side, p, i1)
                net = (E.net_pnl("BUY" if side == 1 else "SELL", entry, px, capital)
                       if px is not None else np.nan)
                rows.append(dict(sym=sym, setup=name, day=d, side=side,
                                 t_sig=ind["index"][j] + STEP,
                                 t_exit=(ind["index"][k] + STEP) if k is not None else pd.NaT,
                                 entry=entry, sl=sl, tgt=tgt, why=why, exit=px, net=net))
                js.append(j)
                sides.append(side)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    X = features(ind, np.array(js), np.array(sides))
    for c, name in enumerate(FEATURES):
        df[name] = X[:, c]
    return df


# -------------------------------- model ----------------------------------
def fit_logit(X, y, lam=RIDGE, iters=30):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    A = np.column_stack([np.ones(len(X)), _z(X, mu, sd)])
    w = np.zeros(A.shape[1])
    reg = np.full(A.shape[1], lam)
    reg[0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - y) + reg * w
        Hm = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


def _z(X, mu, sd):
    Z = (X - np.asarray(mu)) / np.asarray(sd)
    Z[~np.isfinite(Z)] = 0.0
    return np.clip(Z, -5, 5)


def predict(models, df):
    p = np.full(len(df), np.nan)
    if df.empty:
        return p
    X = df[FEATURES].to_numpy(float)
    for name, m in models.items():
        idx = np.flatnonzero((df["setup"] == name).to_numpy())
        if len(idx):
            A = np.column_stack([np.ones(len(idx)), _z(X[idx], m["mu"], m["sd"])])
            p[idx] = 1 / (1 + np.exp(-np.clip(A @ np.asarray(m["w"]), -30, 30)))
    return p


def fit_models(df):
    models = {}
    for name in SETUPS:
        part = df[(df["setup"] == name) & df["net"].notna()]
        if len(part) >= 200:
            models[name] = fit_logit(part[FEATURES].to_numpy(float),
                                     (part["net"] > 0).to_numpy(float))
    return models


# -------------------------------- policy ---------------------------------
def policy(df, thr, max_per_day=MAX_TRADES_PER_DAY):
    """One trade at a time across all shares: at each moment take the
    highest-confidence signal above the cut-off; wait until it is closed."""
    if df.empty:
        return []
    ok = df[df["p"] >= thr].sort_values(["t_sig", "p", "sym", "setup"],
                                        ascending=[True, False, True, True])
    chosen = []
    for d, g in ok.groupby("day", sort=True):
        busy, n = None, 0
        for r in g.itertuples():
            if n >= max_per_day:
                break
            if busy is not None and (pd.isna(busy) or r.t_sig <= busy):
                continue
            chosen.append(r.Index)
            n += 1
            busy = r.t_exit
    return chosen


def score(df, idx, ndays):
    part = df.loc[idx]
    part = part[part["net"].notna()]
    n = len(part)
    return dict(n=n, per_day=round(n / max(ndays, 1), 2),
                acc=round(float((part["net"] > 0).mean() * 100), 1) if n else 0.0,
                net=round(float(part["net"].sum()), 0))


# -------------------------------- learn ----------------------------------
def learn():
    capital = float(os.getenv("BACKTEST_CAPITAL", "200000") or "200000")
    g = E.login()
    now = datetime.now(IST)
    start = (now - timedelta(days=LEARN_DAYS)).date()
    frames, failed = [], []
    for sym in POOL:
        try:
            df = E.fetch_5min(g, sym, start, now)
            if len(df) < 75 * 60:
                failed.append(sym)
                continue
            ind = E.indicators(df)
            days = [d for d, _, _ in ind["days"]][WARMUP_DAYS:]
            c = stock_candidates(sym, ind, capital, day_from=days[0])
            if not c.empty:
                frames.append(c)
            print(sym, len(c), "signals")
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
            failed.append(sym)
    if not frames:
        send("🤖 Learning study: no data received from Groww.")
        return
    allc = pd.concat(frames, ignore_index=True)
    days = sorted(allc["day"].unique())
    n = len(days)
    cut1, cut2 = days[n // 2], days[3 * n // 4]
    tr, va, te = (allc["day"] < cut1), (allc["day"] >= cut1) & (allc["day"] < cut2), allc["day"] >= cut2
    nd_va, nd_te = sum(1 for d in days if cut1 <= d < cut2), sum(1 for d in days if d >= cut2)

    # 1) learn on the oldest half
    m_train = fit_models(allc[tr])
    allc["p"] = predict(m_train, allc)
    # 2) tune the confidence cut-off on the next quarter
    best = None
    for thr in np.arange(0.30, 0.801, 0.025):
        s = score(allc[va], policy(allc[va], thr), nd_va)
        if s["per_day"] >= MIN_CALLS_PER_DAY and (best is None or s["net"] > best[1]["net"]):
            best = (round(float(thr), 3), s)
    if best is None:
        best = (0.5, score(allc[va], policy(allc[va], 0.5), nd_va))
    thr, val = best
    # 3) honest check on the newest quarter (never used for any choice)
    test = score(allc[te], policy(allc[te], thr), nd_te)
    base = score(allc[te], policy(allc[te], -1.0), nd_te)      # same rules, no learning
    status = "tested" if (test["net"] > 0 and val["net"] > 0
                          and test["n"] >= MIN_TEST_TRADES) else "paper"
    # 4) final model learns from the whole year, newest weeks included
    models = fit_models(allc)
    cfg = dict(trained_at=now.isoformat(timespec="minutes"), capital=capital,
               threshold=thr, status=status, validate=val, test=test, baseline=base,
               shares=[s for s in POOL if s not in failed], models=models,
               max_trades=MAX_TRADES_PER_DAY)
    with open(MODEL_FILE, "w") as f:
        json.dump(cfg, f)

    # ---- report ----
    icon = "✅" if status == "tested" else "📝"
    lines = [f"🤖 Learning study: {len(cfg['shares'])} shares x {len(SETUPS)} setups, "
             f"{len(allc):,} past signals learned from (~{n} trading days), "
             f"Rs {capital:,.0f} per trade, after charges.", "",
             f"Honest test on the newest {nd_te} days (never seen while learning):",
             f"  Without learning (take every signal): {base['n']} trades, "
             f"{base['acc']:.0f}% in profit, Rs {base['net']:+,.0f}",
             f"  With learning (confidence >= {thr * 100:.0f}%): {test['n']} trades "
             f"({test['per_day']:.1f}/day), {test['acc']:.0f}% in profit, Rs {test['net']:+,.0f}",
             f"  (Tuning period: {val['n']} trades, {val['acc']:.0f}% in profit, "
             f"Rs {val['net']:+,.0f})", "",
             f"{icon} Status: " + ("LIVE calls - passed the unseen test." if status == "tested"
                                  else "PAPER only - did not pass the unseen test. Watch, "
                                       "don't trade with real money."), "",
             "What it learned (strongest patterns per setup):"]
    for name, m in models.items():
        w = np.asarray(m["w"][1:])
        order = np.argsort(-np.abs(w))[:2]
        bits = [f"{'wins more' if w[k] > 0 else 'loses more'} {PLAIN[FEATURES[k]]}" for k in order]
        lines.append(f"  {SETUP_NAME[name]}: " + "; ".join(bits))
    if failed:
        lines.append("No data for: " + ", ".join(failed))
    send("\n".join(lines))


# --------------------------------- live -----------------------------------
def load_model():
    try:
        with open(MODEL_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_state(today):
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        if s.get("date") == str(today):
            return s
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return dict(date=str(today), sent=[], missed=[], plan_sent=False, done=False)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


class Market:
    """Keeps each share's older candles for the whole job; fetches only
    today's candles on every check."""

    def __init__(self, g, shares, today):
        self.g, self.shares, self.today, self.hist = g, shares, today, {}

    def frames(self, now):
        out = {}
        y_end = datetime.combine(self.today - timedelta(days=1), dtime(15, 30), tzinfo=IST)
        for sym in self.shares:
            try:
                if sym not in self.hist:
                    self.hist[sym] = E.fetch_5min(
                        self.g, sym, self.today - timedelta(days=HISTORY_DAYS), y_end, pause=0.1)
                today = E.fetch_5min(self.g, sym, self.today, now, pause=0.05)
            except Exception as e:
                print(sym, "data failed:", type(e).__name__, e)
                continue
            df = pd.concat([self.hist[sym], today]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            out[sym] = df[df.index + STEP <= now]                 # completed candles only
        return out


def day_calls(frames, model, day):
    """Today's chosen calls (one at a time), exactly as the backtest picks them."""
    parts = []
    for sym, df in frames.items():
        if df.empty or day not in set(df.index.date):
            continue
        c = stock_candidates(sym, E.indicators(df), model["capital"], only_day=day)
        if not c.empty:
            parts.append(c)
    if not parts:
        return pd.DataFrame(), False
    allc = pd.concat(parts, ignore_index=True)
    allc["p"] = predict(model["models"], allc)
    return allc.loc[policy(allc, model["threshold"], model["max_trades"])], True


def plan_msg(model, now, late):
    te, base = model["test"], model["baseline"]
    icon = "✅ LIVE" if model["status"] == "tested" else "📝 PAPER ONLY"
    lines = [f"🤖 Learning bot - {now:%a %d %b} ({icon})",
             f"Watching {len(model['shares'])} shares, 7 setups. A call is sent only when "
             f"its confidence is >= {model['threshold'] * 100:.0f}%, one trade at a time, "
             f"max {model['max_trades']} a day, Rs {model['capital']:,.0f} each.",
             f"Unseen test: {te['n']} trades, {te['acc']:.0f}% in profit, Rs {te['net']:+,.0f} "
             f"(without learning: {base['acc']:.0f}%, Rs {base['net']:+,.0f}).",
             f"Model from {model['trained_at'][:10]}. Calls 9:35 - 2:30, square off by 3:00."]
    if model["status"] != "tested":
        lines.append("📝 Paper only: watch the calls, don't use real money.")
    if late:
        lines.append(f"⚠️ Started late ({now:%H:%M}): calls older than {LATE_MIN} min are "
                     f"marked missed.")
    return "\n".join(lines)


def entry_msg(r, model):
    qty = max(1, int(model["capital"] // r.entry))
    verb = "BUY" if r.side == 1 else "SELL (short)"
    icon = "🟢" if r.side == 1 else "🔴"
    tag = "" if model["status"] == "tested" else "\n📝 Paper call."
    return (f"{icon} {verb} {r.sym} at ~{fmt(r.entry)}  ({r.t_sig:%H:%M})\n"
            f"SL {fmt(r.sl)} | Target {fmt(r.tgt)}\n"
            f"Confidence {r.p * 100:.0f}% | {SETUP_NAME[r.setup]}\n"
            f"Qty ~{qty}. Intraday (MIS). Skip if the price already moved 0.3% past entry.{tag}")


def exit_msg(r):
    side = "BUY" if r.side == 1 else "SELL"
    if r.why == "TARGET":
        return f"🎯 {r.sym}: target {fmt(r.exit)} reached. Exit the {side} trade."
    if r.why == "STOP":
        return f"❌ {r.sym}: SL {fmt(r.exit)} hit. Exit the {side} trade."
    return f"⏰ {r.sym}: square off your {side} trade now (last {fmt(r.exit)})."


def scorecard(state, calls, model):
    rows = []
    for r in calls.itertuples():
        if pd.isna(r.net):
            continue
        rows.append(dict(date=state["date"], symbol=r.sym, side="BUY" if r.side == 1 else "SELL",
                         setup=r.setup, confidence=round(r.p, 3), entry=r.entry, exit=r.exit,
                         why=r.why, net=round(r.net, 2),
                         alerted=int(f"{r.sym}:{r.t_sig:%H%M}:{r.setup}" not in state["missed"]),
                         status=model["status"]))
    if rows:
        new_file = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new_file:
                w.writeheader()
            w.writerows(rows)
    lines = [f"📊 Day scorecard {state['date']}"]
    if not rows:
        lines.append("No calls today (nothing passed the confidence cut-off).")
    for x in rows:
        mark = "✅" if x["net"] > 0 else "❌"
        miss = " (missed alert)" if not x["alerted"] else ""
        lines.append(f"{mark} {x['side']} {x['symbol']} {fmt(x['entry'])} -> {fmt(x['exit'])} "
                     f"{x['why'].lower()}, conf {x['confidence'] * 100:.0f}%, "
                     f"net Rs {x['net']:+,.0f}{miss}")
    if rows:
        lines.append(f"Today: {sum(x['net'] > 0 for x in rows)}/{len(rows)} in profit, "
                     f"net Rs {sum(x['net'] for x in rows):+,.0f} after charges")
    try:
        with open(LOG_FILE) as f:
            hist = list(csv.DictReader(f))
    except FileNotFoundError:
        hist = []
    for name, part in (("Last 30 calls", hist[-30:]), ("All calls so far", hist)):
        if part:
            n = len(part)
            wins = sum(1 for x in part if float(x["net"]) > 0)
            tot = sum(float(x["net"]) for x in part)
            lines.append(f"{name}: {n}, accuracy {wins / n * 100:.0f}%, net Rs {tot:+,.0f}")
    send("\n".join(lines))


def check(market, model, state, now):
    calls, have = day_calls(market.frames(now), model, now.date())
    if not have and now.time() >= dtime(9, 40):
        send("⚠️ No candles from Groww today by 9:40. Market holiday or data problem: "
             "no calls today.")
        state["done"] = True
        return calls
    for r in calls.itertuples():
        tkey = f"{r.sym}:{r.t_sig:%H%M}:{r.setup}"
        ev = [("ENTRY", r.t_sig)] + ([("EXIT", r.t_exit)] if not pd.isna(r.t_exit) else [])
        for kind, ts in ev:
            key = f"{tkey}:{kind}"
            if key in state["sent"]:
                continue
            state["sent"].append(key)
            if kind == "ENTRY" and (now - ts).total_seconds() / 60 > LATE_MIN:
                state["missed"].append(tkey)
                send(f"⏱ Missed: {'BUY' if r.side == 1 else 'SELL'} {r.sym} was due at "
                     f"{ts:%H:%M} (~{fmt(r.entry)}). Too late now. Don't chase it.")
                continue
            if tkey in state["missed"]:
                continue
            send(entry_msg(r, model) if kind == "ENTRY" else exit_msg(r))
    return calls


def next_wake(now, end_dt):
    base = now.replace(hour=9, minute=15, second=0, microsecond=0)
    k = max(1, int((now - base - timedelta(seconds=GRACE_S)) // STEP) + 1)
    return min(base + k * STEP + timedelta(seconds=GRACE_S), end_dt)


def session(market, model, state, end_t, clock=lambda: datetime.now(IST), sleep=time.sleep):
    today = clock().date()
    end_dt = datetime.combine(today, end_t, tzinfo=IST)
    first = datetime.combine(today, dtime(9, 35, GRACE_S), tzinfo=IST)
    catch_up = clock() >= first
    while True:
        now = clock()
        if now >= end_dt:
            break
        if not catch_up:
            wake = max(next_wake(now, end_dt), min(first, end_dt))
            if wake > now:
                sleep((wake - now).total_seconds())
        catch_up = False
        now = clock()
        calls = check(market, model, state, now)
        if not state["done"] and now.time() >= SUMMARY_AT:
            scorecard(state, calls, model)
            state["done"] = True
        save_state(state)
        if state["done"]:
            break


def run():
    now = datetime.now(IST)
    model = load_model()
    if not model:
        if now.time() < dtime(11, 30):
            send("⚠️ Learning bot: no model yet. Run the 'Learning study' workflow first.")
        return
    if now.weekday() >= 5:
        print("Weekend.")
        return
    state = load_state(now.date())
    if state["done"]:
        print("Today is already finished.")
        return
    if not state["plan_sent"]:
        send(plan_msg(model, now, now.time() > dtime(9, 35)))
        state["plan_sent"] = True
        save_state(state)
    market = Market(E.login(), model["shares"], now.date())
    session(market, model, state, HANDOFF if now.time() < dtime(11, 30) else DAY_END)
    save_state(state)


def test():
    now = datetime.now(IST)
    model = load_model()
    g = E.login()
    df = E.fetch_5min(g, "SBIN", now.date() - timedelta(days=7), now)
    last = (f"Latest SBIN candle: {fmt(float(df['Close'].iloc[-1]))} at {df.index[-1]:%d %b %H:%M}"
            if not df.empty else "No candles received.")
    send(f"✅ Learning bot test: Groww OK, Telegram OK.\n{last}\n\n"
         + (plan_msg(model, now, False) if model else
            "No model yet: run the 'Learning study' workflow."))


def replay():
    days_n = max(1, int(os.getenv("REPLAY_DAYS", "1") or "1"))
    model = load_model()
    if not model:
        send("🔁 Replay: no model yet. Run the 'Learning study' workflow first.")
        return
    g = E.login()
    now = datetime.now(IST)
    start = now.date() - timedelta(days=HISTORY_DAYS + days_n * 2 + 7)
    frames = {}
    for sym in model["shares"]:
        try:
            df = E.fetch_5min(g, sym, start, now, pause=0.1)
            frames[sym] = df[df.index + STEP <= now]
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
    all_days = sorted({ts.date() for df in frames.values()            # normal sessions only
                       for ts in df.index if ts.time() == dtime(9, 15)})
    days = all_days[-days_n:]
    parts = []
    for sym, df in frames.items():
        if df.empty:
            continue
        ind = E.indicators(df)
        c = stock_candidates(sym, ind, model["capital"], day_from=days[0])
        if not c.empty:
            parts.append(c)
    allc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not allc.empty:
        allc["p"] = predict(model["models"], allc)
    chosen = allc.loc[policy(allc, model["threshold"], model["max_trades"])] if not allc.empty else allc
    lines = [f"🔁 DRY RUN - learning bot ({len(days)} day(s)), cut-off "
             f"{model['threshold'] * 100:.0f}%, Rs {model['capital']:,.0f} per call, after charges, "
             f"one trade at a time. No orders were placed."]
    tn = tw = 0
    tnet = 0.0
    for d in days:
        part = chosen[chosen["day"] == d] if not chosen.empty else chosen
        lines += ["", f"📅 {d:%a %d %b}"]
        if part.empty:
            lines.append("  No calls.")
            continue
        dn = dw = 0
        dnet = 0.0
        for r in part.itertuples():
            icon = "🟢 BUY" if r.side == 1 else "🔴 SELL"
            lines.append(f"  {r.t_sig:%H:%M} {icon} {r.sym} at {fmt(r.entry)} "
                         f"(SL {fmt(r.sl)}, Tg {fmt(r.tgt)}) conf {r.p * 100:.0f}%")
            if pd.isna(r.net):
                lines.append("        still open when the data ended")
                continue
            what = {"TARGET": "🎯 target", "STOP": "❌ stop-loss", "CLOSE": "⏰ squared off"}[r.why]
            lines.append(f"        {r.t_exit:%H:%M} {what} at {fmt(r.exit)} -> Rs {r.net:+,.0f}")
            dn += 1
            dw += r.net > 0
            dnet += r.net
        lines.append(f"  Day: {dw}/{dn} in profit, net Rs {dnet:+,.0f}")
        tn, tw, tnet = tn + dn, tw + dw, tnet + dnet
    if len(days) > 1 and tn:
        lines += ["", f"TOTAL: {tn} calls, {tw / tn * 100:.0f}% in profit, "
                      f"net Rs {tnet:+,.0f} after charges."]
    buf = ""
    for p in lines:
        if buf and len(buf) + len(p) + 1 > 3500:
            send(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        send(buf)


if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("RUN_MODE") or "run").strip().lower()
    {"learn": learn, "run": run, "test": test, "replay": replay}.get(mode, run)()
