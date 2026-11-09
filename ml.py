"""
Learning bot v4: rules suggest trades, a model learns which ones usually fail,
and every trade is managed with two targets.

Trade plan (every call)
  Entry, stop-loss (SL), T1 and T2 in one message.
  At T1: book HALF, move the SL to entry (break-even) for the other half.
  Other half: T2, or back to entry (break-even), or squared off at 3:00 PM.

How it learns
  1. Seven setups from engine.py produce candidate trades on 40 shares.
  2. Each candidate gets 15 facts known at that moment (time of day, 15-min
     and daily trend, VWAP distance, RSI, volume, gap, today's move, NIFTY's
     move and trend...) and its result after Groww charges and slippage.
  3. One small model per setup learns which conditions lose money.
  4. The study tests side by side: T1 at 0.75x or 1x the risk, and "stop for
     the day after 2 losses" or no limit. The best on the tuning quarter is
     used; the newest quarter (never used for any choice) is the honest test.
  5. Live: only calls above the confidence cut-off, one trade at a time.

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
from backtest import SLIP_PCT, leg_cost
from sail_intraday_bot import IST, fmt, send

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_FILE = "ml_model.json"
STATE_FILE = "ml_state.json"
LOG_FILE = "ml_trades.csv"
INDEX = "NIFTYBEES"          # NIFTY 50 ETF: the market's direction
LEARN_DAYS = 380
WARMUP_DAYS = 25
HISTORY_DAYS = 45
MAX_TRADES_PER_DAY = 4
MIN_CALLS_PER_DAY = 0.3
MIN_TEST_TRADES = 15
RIDGE = 10.0
PLANS = {"A": 0.75, "B": 1.0}          # T1 distance as a multiple of the risk
LOSS_LIMITS = [2, 99]                  # stop for the day after N losing calls
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
            "day_move", "gap", "or_width", "mom30", "st_agree", "bb_pos",
            "nifty_day", "nifty_trend"]
PLAIN = {"time": "later in the day", "atr": "in jumpier shares",
         "trend15": "with the 15-min trend", "trend_day": "with the daily trend",
         "vwap_dist": "when price is already far past VWAP",
         "rsi": "when RSI is already stretched", "volume": "on a volume surge",
         "day_move": "when the share already moved a lot today",
         "gap": "after a gap in the trade direction", "or_width": "after a wide opening range",
         "mom30": "after a strong last-30-min move", "st_agree": "when the 15-min Supertrend agrees",
         "bb_pos": "when price is stretched outside its band",
         "nifty_day": "when NIFTY moved the same way today",
         "nifty_trend": "with NIFTY's 15-min trend"}
OUTCOME = {"T2": "🎯🎯 T1 + T2", "T1BE": "🎯 T1, rest at entry", "T1CLOSE": "🎯 T1, rest squared off",
           "STOP": "❌ stop-loss", "CLOSE": "⏰ squared off before T1"}


# ------------------------------ candidates -------------------------------
def index_context(df):
    """NIFTY ETF day move and 15-min trend per 5-min candle (by candle start)."""
    if df is None or df.empty:
        return None
    ind = E.indicators(df)
    C = ind["C"]
    with np.errstate(all="ignore"):
        return pd.DataFrame({"nd": (C - ind["dopen"]) / ind["dopen"] * 100,
                             "nt": (ind["m_e20"] - ind["m_e50"]) / C * 100},
                            index=ind["index"])


def features(ind, ii, sd, ctx):
    """15 facts known at the close of each signal candle, oriented so that
    positive = 'in the trade direction'."""
    C = ind["C"][ii]
    s = sd.astype(float)
    mins = np.array([(t.hour * 60 + t.minute) - 555 for t in ind["tod"][ii]], float)
    if ctx is not None:
        cx = ctx.reindex(ind["index"][ii])
        nd, nt = cx["nd"].to_numpy(float), cx["nt"].to_numpy(float)
    else:
        nd = nt = np.full(len(ii), np.nan)
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
            nd * s,
            nt * s,
        ])
    X[~np.isfinite(X)] = np.nan
    return X


def sim_two_targets(ind, j, side, p, i1, k1):
    """Entry at the close of candle j. Full position until SL or T1; at T1 half
    is booked and the SL moves to entry from the next candle; the rest runs to
    T2, back to entry, or the 3:00 square-off. SL is assumed first whenever a
    candle touches both."""
    H, L, C, atr, tod = ind["H"], ind["L"], ind["C"], ind["atr14"], ind["tod"]
    entry = E.tick(C[j])
    a = atr[j] if np.isfinite(atr[j]) else entry * E.MIN_SL_PCT / 100
    dist = min(max(p.get("am", E.ATR_MULT) * a, entry * E.MIN_SL_PCT / 100),
               entry * E.MAX_SL_PCT / 100)
    sl = E.tick(entry - side * dist)
    t1 = E.tick(entry + side * k1 * dist)
    t2 = E.tick(entry + side * p["R"] * dist)
    r = dict(entry=entry, sl=sl, t1=t1, t2=t2, k1=None, k=None, why=None, x1=None, x2=None)
    hit = (lambda k, lvl: H[k] >= lvl) if side == 1 else (lambda k, lvl: L[k] <= lvl)
    back = (lambda k, lvl: L[k] <= lvl) if side == 1 else (lambda k, lvl: H[k] >= lvl)
    for k in range(j + 1, i1):
        if r["k1"] is None:                              # full position
            if back(k, sl):
                r.update(k=k, why="STOP", x2=sl)
            elif hit(k, t1):
                r.update(k1=k, x1=t1)
                if hit(k, t2):
                    r.update(k=k, why="T2", x2=t2)
                elif tod[k] >= E.LAST_BAR:
                    r.update(k=k, why="T1CLOSE", x2=E.tick(C[k]))
            elif tod[k] >= E.LAST_BAR:
                r.update(k=k, why="CLOSE", x2=E.tick(C[k]))
        else:                                            # half left, SL at entry
            if back(k, entry):
                r.update(k=k, why="T1BE", x2=entry)
            elif hit(k, t2):
                r.update(k=k, why="T2", x2=t2)
            elif tod[k] >= E.LAST_BAR:
                r.update(k=k, why="T1CLOSE", x2=E.tick(C[k]))
        if r["k"] is not None:
            return r
    return r                                             # still open / data ended


def split_qty(capital, entry):
    q = max(1, int(capital // entry))
    q1 = q // 2
    return q, q1, q - q1


def net_result(side, r, capital):
    """Net Rs after charges and slippage; None while the trade is open."""
    if r["k"] is None:
        return None
    q, q1, q2 = split_qty(capital, r["entry"])
    s = SLIP_PCT / 100
    sign = 1 if side == 1 else -1
    es, xs = ("buy", "sell") if side == 1 else ("sell", "buy")
    e_fill = r["entry"] * (1 + sign * s)
    legs = [(q1, r["x1"]), (q2, r["x2"])] if r["x1"] is not None else [(q, r["x2"])]
    pnl = -leg_cost(q * e_fill, es)
    for qty, px in legs:
        if qty <= 0:
            continue
        x_fill = px * (1 - sign * s)
        pnl += sign * qty * (x_fill - e_fill) - leg_cost(qty * x_fill, xs)
    return pnl


def stock_candidates(sym, ind, capital, ctx, plans, day_from=None, only_day=None):
    """Every signal of every setup for one share, for each T1 plan."""
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
                for plan in plans:
                    r = sim_two_targets(ind, j, side, p, i1, PLANS[plan])
                    net = net_result(side, r, capital)
                    rows.append(dict(
                        sym=sym, setup=name, plan=plan, day=d, side=side,
                        t_sig=ind["index"][j] + STEP,
                        t_t1=(ind["index"][r["k1"]] + STEP) if r["k1"] is not None else pd.NaT,
                        t_exit=(ind["index"][r["k"]] + STEP) if r["k"] is not None else pd.NaT,
                        entry=r["entry"], sl=r["sl"], t1=r["t1"], t2=r["t2"], why=r["why"],
                        x1=r["x1"], x2=r["x2"], net=np.nan if net is None else net))
                    js.append(j)
                    sides.append(side)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    X = features(ind, np.array(js), np.array(sides), ctx)
    for c, name in enumerate(FEATURES):
        df[name] = X[:, c]
    return df


# -------------------------------- model ----------------------------------
def _z(X, mu, sd):
    Z = (X - np.asarray(mu)) / np.asarray(sd)
    Z[~np.isfinite(Z)] = 0.0
    return np.clip(Z, -5, 5)


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
        pr = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (pr - y) + reg * w
        Hm = (A * (pr * (1 - pr))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist())


def fit_models(df):
    models = {}
    for name in SETUPS:
        part = df[(df["setup"] == name) & df["net"].notna()]
        if len(part) >= 200:
            models[name] = fit_logit(part[FEATURES].to_numpy(float),
                                     (part["net"] > 0).to_numpy(float))
    return models


def predict(models, df):
    pr = np.full(len(df), np.nan)
    if df.empty:
        return pr
    X = df[FEATURES].to_numpy(float)
    for name, m in models.items():
        idx = np.flatnonzero((df["setup"] == name).to_numpy())
        if len(idx):
            A = np.column_stack([np.ones(len(idx)), _z(X[idx], m["mu"], m["sd"])])
            pr[idx] = 1 / (1 + np.exp(-np.clip(A @ np.asarray(m["w"]), -30, 30)))
    return pr


# -------------------------------- policy ---------------------------------
def policy(df, thr, max_losses=99, max_per_day=MAX_TRADES_PER_DAY):
    """One trade at a time across all shares, highest confidence first; no new
    trade after max_losses losing calls that day."""
    if df.empty:
        return []
    ok = df[df["p"] >= thr].sort_values(["t_sig", "p", "sym", "setup"],
                                        ascending=[True, False, True, True])
    chosen = []
    for d, g in ok.groupby("day", sort=True):
        busy, n, losses = None, 0, 0
        for r in g.itertuples():
            if n >= max_per_day or losses >= max_losses:
                break
            if busy is not None and (pd.isna(busy) or r.t_sig <= busy):
                continue
            chosen.append(r.Index)
            n += 1
            busy = r.t_exit
            if not pd.isna(r.net) and r.net < 0:
                losses += 1
    return chosen


def score(df, idx, ndays):
    part = df.loc[idx]
    part = part[part["net"].notna()]
    n = len(part)
    return dict(n=n, per_day=round(n / max(ndays, 1), 2),
                t1=round(float(part["x1"].notna().mean() * 100), 1) if n else 0.0,
                acc=round(float((part["net"] > 0).mean() * 100), 1) if n else 0.0,
                net=round(float(part["net"].sum()), 0))
  # -------------------------------- learn ----------------------------------
def learn():
    capital = float(os.getenv("BACKTEST_CAPITAL", "200000") or "200000")
    g = E.login()
    now = datetime.now(IST)
    start = (now - timedelta(days=LEARN_DAYS)).date()
    try:
        ctx = index_context(E.fetch_5min(g, INDEX, start, now))
    except Exception as e:
        print("index failed:", type(e).__name__, e)
        ctx = None
    frames, failed = [], []
    for sym in POOL:
        try:
            df = E.fetch_5min(g, sym, start, now)
            if len(df) < 75 * 60:
                failed.append(sym)
                continue
            ind = E.indicators(df)
            days = [d for d, _, _ in ind["days"]][WARMUP_DAYS:]
            c = stock_candidates(sym, ind, capital, ctx, list(PLANS), day_from=days[0])
            if not c.empty:
                frames.append(c)
            print(sym, len(c), "rows")
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
    nd_va = sum(1 for d in days if cut1 <= d < cut2)
    nd_te = sum(1 for d in days if d >= cut2)

    combos = []
    for plan in PLANS:
        P = allc[allc["plan"] == plan].copy()
        tr, va, te = P[P["day"] < cut1], P[(P["day"] >= cut1) & (P["day"] < cut2)], P[P["day"] >= cut2]
        m_train = fit_models(tr)                             # 1) learn on the oldest half
        va = va.assign(p=predict(m_train, va))
        te = te.assign(p=predict(m_train, te))
        for ml in LOSS_LIMITS:
            best = None                                      # 2) tune the cut-off
            for thr in np.arange(0.30, 0.801, 0.025):
                s = score(va, policy(va, thr, ml), nd_va)
                if s["per_day"] >= MIN_CALLS_PER_DAY and (best is None or s["net"] > best[1]["net"]):
                    best = (round(float(thr), 3), s)
            if best is None:
                best = (0.5, score(va, policy(va, 0.5, ml), nd_va))
            thr, val = best
            test = score(te, policy(te, thr, ml), nd_te)      # 3) honest unseen check
            base = score(te, policy(te, -1.0, ml), nd_te)     # same plan, no learning
            combos.append(dict(plan=plan, max_losses=ml, threshold=thr, validate=val,
                               test=test, baseline=base))
    best = max(combos, key=lambda c: c["validate"]["net"])  # chosen on tuning only
    status = "tested" if (best["test"]["net"] > 0 and best["validate"]["net"] > 0
                          and best["test"]["n"] >= MIN_TEST_TRADES) else "paper"
    models = fit_models(allc[allc["plan"] == best["plan"]])  # 4) final model, whole year
    cfg = dict(trained_at=now.isoformat(timespec="minutes"), capital=capital,
               plan=best["plan"], k1=PLANS[best["plan"]], max_losses=best["max_losses"],
               threshold=best["threshold"], status=status, validate=best["validate"],
               test=best["test"], baseline=best["baseline"], combos=combos,
               shares=[s for s in POOL if s not in failed], index_ok=ctx is not None,
               models=models, max_trades=MAX_TRADES_PER_DAY)
    with open(MODEL_FILE, "w")
  def check(market, model, state, now):
    calls, have = pick(market.frames(now), model, only_day=now.date())
    if not have and now.time() >= dtime(9, 40):
        send("⚠️ No candles from Groww today by 9:40. Market holiday or data problem: no calls today.")
        state["done"] = True
        return calls
    for r in calls.itertuples():
        tkey = f"{r.sym}:{r.t_sig:%H%M}:{r.setup}"
        ev = [("ENTRY", r.t_sig)]
        if not pd.isna(r.t_t1):
            ev.append(("T1", r.t_t1))
        if not pd.isna(r.t_exit):
            ev.append(("EXIT", r.t_exit))
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
            same = r.why == "T2" and r.t_t1 == r.t_exit       # T1 and T2 in one candle
            if kind == "T1" and same:
                continue
            if kind == "EXIT" and same:
                send(f"🎯🎯 {r.sym}: T1 and T2 ({fmt(r.t2)}) both reached. Exit the whole trade.")
                continue
            send(entry_msg(r, model) if kind == "ENTRY" else
                 t1_msg(r, model) if kind == "T1" else exit_msg(r))
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


def need_model(now):
    model = load_json(MODEL_FILE)
    if model and "plan" not in model:
        model = None                                       # model from an older version
    if not model and now.time() < dtime(11, 30):
        send("⚠️ Learning bot: no up-to-date model. Run the 'Learning study' workflow first.")
    return model


def run():
    now = datetime.now(IST)
    model = need_model(now)
    if not model:
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
    model = need_model(datetime.combine(now.date(), dtime(9, 0), tzinfo=IST))
    g = E.login()
    df = E.fetch_5min(g, "SBIN", now.date() - timedelta(days=7), now)
    last = (f"Latest SBIN candle: {fmt(float(df['Close'].iloc[-1]))} at {df.index[-1]:%d %b %H:%M}"
            if not df.empty else "No candles received.")
    send(f"✅ Learning bot test: Groww OK, Telegram OK.\n{last}"
         + (f"\n\n{plan_msg(model, now, False)}" if model else ""))


def replay():
    days_n = max(1, int(os.getenv("REPLAY_DAYS", "1") or "1"))
    now = datetime.now(IST)
    model = need_model(datetime.combine(now.date(), dtime(9, 0), tzinfo=IST))
    if not model:
        return
    g = E.login()
    start = now.date() - timedelta(days=HISTORY_DAYS + days_n * 2 + 7)
    frames = {}
    for sym in model["shares"] + [INDEX]:
        try:
            df = E.fetch_5min(g, sym, start, now, pause=0.1)
            frames[sym] = df[df.index + STEP <= now]
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
    days = sorted({ts.date() for s, df in frames.items() if s != INDEX
                   for ts in df.index if ts.time() == dtime(9, 15)})[-days_n:]
    if not days:
        send("🔁 Replay: no data received from Groww.")
        return
    chosen, _ = pick(frames, model, day_from=days[0])
    lines = [f"🔁 DRY RUN v4 ({len(days)} day(s)): T1 {model['k1']:g}x risk (book half, SL to entry), "
             f"T2 for the rest, cut-off {model['threshold'] * 100:.0f}%, Rs {model['capital']:,.0f} "
             f"per call, after charges, one trade at a time. No orders were placed."]
    tot = []
    for d in days:
        part = chosen[chosen["day"] == d] if not chosen.empty else chosen
        lines += ["", f"📅 {d:%a %d %b}"]
        if part.empty:
            lines.append("  No calls.")
            continue
        day_net = 0.0
        for r in part.itertuples():
            icon = "🟢 BUY" if r.side == 1 else "🔴 SELL"
            lines.append(f"  {r.t_sig:%H:%M} {icon} {r.sym} at {fmt(r.entry)} (SL {fmt(r.sl)}, "
                         f"T1 {fmt(r.t1)}, T2 {fmt(r.t2)}) conf {r.p * 100:.0f}%")
            if pd.isna(r.net):
                lines.append("        still open when the data ended")
                continue
            t1 = f"T1 at {r.t_t1:%H:%M}, " if not pd.isna(r.t_t1) else ""
            lines.append(f"        {t1}{OUTCOME[r.why]} at {r.t_exit:%H:%M} -> Rs {r.net:+,.0f}")
            tot.append((r.net, not pd.isna(r.x1)))
            day_net += r.net
        lines.append(f"  Day: net Rs {day_net:+,.0f}")
    if tot:
        n = len(tot)
        lines += ["", f"TOTAL: {n} calls, T1 reached {sum(x[1] for x in tot) / n * 100:.0f}%, "
                      f"in profit {sum(x[0] > 0 for x in tot) / n * 100:.0f}%, "
                      f"net Rs {sum(x[0] for x in tot):+,.0f} after charges."]
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
