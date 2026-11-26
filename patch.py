# patch 4: 3 years of history + locked test, max-loss sizing, history line on calls, skip results days
import ast

s = open("ml.py").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, "not found exactly once: " + old[:60]
    s = s.replace(old, new)


rep("LEARN_DAYS = 380", "LEARN_DAYS = 1100             # ~3 years of 5-min history (if Groww has it)\nLOCK_DAYS = 40")

rep("def tune_thr(va, nd, ml):", '''def fetch_long(g, sym, start, end):
    """Long history in 180-day pieces; older pieces Groww doesn't have are skipped."""
    parts, cur = [], start
    while cur < end.date():
        nxt = min(cur + timedelta(days=180), end.date())
        try:
            parts.append(E.fetch_5min(g, sym, cur, datetime.combine(nxt, dtime(15, 30), tzinfo=IST)))
        except Exception as e:
            if (end.date() - cur).days < 400:
                raise
            print(sym, "no older data from", cur, type(e).__name__)
        cur = nxt
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="last")]


def tune_thr(va, nd, ml):''')
rep("ctx = index_context(E.fetch_5min(g, INDEX, start, now))",
    "ctx = index_context(fetch_long(g, INDEX, start, now))")
rep("            df = E.fetch_5min(g, sym, start, now)\n",
    "            df = fetch_long(g, sym, start, now)\n")

rep("    wf, blocks = walk_forward(allc, days, cut1)",
    "    wf, blocks = walk_forward(allc, days[:-LOCK_DAYS], cut1)")
rep('''    w = wf[pick_plan]
    status = "tested" if (w["net"] > 0''', '''    w = wf[pick_plan]
    lock_d = list(days[-LOCK_DAYS:])                       # 🔒 never used for any choice
    PL = allc[allc["plan"] == pick_plan]
    LK = PL[PL["day"].isin(lock_d)].copy()
    LK["p"] = predict(fit_models(PL[PL["day"] < lock_d[0]]), LK)
    lthr = max(best["threshold"], MIN_CONF)
    locked = score(LK, policy(LK, lthr, LOSS_LIMITS[0]), len(lock_d))
    status = "tested" if (locked["net"] > 0 and w["net"] > 0''')
rep("backup_p=BACKUP_P, wf=wf)", "backup_p=BACKUP_P, wf=wf, locked=locked)")
rep('''    wl += ["", f"Live cut-off for plan {best['plan']}''', '''    wl += ["", f"🔒 Locked test - newest {LOCK_DAYS} days ({lock_d[0]:%d %b} - {lock_d[-1]:%d %b}), "
               f"never used for any choice. Plan {best['plan']} at conf >= {lthr * 100:.0f}% (as live): "
               f"{locked['n']} calls, profit {locked['acc']:.0f}%, Rs {locked['net']:+,.0f}"]
    wl += ["", f"Live cut-off for plan {best['plan']}''')
rep('"✅ Status: LIVE calls - passed walk-forward (profit, and won in 60%+ of months)."',
    '"✅ Status: LIVE calls - passed walk-forward (profit, 60%+ winning months) AND the locked test."')
rep('"📝 Status: PAPER only - did not pass walk-forward. Watch, don\'t use real money."',
    '"📝 Status: PAPER only - needs walk-forward AND locked test to pass. Watch, don\'t use real money."')

# --- max loss per trade: quantity is sized so a stop-loss costs about MAX_LOSS ---
rep("HISTORY_DAYS = 45", "HISTORY_DAYS = 90\nMAX_LOSS = 1000                # Rs lost if the SL hits (qty sized to this, capital is the cap)")
rep('''def split_qty(capital, entry):
    q = max(1, int(capital // entry))''', '''def split_qty(capital, entry, sl=None):
    q = max(1, int(capital // entry))
    if sl is not None and MAX_LOSS and abs(entry - sl) > 0:
        q = max(1, min(q, int(MAX_LOSS // abs(entry - sl))))''')
rep('q, q1, q2 = split_qty(capital, r["entry"])', 'q, q1, q2 = split_qty(capital, r["entry"], r["sl"])')
s = s.replace('q, q1, q2 = split_qty(model["capital"], r.entry)', 'q, q1, q2 = split_qty(model["capital"], r.entry, r.sl)')
rep('f"Qty {q} | Confidence {r.p * 100:.0f}% | {SETUP_NAME[r.setup]}\\n"',
    'f"Qty {q} (~Rs {q * r.entry:,.0f}, risk Rs {q * abs(r.entry - r.sl):,.0f}) | Confidence {r.p * 100:.0f}% | {SETUP_NAME[r.setup]}\\n"')

# --- history line on every call: how this setup did on this share recently ---
rep("def check(market, model, state, now):", '''def history_line(market, model, now, r):
    try:
        if getattr(market, "hstats", None) is None:
            fr = market.frames(now)
            ctx = index_context(fr.get(INDEX)) if model.get("index_ok", True) else None
            market.hstats = {}
            for sym, df in fr.items():
                if sym == INDEX or df.empty:
                    continue
                c = stock_candidates(sym, E.indicators(df), model["capital"], ctx, [model["plan"]])
                if c.empty:
                    continue
                c = c[(c["day"] < now.date()) & c["net"].notna()]
                for st, g in c.groupby("setup"):
                    market.hstats[(sym, st)] = (len(g), int(g["x1"].notna().sum()),
                                                int((g["why"] == "STOP").sum()), float(g["net"].sum()))
        h = market.hstats.get((r.sym, r.setup))
        if not h:
            return f"\\n📜 No {SETUP_NAME[r.setup]} signals on {r.sym} in the last {HISTORY_DAYS} days."
        return (f"\\n📜 Last {HISTORY_DAYS} days, {SETUP_NAME[r.setup]} on {r.sym}: fired {h[0]}x, "
                f"T1 hit {h[1]}, SL hit {h[2]}, Rs {h[3]:+,.0f} (every signal, same exits)")
    except Exception as e:
        print("history failed:", type(e).__name__, e)
        return ""


def results_today(today):
    """Shares announcing results today (NSE event calendar). None if NSE can't be reached."""
    try:
        import requests
        h = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://www.nseindia.com/"}
        ss = requests.Session()
        ss.get("https://www.nseindia.com", headers=h, timeout=15)
        out = set()
        for e in ss.get("https://www.nseindia.com/api/event-calendar", headers=h, timeout=15).json():
            txt = (str(e.get("purpose", "")) + " " + str(e.get("bm_desc", ""))).lower()
            if "result" in txt and datetime.strptime(str(e.get("date")), "%d-%b-%Y").date() == today:
                out.add(str(e.get("symbol")))
        return out
    except Exception as e:
        print("results calendar failed:", type(e).__name__, e)
        return None


def check(market, model, state, now):''')
rep('send(entry_msg(r, model) if kind == "ENTRY" else',
    'send(entry_msg(r, model) + history_line(market, model, now, r) if kind == "ENTRY" else')

# --- skip shares that announce results today ---
rep('''        send(plan_msg(model, now, now.time() > dtime(9, 35)))''', '''        res = results_today(now.date())
        note = ("\\n⚠️ Results calendar (NSE) not reachable today: results days are not filtered." if res is None else
                "\\n📅 Results today, skipped: " + ", ".join(sorted(set(model["shares"]) & res)) if set(model["shares"]) & res else
                "\\n📅 No results today among the watched shares.")
        send(plan_msg(model, now, now.time() > dtime(9, 35)) + note)''')
rep('''    market = Market(E.login(), model["shares"], now.date())''', '''    res = results_today(now.date()) or set()
    market = Market(E.login(), [x for x in model["shares"] if x not in res], now.date())''')

ast.parse(s)
open("ml.py", "w").write(s)
print("patch 4 applied")
