# patch 3: walk-forward testing - the year is cut into ~monthly blocks, each block is
# traded with a model trained only on OLDER days; a plan must win across the blocks
import ast

s = open("ml.py").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, "not found exactly once: " + old[:60]
    s = s.replace(old, new)


rep("MIN_CONF = 0.75", "WF_BLOCK = 20                  # walk-forward block = ~1 month of trading days\nMIN_CONF = 0.75")

rep("def score(df, idx, ndays):", '''def tune_thr(va, nd, ml):
    best = None
    for thr in np.arange(0.30, 0.801, 0.025):
        sc = score(va, policy(va, thr, ml), nd)
        if sc["per_day"] >= MIN_CALLS_PER_DAY and (best is None or sc["net"] > best[1]):
            best = (round(float(thr), 3), sc["net"])
    return 0.5 if best is None else best[0]


def walk_forward(allc, days, cut1):
    dl = list(days)
    wd = [d for d in dl if d >= cut1]
    blocks = [wd[i:i + WF_BLOCK] for i in range(0, len(wd), WF_BLOCK)]
    blocks = [b for b in blocks if len(b) >= WF_BLOCK // 2]
    ml = LOSS_LIMITS[0]
    wf = {}
    for plan in PLANS:
        P = allc[allc["plan"] == plan]
        res, thrs = [], []
        for b in blocks:
            i0 = dl.index(b[0])
            vd = dl[max(0, i0 - WF_BLOCK):i0]
            m = fit_models(P[P["day"] < vd[0]])              # only older days
            va = P[P["day"].isin(vd)].copy()
            te = P[P["day"].isin(b)].copy()
            va["p"] = predict(m, va)
            te["p"] = predict(m, te)
            thr = tune_thr(va, len(vd), ml)                   # cut-off from the month before
            res.append(score(te, policy(te, thr, ml), len(b)))
            thrs.append(thr)
        n = sum(r["n"] for r in res)
        wins = sum(r["acc"] * r["n"] / 100 for r in res)
        wf[plan] = dict(n=int(n), net=float(sum(r["net"] for r in res)),
                        acc=round(float(wins / n * 100), 1) if n else 0.0,
                        good=int(sum(1 for r in res if r["net"] > 0)), blocks=res,
                        thr=float(np.median(thrs)) if thrs else 0.5)
    return wf, blocks


def score(df, idx, ndays):''')

rep('''    best = max(combos, key=lambda c: (c["validate"]["acc"], c["validate"]["net"]))  # most accurate on tuning
    status = "tested" if (best["test"]["net"] > 0 and best["validate"]["net"] > 0
                          and best["test"]["n"] >= MIN_TEST_TRADES) else "paper"''',
    '''    wf, blocks = walk_forward(allc, days, cut1)
    ok = [p for p in PLANS if wf[p]["net"] > 0] or list(PLANS)   # profitable plans first
    pick_plan = max(ok, key=lambda p: (wf[p]["acc"], wf[p]["net"]))  # then the most accurate
    best = [c for c in combos if c["plan"] == pick_plan][0]
    best = dict(best, threshold=wf[pick_plan]["thr"])
    w = wf[pick_plan]
    status = "tested" if (w["net"] > 0 and w["good"] * 10 >= 6 * len(blocks)
                          and w["n"] >= MIN_TEST_TRADES) else "paper"''')
rep("models=models, max_trades=MAX_TRADES_PER_DAY, backup_p=BACKUP_P)",
    "models=models, max_trades=MAX_TRADES_PER_DAY, backup_p=BACKUP_P, wf=wf)")
rep('''        mark = "👉 " if c is best else "   "''', '''        mark = "👉 " if c["plan"] == best["plan"] else "   "''')
rep('f"👉 = most accurate on the tuning period: {desc(best)}."',
    'f"👉 = chosen by walk-forward (profitable first, then most accurate): {desc(best)}."')
rep('"✅ Status: LIVE calls - passed the unseen test."',
    '"✅ Status: LIVE calls - passed walk-forward (profit, and won in 60%+ of months)."')
rep('"📝 Status: PAPER only - did not pass the unseen test. Watch, don\'t use real money."',
    '"📝 Status: PAPER only - did not pass walk-forward. Watch, don\'t use real money."')
rep('''    send("\\n".join(lines))


# --------------------------------- live''', '''    send("\\n".join(lines))
    wl = [f"🚶 Walk-forward: {len(blocks)} blocks of ~{WF_BLOCK} trading days "
          f"({blocks[0][0]:%d %b} - {blocks[-1][-1]:%d %b}). Each block is traded by a model "
          f"trained only on older days, cut-off set on the month before.", ""]
    for p in PLANS:
        r = wf[p]
        mark = "👉 " if p == best["plan"] else "   "
        wl.append(f"{mark}Plan {p}: {r['n']} calls, profit {r['acc']:.0f}%, Rs {r['net']:+,.0f}, "
                  f"winning months {r['good']}/{len(blocks)}")
        wl.append("      " + " | ".join(f"{b['net']:+,.0f}" for b in r["blocks"]))
    wl += ["", f"Live cut-off for plan {best['plan']}: {best['threshold'] * 100:.0f}% "
               f"(median of the monthly cut-offs; live calls still need 75%+)."]
    send("\\n".join(wl))


# --------------------------------- live''')

ast.parse(s)
open("ml.py", "w").write(s)
print("patch 3 applied")
