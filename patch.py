# patch 1: the study also learns the SL, T1, T2 and how far to trail the SL after T1
import ast

s = open("ml.py").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, "not found exactly once: " + old[:60]
    s = s.replace(old, new)


rep('PLANS = {"A": 0.75, "B": 1.0, "C": 0.5}',
    'PLANS = {"A": 0.75, "B": 1.0, "C": 0.5, "D": 1.0, "E": 1.5, "F": 0.75, "G": 0.75}')
rep('LOSS_LIMITS = [2, 99]', 'LOSS_LIMITS = [2]')
rep('POOL = [', '''# per plan: SL distance x, T2 distance x, and where the SL moves after T1
# (0 = entry, 0.6 = 60% of the way to T1, locking in profit)
EXITS = {"A": (1.0, 1.0, 0.0), "B": (1.0, 1.0, 0.0), "C": (1.0, 1.0, 0.0),
         "D": (1.0, 1.0, 0.6), "E": (1.0, 1.5, 0.6), "F": (0.75, 1.0, 0.5),
         "G": (1.5, 1.0, 0.5)}
POOL = [''')

# simulation
rep('def sim_two_targets(ind, j, side, p, i1, k1):',
    'def sim_two_targets(ind, j, side, p, i1, k1, ex=(1.0, 1.0, 0.0)):')
rep('''               entry * E.MAX_SL_PCT / 100)
    sl = E.tick(entry - side * dist)''', '''               entry * E.MAX_SL_PCT / 100) * ex[0]
    sl = E.tick(entry - side * dist)''')
rep('''    t2 = E.tick(entry + side * p["R"] * dist)
    r = dict(entry=entry, sl=sl, t1=t1, t2=t2, k1=None,''', '''    t2 = E.tick(entry + side * max(p["R"] * ex[1], k1) * dist)
    be = E.tick(entry + ex[2] * (t1 - entry))
    r = dict(entry=entry, sl=sl, t1=t1, t2=t2, be=be, k1=None,''')
rep('''            if back(k, entry):
                r.update(k=k, why="T1BE", x2=entry)''', '''            if back(k, be):
                r.update(k=k, why="T1BE", x2=be)''')
rep('r = sim_two_targets(ind, j, side, p, i1, PLANS[plan])',
    'r = sim_two_targets(ind, j, side, p, i1, PLANS[plan], EXITS[plan])')
rep('entry=r["entry"], sl=r["sl"], t1=r["t1"], t2=r["t2"], why=r["why"],',
    'entry=r["entry"], sl=r["sl"], t1=r["t1"], t2=r["t2"], be=r["be"], why=r["why"],')

# model + report
rep('plan=best["plan"], k1=PLANS[best["plan"]], max_losses=best["max_losses"],',
    'plan=best["plan"], k1=PLANS[best["plan"]], exits=EXITS[best["plan"]], max_losses=best["max_losses"],')
rep('''        return f"T1 {PLANS[c['plan']]:g}x risk, {lim}"''',
    '''        sl, t2, lk = EXITS[c["plan"]]
        return (f"Plan {c['plan']}: SL {sl:g}x, T1 {PLANS[c['plan']]:g}x, T2 {t2:g}x, "
                f"after T1 SL to {'entry' if lk == 0 else f'{lk * 100:.0f}% of T1'}")''')
rep('Learning study v4 (T1 + T2)', 'Learning study v7 (bot picks SL/T1/T2)')

# live messages
rep('''    if r.p < model["threshold"]: icon, tag''', '''    lock = "entry" if r.be == r.entry else fmt(r.be)
    if r.p < model["threshold"]: icon, tag''')
rep('f"SL {fmt(r.sl)}\\nT1 {fmt(r.t1)}  -> book {q1}, move SL to {fmt(r.entry)}\\n"',
    'f"SL {fmt(r.sl)}\\nT1 {fmt(r.t1)}  -> book {q1}, move SL to {lock}\\n"')
rep('f"for the other {q2} to entry {fmt(r.entry)}. Target T2 {fmt(r.t2)}.")',
    'f"for the other {q2} to {fmt(r.be)}. Target T2 {fmt(r.t2)}.")')
rep('return f"↩️ {r.sym}: back to entry {fmt(r.entry)}. The rest closed at break-even."',
    'return f"↩️ {r.sym}: back to {fmt(r.be)} after T1. Exit the rest of the trade."')
rep('''f"Every call: at T1 ({model['k1']:g}x risk) book half and move SL to entry; "''',
    '''f"Every call: at T1 ({model['k1']:g}x risk) book half and move the SL up (plan {model['plan']}); "''')
rep('''f"🔁 DRY RUN v4 ({len(days)} day(s)): T1 {model['k1']:g}x risk (book half, SL to entry), "''',
    '''f"🔁 DRY RUN v7 ({len(days)} day(s)): plan {model['plan']}, T1 {model['k1']:g}x risk (book half, then trail SL), "''')

rep('"T1BE": "🎯 T1, rest at entry"', '"T1BE": "🎯 T1, rest at trailed SL"')

ast.parse(s)
open("ml.py", "w").write(s)
print("patch 1 applied")
