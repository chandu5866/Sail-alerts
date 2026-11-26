# patch 2: pick the exit plan that was MOST ACCURATE on the tuning period
import ast

s = open("ml.py").read()
old = '    best = max(combos, key=lambda c: c["validate"]["net"])  # chosen on tuning only'
new = '    best = max(combos, key=lambda c: (c["validate"]["acc"], c["validate"]["net"]))  # most accurate on tuning'
assert s.count(old) == 1, "selection line not found"
s = s.replace(old, new)
s = s.replace('f"👉 = chosen on the tuning period: {desc(best)}."', 'f"👉 = most accurate on the tuning period: {desc(best)}."')
ast.parse(s)
open("ml.py", "w").write(s)
print("patch 2 applied")
