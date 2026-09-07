import sys
import _repro as R
import event_chain as EC

for s in (0.0, 0.5, 1.0):
    hg, ag, hrc, arc, f, y, r = R.run_one_instr(s, 9000 + hash(s))
    print(f"s={s}: {hg}-{ag} | fouls={f} yellow={y} red={r} (state reds h={hrc} a={arc})")
