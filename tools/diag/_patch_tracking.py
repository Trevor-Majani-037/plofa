"""Add the `ever_ahead` flag so a support run requires having been up front.

`support` means "leaves the attack to offer a recycling option behind the
ball". That is a claim about a TRANSITION — advanced -> behind — not merely a
position. Gating it on `behind_now and 6m of backward drift` let ordinary
backward drift qualify, and support was 98% of all observed runs: the taxonomy
was one label wearing six hats.

`ever_ahead` is the mirror of the existing `ever_behind` and is set the same
way, so the two together describe a crossing in either direction.
"""
import pathlib

P = pathlib.Path("run_tracking.py")
t = P.read_text(encoding="utf-8")

pairs = [
    # 1. declare the flag alongside _ever_behind
    ("        self._ever_behind: Dict[str, bool] = {}\n",
     "        self._ever_behind: Dict[str, bool] = {}\n"
     "        self._ever_ahead: Dict[str, bool] = {}\n"),
    # 2. reset it with the rest of the segment state
    ("        self._ever_behind[name] = False\n        self._ever_wider[name] = False\n",
     "        self._ever_behind[name] = False\n"
     "        self._ever_ahead[name] = False\n"
     "        self._ever_wider[name] = False\n"),
    # 3. read it next to the others in the classification block
    ("        ever_behind = self._ever_behind.get(name, False)\n",
     "        ever_behind = self._ever_behind.get(name, False)\n"
     "        ever_ahead = self._ever_ahead.get(name, False)\n"),
    # 4. set it on the FIRST sample too, not only inside the later branch
    ("            if rel_x < -self.AHEAD_M:\n"
     "                self._ever_behind[name] = True\n",
     "            if rel_x < -self.AHEAD_M:\n"
     "                self._ever_behind[name] = True\n"
     "            if rel_x > self.AHEAD_M:\n"
     "                self._ever_ahead[name] = True\n"),
]
for old, new in pairs:
    assert t.count(old) == 1, (t.count(old), old[:60])
    t = t.replace(old, new, 1)
P.write_bytes(t.encode("utf-8"))
print("ever_ahead wired: declare, reset, read, first-sample set")
