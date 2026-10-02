# -*- coding: utf-8 -*-
"""Step-9 sidecar: byte-truth of the matchday-4 export tail that raised
OSError 22 on THIS machine. No PIL import path guessing, no savefig
guesswork -- the probe reproduces the exact open modes the failing tail
uses and reports which one breaks, so the fix lands on the real cause
(after 20+ chaotic channels I stop trusting any of them as arbiters;
the write-tool pair + interpreter run is the only byte-true channel).
Removed immediately after the run (sidecar discipline)."""

import os

TARGET = r"C:\Users\Trevor Majani\Downloads\plofa_checkpoint6\plofa\plofa_output\Hartwell_City_vs_Thornfield_United_MD01\Hartwell_City_vs_Thornfield_United_MD1_pass_network.png"


def report(label, value):
    print("PROBE", label, "->", value)


report("target exists", os.path.exists(TARGET))
report("target is file", os.path.isfile(TARGET))
if os.path.isfile(TARGET):
    report("target size bytes", os.path.getsize(TARGET))

# 1. the exact PIL-style write: open(path, "w+b")
try:
    with open(TARGET, "w+b"):
        report("PIL-style w+b open (exact tail mode)", "OK")
except OSError as exc:
    report("PIL-style w+b open", "OSError errno={0} str={1}".format(exc.errno, exc.strerror))

# 2. plain "w" overwrite of the same existing file
try:
    with open(TARGET, "w"):
        report("plain 'w' overwrite", "OK")
except OSError as exc:
    report("plain 'w' overwrite", "OSError errno={0} str={1}".format(exc.errno, exc.strerror))

# 3. bare matplotlib savefig over the SAME target (the true tail)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1])
    fig.savefig(TARGET, dpi=150, bbox_inches="tight", facecolor="#0d1220")
    plt.close(fig)
    report("matplotlib savefig overwrite", "OK")
except OSError as exc:
    report("matplotlib savefig overwrite", "OSError errno={0} str={1}".format(exc.errno, exc.strerror))

# 4. coexistence test: hold a read handle open, then try a w+b handle
try:
    with open(TARGET, "rb") as held:
        try:
            with open(TARGET, "r+b"):
                report("read-held + r+b coexisting", "OK")
        except OSError as exc:
            report("read-held + r+b coexisting", "OSError errno={0} str={1}".format(exc.errno, exc.strerror))
except OSError as exc:
    report("open rb held", "OSError errno={0} str={1}".format(exc.errno, exc.strerror))
