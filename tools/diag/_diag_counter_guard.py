"""Negative control for test_a_counter_carrier_is_not_structurally_barred_from_scoring.

A test that pins a FIX must be shown to go RED when the fix is undone, or it
proves nothing. This strips the two kwargs at the `AttackChain.generate`
boundary (the same descriptor-reading technique as `install_no_thread`) and
re-runs the real test, so the source is never modified and the check is
reversible by construction.
"""
import subprocess
import sys

import event_chain
from event_chain import AttackChain

raw = AttackChain.__dict__["generate"]
inner = raw.__func__


if isinstance(raw, classmethod):
    def stripped(cls, *a, **kw):
        kw.pop("shooter_name", None)
        kw.pop("assister_name", None)
        return inner(cls, *a, **kw)
    AttackChain.generate = classmethod(stripped)
else:
    def stripped(*a, **kw):
        kw.pop("shooter_name", None)
        kw.pop("assister_name", None)
        return inner(*a, **kw)
    AttackChain.generate = staticmethod(stripped)

# Import the test module and run ONLY the counter test, in this process, with
# the names stripped.
import test_chance_truth as t  # noqa: E402

fn = t.test_a_counter_carrier_is_not_structurally_barred_from_scoring
try:
    fn()
except AssertionError as e:
    print("NEGATIVE CONTROL PASSED — the test goes RED without the fix:")
    print(f"   {str(e).splitlines()[0]}")
    print(f"   {str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else ''}")
    sys.exit(0)
except Exception as e:  # noqa: BLE001
    print(f"NEGATIVE CONTROL INCONCLUSIVE — test raised {type(e).__name__}: {e}")
    sys.exit(2)

print("NEGATIVE CONTROL FAILED — the test PASSED with the names stripped,")
print("so it is not actually testing the threading it was written for.")
sys.exit(1)