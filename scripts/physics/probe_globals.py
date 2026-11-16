"""STEP 2b: which global object actually changes between two matches?

`probe_leak` eliminated the candidates we could name (brain registry, gate
counters, numpy global RNG, match_engine's brain singletons) by resetting them
and watching run 1 stay wrong. So instead of guessing, this walks the
interpreter and asks the machine: **what changed?**

Method
------
1. snapshot every module-level mutable value in the project before match 0;
2. run match 0, snapshot again;
3. run match 1 and confirm it diverges (the bug is present);
4. **restore only the values that step 1→2 changed** and run a third match;
5. if the third match matches match 0, that set of values IS the leak.

Nothing is restored in step 4 that did not change, so the diagnosis cannot be
"we reset the whole world and it worked".

Scope of a snapshot: for every module whose file lives under the repo, every
attribute that is a mutable container (dict / list / set / bytearray), a numpy
array, or a project-defined instance; plus, for project-defined classes, the
mutable entries of their class namespace. Functions, modules and types are
skipped -- they carry no match state.

Run:  $env:PYTHONHASHSEED="0"; python -m scripts.physics.probe_globals
"""
from __future__ import annotations

import copy
import sys

sys.path.insert(0, r"D:\PLOFA\plofa")
sys.path.insert(0, r"D:\PLOFA\plofa\tools\diag")

ROOT = r"D:\PLOFA\plofa"
SEED = 1000

#: Directories that hold no match state (and cost a lot to copy).
_SKIP_DIRS = ("\\.venv", "\\node_modules", "\\brains", "\\manager_brains",
              "\\tests", "\\scripts", "\\__pycache__", "\\.git")

_MUTABLE = (dict, list, set, bytearray)


def _project_module(mod) -> bool:
    f = getattr(mod, "__file__", None)
    if not f:
        return False
    f = f.replace("/", "\\")
    if not f.startswith(ROOT.replace("/", "\\")):
        return False
    return not any(s in f for s in _SKIP_DIRS)


def _copy(value):
    """Deep-copy an attribute, or return a sentinel if it is not copyable."""
    try:
        return copy.deepcopy(value)
    except Exception:
        return _UNCOPYABLE


class _Uncopyable:
    def __repr__(self) -> str:
        return "<uncopyable>"


_UNCOPYABLE = _Uncopyable()


def _interesting(value) -> bool:
    if value is _UNCOPYABLE:
        return False
    if isinstance(value, _MUTABLE):
        return True
    if type(value).__module__.split(".")[0] == "numpy":
        return True
    # A project-defined instance may hold mutable state in its __dict__.
    cls = type(value)
    return cls.__module__ != "builtins" and hasattr(value, "__dict__")


def _capture() -> dict:
    """{('module', attr): value} plus {('class', qualname, attr): value}."""
    snap: dict = {}
    for name, mod in list(sys.modules.items()):
        if not _project_module(mod):
            continue
        for attr, value in list(vars(mod).items()):
            if attr.startswith("__"):
                continue
            value = _copy(value)
            if _interesting(value):
                snap[(name, attr)] = value
        for attr, cls in list(vars(mod).items()):
            if not isinstance(cls, type) or cls.__module__ != name:
                continue
            for cattr, cvalue in list(vars(cls).items()):
                if cattr.startswith("__"):
                    continue
                cvalue = _copy(cvalue)
                if _interesting(cvalue):
                    snap[(name, f"{cls.__name__}.{cattr}")] = cvalue
    return snap


def _same(a, b) -> bool:
    if type(a) is not type(b):
        return False
    mod = type(a).__module__.split(".")[0]
    if mod == "numpy":
        try:
            return a.shape == b.shape and bool((a == b).all())
        except Exception:
            return False
    if isinstance(a, (dict, list, set, bytearray)):
        try:
            return a == b
        except Exception:
            return repr(a) == repr(b)
    if isinstance(a, _Uncopyable):
        return True
    try:
        return a.__dict__ == b.__dict__
    except Exception:
        return repr(a) == repr(b)


def _diff(before: dict, after: dict) -> list:
    out = []
    for key in sorted(set(before) | set(after), key=str):
        if key not in before or key not in after:
            out.append((key, "appeared/disappeared"))
        elif not _same(before[key], after[key]):
            out.append((key, "changed"))
    return out


def _restore(before: dict, keys) -> int:
    """Put the pre-match-0 value back for each changed key."""
    import importlib

    done = 0
    for key, _why in keys:
        if isinstance(key, tuple) and len(key) == 2:
            modname, attr = key
        else:                                            # pragma: no cover
            continue
        if key not in before:
            continue
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        value = copy.deepcopy(before[key])
        if "." in attr and not hasattr(mod, attr):
            cls_name, cattr = attr.split(".", 1)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            setattr(cls, cattr, value)
            done += 1
            continue
        setattr(mod, attr, value)
        done += 1
    return done


def main() -> int:
    import hashlib

    from scripts.physics.probe_determinism import fingerprint

    def sha(fp):
        return hashlib.sha256("\n".join(fp).encode()).hexdigest()[:16]

    # Warm the import graph first: importing a module *is* a state change and
    # would otherwise be reported as the leak.
    fingerprint(SEED)
    importlib_ok = True
    try:
        import importlib  # noqa: F401
    except Exception:                                    # pragma: no cover
        importlib_ok = False

    pre = _capture()
    sha0 = sha(fingerprint(SEED))
    post = _capture()

    sha1 = sha(fingerprint(SEED))
    print(f"match 0  {sha0}")
    print(f"match 1  {sha1}   {'SAME' if sha1 == sha0 else 'DIVERGED (bug present)'}")

    changed = _diff(pre, post)
    print()
    print(f"module-level values changed by one match: {len(changed)}")
    for (modname, attr), why in changed[:80]:
        print(f"   {modname}.{attr:<40} {why}")
    if len(changed) > 80:
        print(f"   ... and {len(changed) - 80} more")

    if sha1 == sha0:
        print()
        print("No divergence in this process -- nothing to diagnose.")
        return 0

    print()
    if not importlib_ok:                                 # pragma: no cover
        print("cannot restore without importlib")
        return 1
    n = _restore(pre, changed)
    sha2 = sha(fingerprint(SEED))
    print(f"match 2 after restoring those {n} values")
    print(f"         {sha2}   {'SAME as match 0 -- LEAK FOUND' if sha2 == sha0 else 'still differs'}")
    return 0 if sha2 == sha0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
