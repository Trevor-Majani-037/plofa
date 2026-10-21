"""WHERE DOES THE WORK ACTUALLY LIVE?  (the push state)

`git commit` does not reach GitHub.  This repo sat with SEVEN finished
commits unpushed for days — six of them from earlier sessions — and the
only way it was noticed was by refreshing the browser and finding a
five-day-old HEAD.  So: after every commit in this project, run this.

It answers three questions in one screen:
  1. is the branch ahead of the remote, and by how much;
  2. WHICH commits are waiting, and how old are they;
  3. what is still uncommitted in the working tree — because a commit is
     not the same thing as a push, and neither is the same thing as
     somebody else's work in flight.

Read-only: it runs `git` queries and never writes.

Run:  .venv\\Scripts\\python.exe -u _diag_push_state.py
"""
from __future__ import annotations

import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return (r.stdout or "").strip()


def main() -> int:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name",
                   "@{u}")
    if not upstream:
        # No tracking branch set.  `git push origin main` works fine without
        # one, but then nothing tells you that the branch has drifted from
        # the remote — which is exactly how seven commits sat unpushed for
        # days.  Fall back to `origin/<branch>` and say so.
        candidate = f"origin/{branch}"
        if git("rev-parse", "--verify", "--quiet", candidate):
            upstream = candidate
            print(f"NOTE: no upstream tracking branch is set for {branch!r}.")
            print(f"      Comparing against {candidate} directly.  Fix it with:")
            print(f"      git branch --set-upstream-to={candidate} {branch}")
        else:
            upstream = f"origin/{branch} (not found)"

    print("=" * 72)
    print(f"BRANCH   {branch}   ->   {upstream}")
    print("=" * 72)

    remote_ref = upstream.split()[0]
    counts = git("rev-list", "--left-right", "--count", f"{remote_ref}...HEAD")
    if counts:
        parts = counts.split()
        if len(parts) == 2:
            behind, ahead = int(parts[0]), int(parts[1])
            print(f"unpushed commits: {ahead}      behind the remote: {behind}")
            if ahead:
                print("\n--- WAITING TO BE PUSHED ---")
                for line in git("log", "--format=%h  %ad  %s", "--date=short",
                                f"-{ahead}", f"{remote_ref}..HEAD").splitlines():
                    print(f"  {line}")
                print("\n  push with:  git push origin " + branch)
            else:
                print("\n  the branch is IN SYNC with the remote.")
        else:
            print(f"  (could not read the divergence: {counts!r})")
    else:
        print("  (no remote reference to compare against)")

    # ── the working tree ────────────────────────────────────────────────
    st = git("status", "--porcelain")
    lines = [ln for ln in st.splitlines() if ln.strip()]
    if lines:
        modified = [ln for ln in lines if ln[:2].strip() == "M"]
        untracked = [ln for ln in lines if ln.startswith("??")]
        deleted = [ln for ln in lines if ln[:2].strip() == "D"]
        print("\n--- UNCOMMITTED IN THE WORKING TREE ---")
        print(f"  modified: {len(modified)}   untracked: {len(untracked)}"
              f"   deleted: {len(deleted)}")
        print("  (a commit is not a push, and none of this is on GitHub yet)")
        for ln in lines[:14]:
            print(f"    {ln}")
        if len(lines) > 14:
            print(f"    ... and {len(lines) - 14} more")
    else:
        print("\n--- the working tree is clean ---")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
