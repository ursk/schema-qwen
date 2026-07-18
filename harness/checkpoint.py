"""Named, resumable run checkpoints.

A run's entire durable state is its run dir: timeline.jsonl (ground truth),
notes.md + world_model.py (+ world_model_best.py) — Schema's "the agent's
weights" — plus events.jsonl/trace.log for provenance. The live game holds no
extra state we rely on: every resume starts from RESET (a legal action) and
the timeline's segment logic handles it. So checkpoint = snapshot of the run
dir, and ANY model can continue from ANY player's checkpoint.

  python -m harness.checkpoint save runs/ls20-gemma1 gemma-moe-hud
  python -m harness.checkpoint list
  python -m harness.run --game ls20 --model gemma31 --from-ckpt gemma-moe-hud --run-name dense1
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"

FILES = ["timeline.jsonl", "notes.md", "world_model.py", "world_model_best.py",
         "events.jsonl", "trace.log"]


def save(run_dir, name):
    src = Path(run_dir)
    dst = CKPT_DIR / name
    if dst.exists():
        sys.exit(f"checkpoint {name!r} already exists — pick a new name")
    dst.mkdir(parents=True)
    copied = []
    for f in FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
            copied.append(f)
    (dst / "ORIGIN").write_text(f"{src}\n")
    print(f"checkpoint {name!r} saved ({', '.join(copied)})")


def seed(name, run_dir):
    """Populate a fresh run dir from a checkpoint (used by run.py --from-ckpt)."""
    src = CKPT_DIR / name
    if not src.exists():
        sys.exit(f"no checkpoint named {name!r} — see: python -m harness.checkpoint list")
    dst = Path(run_dir)
    if (dst / "timeline.jsonl").exists():
        sys.exit(f"{dst} already has a timeline — refusing to overwrite; use a fresh --run-name")
    dst.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    print(f"run dir {dst} seeded from checkpoint {name!r}")


def list_ckpts():
    if not CKPT_DIR.exists():
        print("(no checkpoints)")
        return
    for d in sorted(CKPT_DIR.iterdir()):
        if d.is_dir():
            origin = (d / "ORIGIN").read_text().strip() if (d / "ORIGIN").exists() else "?"
            n_events = sum(1 for _ in open(d / "timeline.jsonl")) if (d / "timeline.jsonl").exists() else 0
            print(f"{d.name:30s} {n_events:5d} timeline events   from {origin}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "save":
        save(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        list_ckpts()
    else:
        sys.exit(__doc__)
