"""Saving what an experiment printed, without having to remember to redirect it.

The results/ directory currently contains .txt files captured by hand with a
shell redirect. That works until the run you forget to redirect, which is always
the run worth keeping. `Report` mirrors everything printed into a file, so the
console transcript and the committed record cannot diverge.

    with Report("probe_mechanism_recflat") as rep:
        rep.print("...")            # goes to stdout AND results/<name>.txt
        rep.save_json({...})        # results/<name>.json
        rep.save_fig(fig)           # results/<name>.png

Every artefact from one run shares a stem, so a directory listing groups them and
the name says which condition produced them.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from horn.paths import results


def _git_commit() -> str:
    """Short commit hash, so a figure can be traced back to the code that made it."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent.parent)
        return out.stdout.strip() or "unknown"
    except Exception:                                  # noqa: BLE001
        return "unknown"


class Report:
    """Tee stdout to results/<name>.txt and collect sibling artefacts."""

    def __init__(self, name: str, header: str = ""):
        self.name = name
        self.path = results(f"{name}.txt")
        self._lines: list[str] = []
        self.artefacts: list[Path] = [self.path]

        # A provenance header on every table. Six months from now the question is
        # always "which version of the code produced this number".
        self._lines.append(f"# {name}")
        if header:
            self._lines.append(f"# {header}")
        self._lines.append(f"# generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
                           f"  commit {_git_commit()}"
                           f"  python {platform.python_version()}")
        self._lines.append("")

    def print(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        self._lines.append(text)

    def save_json(self, obj) -> Path:
        path = results(f"{self.name}.json")
        path.write_text(json.dumps(obj, indent=2, default=str))
        self.artefacts.append(path)
        return path

    def save_fig(self, fig, suffix: str = "") -> Path:
        stem = f"{self.name}{'_' + suffix if suffix else ''}"
        path = results(f"{stem}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        self.artefacts.append(path)
        return path

    def flush(self):
        self.path.write_text("\n".join(self._lines) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # Write even on failure: a partial transcript of a crashed run is often
        # exactly what you need to see.
        if exc is not None:
            self._lines.append(f"\n!! run failed: {exc_type.__name__}: {exc}")
        self.flush()
        print("\nwrote:", file=sys.stderr)
        for p in self.artefacts:
            print(f"  {p}", file=sys.stderr)
        return False
