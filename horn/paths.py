"""Where things live, in one place.

Now that the package can be installed with `pip install -e .`, scripts and notebooks
can be launched from any working directory.

"""

from pathlib import Path

# Anchor on THIS file's location, not the working directory. __file__ is horn/paths.py,
# so its parent is horn/ and the grandparent is the repo root. This is what makes
# `plt.savefig(results("x.png"))` land in the same place whatever directory the script
# or notebook was launched from.
PACKAGE_DIR = Path(__file__).resolve().parent   # .../horn
REPO = PACKAGE_DIR.parent                       # .../HORN-JAX.dev

# If the package was installed NON-editably it lives in site-packages, whose
# parent is not a useful anchor for data or figures. Fall back to the working
# directory in that case - wrong location beats writing into site-packages.
if not (REPO / "pyproject.toml").exists():          
    REPO = Path.cwd().resolve()

DATA_DIR = REPO / "data"          # dataset cache; gitignored
RESULTS_DIR = REPO / "results"    # figures and run records; committed on purpose


def data(name: str = "") -> Path:
    """Path inside the data cache, creating the directory on demand."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name if name else DATA_DIR


def results(name: str = "") -> Path:
    """Path inside results/, creating the directory on demand.

        plt.savefig(results("demo.png"), dpi=130)

    Directories are created here rather than at import time, so importing the
    module has no side effects on disk.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name if name else RESULTS_DIR
