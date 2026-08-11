"""Where things live, in one place.

<<<<<<< HEAD
Now that the package can be installed with `pip install -e .`, scripts and notebooks
can be launched from any working directory.

=======
Now that the package is installed with `pip install -e .`, scripts and notebooks
can be launched from any working directory. That is the point - but it means a
bare `plt.savefig("demo.png")` writes wherever you happened to be standing, so
running demo.py from your home directory scatters figures across the filesystem
and silently fails to update the one in the repo.

Anchoring to the package location instead of the working directory fixes that:
`horn/paths.py` knows where it is, therefore it knows where the repo is.
>>>>>>> e3b3bd2 (Fixed staled data error and documented)
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO = PACKAGE_DIR.parent

# If the package was installed NON-editably it lives in site-packages, whose
# parent is not a useful anchor for data or figures. Fall back to the working
# directory in that case - wrong location beats writing into site-packages.
<<<<<<< HEAD
if not (REPO / "pyproject.toml").exists():          
=======
if not (REPO / "pyproject.toml").exists():          # pragma: no cover
>>>>>>> e3b3bd2 (Fixed staled data error and documented)
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
