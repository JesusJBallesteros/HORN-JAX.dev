"""MNIST loading.

No torchvision, no tensorflow-datasets. Those pull in a second deep-learning
framework to download four files, and the point of this repo is that the only
framework present is JAX.

WHY THERE IS NO SYNTHETIC FALLBACK
----------------------------------
An earlier version of this loader fell back to random data when the download
failed, printing a warning banner. That is a trap. Warnings scroll off, notebooks
get re-read weeks later, and a plausible-looking accuracy number computed on noise
is worse than a crash - it is a result you might report. If the data cannot be
loaded, this raises.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from horn.paths import DATA_DIR

# torchvision's mirror, then Google's. yann.lecun.com has been unreliable and
# intermittently blocks programmatic access, so it is not used.
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

# Bumped whenever the cache layout changes: key names, dtype, or scaling. An
# earlier loader wrote the same filename with keys train_images/train_labels/...
# and uint8 images, so a cache written by it looks valid and then fails with a
# KeyError deep inside load_mnist. A version stamp turns that into a rebuild and
# one line of explanation.
CACHE_VERSION = 2
_REQUIRED_KEYS = ("xtr", "ytr", "xte", "yte")

FILES = {
    "xtr": "train-images-idx3-ubyte.gz",
    "ytr": "train-labels-idx1-ubyte.gz",
    "xte": "t10k-images-idx3-ubyte.gz",
    "yte": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx(raw: bytes) -> np.ndarray:
    """Parse the IDX binary format used by MNIST.

    The 4-byte header is:  00 00 <dtype> <ndim>
    then `ndim` big-endian uint32 dimension sizes, then the data.

    For MNIST the dtype byte is always 0x08 (unsigned byte). The two leading zero
    bytes are the magic number; validating the dtype byte against zero instead is
    wrong and rejects every valid file. Parsing the header rather than hard-coding
    the offsets (16 for images, 8 for labels) means a truncated or mislabelled
    file fails here rather than silently reshaping into garbage.
    """
    zero, dtype_code, ndim = struct.unpack(">HBB", raw[:4])
    if zero != 0:
        raise ValueError(f"not an IDX file (leading bytes {zero:#06x}, expected 0x0000)")
    if dtype_code != 0x08:
        raise ValueError(f"expected uint8 IDX data (0x08), got {dtype_code:#04x}")
    dims = struct.unpack(f">{ndim}I", raw[4 : 4 + 4 * ndim])
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _cache_is_current(cached: dict) -> bool:
    """Does this loaded .npz match what the current loader expects?

    Checked before use rather than discovered by a KeyError halfway through. The
    version stamp catches same-keys-different-meaning changes too, such as images
    switching from uint8 to float32 - which no key check would notice, and which
    would train silently on data scaled 255x wrong.
    """
    if not all(k in cached for k in _REQUIRED_KEYS):
        return False
    if "_version" not in cached:
        return False
    return int(cached["_version"]) == CACHE_VERSION


def _fetch(name: str) -> bytes:
    errors = []
    for mirror in MIRRORS:
        try:
            print(f"  fetching {name} from {mirror.split('/')[2]} ...", flush=True)
            with urllib.request.urlopen(mirror + name, timeout=60) as response:
                return gzip.decompress(response.read())
        except Exception as exc:                      # noqa: BLE001 - try the next mirror
            errors.append(f"{mirror}: {exc}")
    raise RuntimeError(
        f"could not download {name} from any mirror:\n  " + "\n  ".join(errors)
        + f"\nIf this machine has no network access, fetch the four "
          f"*-ubyte.gz files elsewhere and place mnist.npz in the cache directory."
    )


def load_mnist(cache_dir: str | Path | None = None, scale: str = "unit"):
    """Return (xtr, ytr, xte, yte). Images float32 (N, 28, 28), labels int32.

    scale:
      "unit"        pixels in [0, 1]. Keeps the DC component, which matters for
                    pixel-wise MNIST: most of an image is background, so a
                    zero-centred version and this one drive the oscillators very
                    differently.
      "standard"    zero mean, unit variance over the training set. Removes the DC
                    component. Worth comparing against - the input is the forcing
                    term of a differential equation, so its offset and scale set
                    the operating point of the whole population.

    Downloads on first call and caches everything as a single .npz, so the four
    fetches happen exactly once. `cache_dir` defaults to `<repo>/data`, anchored
    to the package rather than the working directory - otherwise running the same
    script from two places downloads MNIST twice. `data/` is gitignored.
    """
    # Validate before any I/O. A typo here should fail immediately, not after an
    # 11 MB download.
    if scale not in ("unit", "standard"):
        raise ValueError(f"scale must be 'unit' or 'standard', got {scale!r}")

    cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / "mnist.npz"

    out = None
    if npz.exists():
        with np.load(npz) as d:
            cached = {k: d[k] for k in d.files}
        if _cache_is_current(cached):
            out = cached
        else:
            print(f"  cache at {npz} was written by an older loader "
                  f"(keys {sorted(k for k in cached if not k.startswith('_'))}); rebuilding")
            npz.unlink()

    if out is None:
        print(f"MNIST not cached, downloading to {npz} (~11 MB, once):")
        out = {}
        for key, fname in FILES.items():
            arr = _read_idx(_fetch(fname))
            out[key] = (arr.astype(np.float32) / 255.0 if key.startswith("x")
                        else arr.astype(np.int32))
        np.savez_compressed(npz, _version=np.asarray(CACHE_VERSION), **out)
        print(f"  cached to {npz}")

    xtr, ytr, xte, yte = out["xtr"], out["ytr"], out["xte"], out["yte"]

    if scale == "standard":
        # Statistics from the TRAINING set only, applied to both. Using test
        # statistics would be a small but real leak.
        mu, sd = xtr.mean(), xtr.std()
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

    return xtr, ytr, xte, yte
