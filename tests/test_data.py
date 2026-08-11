"""Tests for the MNIST loader.

No network access required. The IDX wire format is constructed from bytes here,
so the parsing path is genuinely exercised rather than mocked past - which is how
a header bug survived in this file once already.
"""

import struct

import numpy as np
import pytest

<<<<<<< HEAD
from horn.data import _read_idx, load_mnist
=======
from horn.data import CACHE_VERSION, _cache_is_current, _read_idx, load_mnist
>>>>>>> e3b3bd2 (Fixed staled data error and documented)


def _idx_blob(arr: np.ndarray, dtype_code: int = 0x08) -> bytes:
    """Serialise a uint8 array in IDX format, exactly as the mirrors serve it."""
    return (struct.pack(">HBB", 0, dtype_code, arr.ndim)
            + struct.pack(f">{arr.ndim}I", *arr.shape)
            + arr.tobytes())


def test_idx_roundtrip():
    """The header is 00 00 <dtype> <ndim>, not <magic> <ndim>.

    The original version validated the dtype byte against zero and so rejected
    every real MNIST file with 'not an IDX file (magic prefix 8)'.
    """
    payload = np.arange(6, dtype=np.uint8).reshape(2, 3)
    np.testing.assert_array_equal(_read_idx(_idx_blob(payload)), payload)

    images = np.arange(2 * 28 * 28, dtype=np.uint8).reshape(2, 28, 28)
    np.testing.assert_array_equal(_read_idx(_idx_blob(images)), images)


def test_idx_rejects_bad_headers():
    """Corruption must raise, not reshape into plausible-looking garbage."""
    payload = np.arange(6, dtype=np.uint8).reshape(2, 3)

    with pytest.raises(ValueError, match="uint8"):
        _read_idx(_idx_blob(payload, dtype_code=0x0D))     # float32 IDX, unsupported

    bad_magic = b"\x01\x00" + _idx_blob(payload)[2:]
    with pytest.raises(ValueError, match="not an IDX file"):
        _read_idx(bad_magic)


<<<<<<< HEAD
=======
def test_cache_validation_rejects_stale_layouts():
    """Regression: a cache from an older loader must be detected, not KeyError'd.

    The first version of this loader wrote the same filename with keys
    train_images/train_labels/... and uint8 images. Loading it with the current
    code raised `KeyError: 'xtr'` from the middle of load_mnist.
    """
    good = {"xtr": 0, "ytr": 0, "xte": 0, "yte": 0, "_version": np.asarray(CACHE_VERSION)}
    assert _cache_is_current(good)

    legacy = {"train_images": 0, "train_labels": 0, "test_images": 0, "test_labels": 0}
    assert not _cache_is_current(legacy), "old key layout accepted"

    unstamped = {"xtr": 0, "ytr": 0, "xte": 0, "yte": 0}
    assert not _cache_is_current(unstamped), "cache with no version stamp accepted"

    # Right keys, wrong meaning - this is what the version stamp exists for.
    wrong_version = dict(good, _version=np.asarray(CACHE_VERSION - 1))
    assert not _cache_is_current(wrong_version), "stale version accepted"


def test_stale_cache_is_deleted_not_used(tmp_path):
    """A cache that fails validation must be removed, so the next call rebuilds.

    No network here: we only check that the bad file is gone and that the failure
    afterwards is the download failing, never a KeyError from a half-read cache.
    """
    npz = tmp_path / "mnist.npz"
    np.savez_compressed(npz, train_images=np.zeros((2, 28, 28), np.uint8),
                        train_labels=np.zeros(2, np.uint8))
    try:
        load_mnist(tmp_path)
    except KeyError:                       # pragma: no cover - the bug being guarded
        raise AssertionError("stale cache was read instead of rebuilt")
    except Exception:
        pass                               # download failure is fine and expected offline
    assert not npz.exists(), "stale cache left in place"


>>>>>>> e3b3bd2 (Fixed staled data error and documented)
def test_load_mnist_reads_cache_and_scales(tmp_path):
    """load_mnist must use the cache without touching the network, and scale correctly.

    A fabricated cache stands in for the real download. If this test ever starts
    making network calls it will fail on an offline machine, which is the point:
    the cached path must be self-sufficient.
    """
    rng = np.random.default_rng(0)
    xtr = rng.random((40, 28, 28), dtype=np.float32)
    xte = rng.random((10, 28, 28), dtype=np.float32)
<<<<<<< HEAD
    np.savez_compressed(tmp_path / "mnist.npz", xtr=xtr,
                        ytr=rng.integers(0, 10, 40).astype(np.int32),
=======
    np.savez_compressed(tmp_path / "mnist.npz", _version=np.asarray(CACHE_VERSION),
                        xtr=xtr, ytr=rng.integers(0, 10, 40).astype(np.int32),
>>>>>>> e3b3bd2 (Fixed staled data error and documented)
                        xte=xte, yte=rng.integers(0, 10, 10).astype(np.int32))

    a_tr, a_ytr, a_te, _ = load_mnist(tmp_path, scale="unit")
    np.testing.assert_allclose(a_tr, xtr)
    assert a_ytr.dtype == np.int32

    b_tr, _, b_te, _ = load_mnist(tmp_path, scale="standard")
    assert abs(float(b_tr.mean())) < 1e-5
    assert float(b_tr.std()) == pytest.approx(1.0, rel=1e-4)

    # Test data must be standardised with TRAIN statistics, not its own - otherwise
    # information leaks from the test set into preprocessing.
    expected_te = (xte - xtr.mean()) / xtr.std()
    np.testing.assert_allclose(b_te, expected_te, rtol=1e-5)


def test_unknown_scale_raises_before_any_io(tmp_path):
    """A bad scale must fail immediately, without downloading or creating a cache."""
    with pytest.raises(ValueError, match="scale"):
        load_mnist(tmp_path / "nonexistent", scale="whatever")
    assert not (tmp_path / "nonexistent").exists(), "validation ran after touching disk"
