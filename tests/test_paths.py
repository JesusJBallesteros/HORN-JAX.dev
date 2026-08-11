"""Tests for repo-anchored paths.

The bug these guard against: with the package installed editable, scripts can be
run from any directory, so a relative path like "demo.png" lands wherever the
shell happened to be. Figures scatter, and the copy in the repo silently stops
being updated.
"""

import os
from pathlib import Path

from horn import paths


def test_repo_is_the_package_parent():
    """REPO must be the directory containing pyproject.toml, not the cwd."""
    assert (paths.REPO / "pyproject.toml").exists()
    assert (paths.REPO / "horn" / "core.py").exists()
    assert paths.PACKAGE_DIR == paths.REPO / "horn"


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    """The whole point: resolution is anchored to the package, not the shell."""
    before = (paths.REPO, paths.DATA_DIR, paths.RESULTS_DIR)
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(paths)
    assert (paths.REPO, paths.DATA_DIR, paths.RESULTS_DIR) == before
    assert not (tmp_path / "results").exists(), "wrote into the working directory"


def test_helpers_create_on_demand_and_return_child():
    """results('x.png') gives <repo>/results/x.png and makes the directory."""
    p = paths.results("demo.png")
    assert p == paths.RESULTS_DIR / "demo.png"
    assert p.parent.is_dir()
    assert paths.results() == paths.RESULTS_DIR
    assert paths.data("mnist.npz") == paths.DATA_DIR / "mnist.npz"


def test_import_has_no_disk_side_effects(tmp_path, monkeypatch):
    """Importing must not create directories - only calling the helpers may."""
    monkeypatch.chdir(tmp_path)
    import importlib
    importlib.reload(paths)
    assert os.listdir(tmp_path) == []
