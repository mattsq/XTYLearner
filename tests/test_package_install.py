import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.xdist_group("pkg")
def test_package_includes_models(tmp_path):
    install_dir = tmp_path / "install"
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path / "tmp")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            ".",
            "-t",
            str(install_dir),
        ],
        env=env,
    )
    sys.path.insert(0, str(install_dir))
    try:
        assert importlib.util.find_spec("xtylearner.models") is not None
    finally:
        sys.path.pop(0)


@pytest.mark.xdist_group("pkg")
def test_package_includes_default_config(tmp_path):
    install_dir = tmp_path / "install"
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path / "tmp")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            ".",
            "-t",
            str(install_dir),
        ],
        env=env,
    )
    cfg_path = install_dir / "xtylearner" / "configs" / "default.yaml"
    assert cfg_path.is_file()
