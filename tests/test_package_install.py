import importlib.util
import subprocess
import sys


def test_package_includes_models(tmp_path):
    install_dir = tmp_path / "install"
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
        ]
    )
    sys.path.insert(0, str(install_dir))
    try:
        assert importlib.util.find_spec("xtylearner.models") is not None
    finally:
        sys.path.pop(0)


def test_package_includes_default_config(tmp_path):
    install_dir = tmp_path / "install"
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
        ]
    )
    cfg_path = install_dir / "xtylearner" / "configs" / "default.yaml"
    assert cfg_path.is_file()
