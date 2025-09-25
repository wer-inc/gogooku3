import builtins
import types
from pathlib import Path

import pytest


class _FakeProc:
    def __init__(self, lines: list[str], returncode: int = 0):
        self._lines = lines
        self.returncode = returncode
        self.stdout = self._iter_lines()

    def _iter_lines(self):
        for ln in self._lines:
            yield ln

    def wait(self):
        return self.returncode


@pytest.mark.unit
def test_hydra_passthrough_filters_unknown_flags(monkeypatch, tmp_path):
    # Import by file path (scripts/ is not a package)
    import importlib.util
    import sys
    mod_path = Path("scripts/integrated_ml_training_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("integrated_ml_training_pipeline", mod_path)
    assert spec and spec.loader
    pip = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pip
    spec.loader.exec_module(pip)  # type: ignore[arg-type]

    # Arrange: craft extra overrides including unsupported flags and positionals
    extra = [
        "--output-base",
        str(tmp_path / "runs"),
        "--run-hpo",
        "1",
        "train.trainer.max_epochs=3",
        "+job=cpu",
        "--config-name",
        "config_production.yaml",
        "random_positional_token",
        "data.source.data_dir=/data",
        "--multirun",
        "--unknown-flag",
        "value",
        "target=case",
    ]

    # Minimal training_data_info stub
    tinfo = {
        "train_files": ["dummy"],
        "val_files": [],
        "test_files": [],
        "data_dir": str(tmp_path),
        "sequence_length": 20,
        "input_dim": 8,
    }

    # Capture the command that would have been executed by mocking Popen
    captured = {}

    def _fake_popen(cmd, stdout=None, stderr=None, text=None, bufsize=None, env=None, universal_newlines=None):
        captured["cmd"] = list(cmd)
        # Yield a couple of benign lines and exit 0
        return _FakeProc(["training start\n", "training done\n"], returncode=0)

    monkeypatch.setattr(pip.subprocess, "Popen", _fake_popen)

    pl = pip.CompleteATFTTrainingPipeline(extra_overrides=extra)
    # Force small fast path but ensure command gets built (epochs > 0)
    pl.atft_settings["max_epochs"] = 1

    import asyncio
    ok, info = asyncio.get_event_loop().run_until_complete(
        pl._execute_atft_training_with_results(tinfo)
    )
    assert ok, info

    # Assert only Hydra-friendly overrides passed through
    cmd = captured.get("cmd", [])
    # Must include these overrides
    expected_present = {
        "train.trainer.max_epochs=3",
        "+job=cpu",
        "--config-name",
        "config_production.yaml",
        "data.source.data_dir=/data",
        "--multirun",
        "target=case",
    }
    for token in expected_present:
        assert token in cmd, f"missing expected token: {token} in {cmd}"

    # Must exclude unsupported flags and their values
    forbidden = {"--output-base", str(tmp_path / "runs"), "--run-hpo", "1", "--unknown-flag", "value", "random_positional_token"}
    for token in forbidden:
        assert token not in cmd, f"unexpected token passed through: {token} in {cmd}"
