"""Gemeinsame Fixtures. Die Oberflächentests (AppTest) laufen mit wenigen Wiederholungen je Messung, damit sie schnell bleiben; die Claims rechnen mit den echten Werten."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mcw_constants as C  # noqa: E402

_KLEIN = ("test_app", "test_layout", "test_footer", "test_app_experiments", "test_presets")


@pytest.fixture(autouse=True)
def wenige_wiederholungen(request, monkeypatch):
    if Path(str(request.node.fspath)).stem in _KLEIN:
        monkeypatch.setattr(C, "R", 12)
        monkeypatch.setattr(C, "R_PILOTS", 12)
