"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mcw_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_matrix", "Sieger je Budget"),
    ("run_cover", "Intervall zum Niveau 95 %"),
    ("run_cross", "Die Regel N·π < 50 trifft in"),
    ("run_pilots", "Wiederholungen der Wahl"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_kernfrage_steht_ohne_klick_da_und_der_baum_hat_sieben_zeilen():
    at = _app()
    assert "Varianzfaktor gegen einfach (kleiner ist besser)" in _texte(at)
    tree = [d for d in at.dataframe]
    assert tree and len(tree[0].value) == 7


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_cover").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(256).run(timeout=TIMEOUT)
    assert not at.exception
    assert "N = 256" in _texte(at)


def test_kernfrage_wechselt_mit_dem_preset():
    at = _app()
    at.sidebar.button(key="preset_Seltener Ausfall").click().run(timeout=TIMEOUT)
    assert "p = 0,99" in _texte(at) and "Importance Sampling" in _texte(at)
    at.sidebar.button(key="preset_Symmetrie").click().run(timeout=TIMEOUT)
    assert "symmetrisch" in _texte(at)


def test_jede_aufgabe_und_jede_informationslage_laeuft_durch():
    at = _app()
    for task in C.TASK_KEYS:
        at.sidebar.selectbox(key="task_select").set_value(task).run(timeout=TIMEOUT)
        assert not at.exception, (task, [str(e) for e in at.exception])
    for info in C.INFO:
        at.sidebar.selectbox(key="info_select").set_value(info).run(timeout=TIMEOUT)
        assert not at.exception, (info, [str(e) for e in at.exception])


def test_baum_endet_bei_der_gewaehlten_informationslage():
    at = _app()
    at.sidebar.selectbox(key="info_select").set_value("anhalten").run(timeout=TIMEOUT)
    assert len(at.dataframe[0].value) == 3


def test_spielwiese_zeigt_diagnosen_und_vorhersage():
    at = _app()
    assert any("Probelauf n₀" in e.value for e in list(at.info) + list(at.warning))
    assert any("Korrelation" in str(v) for d in at.dataframe for v in d.value.get("Diagnose", []))


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="n_select").set_value(256).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["t"] = "unbekannt"        # unbekannte Aufgabe: verworfen
    at.query_params["n"] = "999999"           # oberhalb des Größten: das Größte
    at.query_params["n0"] = "500"             # zwischen den Auswahlwerten: wird auf einen erlaubten Wert eingerastet
    at.query_params["i"] = "anhalten"
    at.query_params["g"] = "3x4"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.selectbox(key="task_select").value == C.DEFAULT_TASK
    assert at.sidebar.select_slider(key="n_select").value == 4096
    assert at.sidebar.select_slider(key="n0_select").value in C.N0_CHOICES
    assert at.sidebar.selectbox(key="info_select").value == "anhalten"
    assert at.sidebar.selectbox(key="netz_select").value == "3x4"
    assert at.sidebar.number_input(key="seed_input").value == 0
