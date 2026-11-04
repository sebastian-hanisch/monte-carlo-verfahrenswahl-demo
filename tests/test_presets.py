"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet (mit wenigen Wiederholungen, siehe conftest). Die inhaltlichen Erwartungen stehen in test_claims.py."""

import pytest

import mcw_constants as C
import mcw_evaluation as ev
import mcw_presets as P

STANDARD = "Standardfall (Voreinstellung)"


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert preset["task"] in C.TASKS and preset["N"] in C.N_CHOICES and preset["n0"] in C.N0_CHOICES and preset["info"] in C.INFO and preset["netz"] in C.NETZE, name
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    for name, preset in C.PRESETS.items():
        if name != STANDARD:
            assert preset != C.PRESETS[STANDARD], name


def test_die_presets_decken_verschiedene_aufgaben_ab():
    assert len({p["task"] for p in C.PRESETS.values()}) == len(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(_settings(name))
    assert a.truth != 0 and set(a.factor) >= {"mc", "anti", "cv", "lhs", "qmc", "qmc8"} and a.factor["mc"] == 1.0 and a.R == C.R


def test_preset_seltener_ausfall_empfiehlt_importance_sampling():
    a = ev.analyse(_settings("Seltener Ausfall"))
    assert a.rec == "is" and "is" in a.factor


def test_preset_ohne_ereignis_empfiehlt_sobol_in_bloecken():
    for name in ("Glatt und hochdimensional", "Symmetrie", "Schwerer Schwanz", "Rau und hochdimensional"):
        assert ev.analyse(_settings(name)).rec == "qmc8", name
