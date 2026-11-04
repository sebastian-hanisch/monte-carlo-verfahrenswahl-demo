"""Randfälle: kleinste Budgets, falsche Eingaben, jede Informationslage, Zahlenformate."""

import math

import numpy as np

import pytest

import mcw_algorithm as A
import mcw_constants as C
import mcw_evaluation as ev
import mcw_scenario as S


def test_importance_sampling_nur_fuer_netz_aufgaben():
    with pytest.raises(ValueError, match="Netz"):
        A.estimate(A.make_task("glatt12"), "is", 256, 1)


def test_sobol_bloecke_brauchen_n_durch_acht():
    t = A.make_task("glatt12")
    with pytest.raises(ValueError, match="durch 8"):
        A.estimate(t, "qmc8", 100, 1)
    assert A.estimate(t, "qmc8", 64, 1) > 0


def test_unbekannte_aufgabe_verfahren_methode():
    with pytest.raises(ValueError):
        A.make_task("gibt_es_nicht")
    with pytest.raises(ValueError):
        A.estimate(A.make_task("glatt12"), "xyz", 64, 1)
    with pytest.raises(ValueError):
        A.points("xyz", 8, 2, 1)


def test_sobol_dimension_ausserhalb_der_eingebetteten_richtungszahlen():
    with pytest.raises(ValueError):
        A.sobol_ints(8, C.MAX_DIM + 1)
    with pytest.raises(ValueError):
        A.sobol_ints(0, 3)


def test_kleinstes_budget_rechnet_alle_verfahren():
    for k in C.TASK_KEYS:
        t = A.make_task(k)
        for m in A.methods_for(t):
            e = A.estimate(t, m, 256, 3)
            assert math.isfinite(e), (k, m)


def test_probelauf_ohne_treffer_hat_varianz_null_und_wird_nicht_zu_nan():
    p = A.pilot(A.make_task("netz33_p99"), 200, 5)
    assert p["hits"] == 0 and p["v"] == 0 and p["rho"] == 0.0 and p["pred"]["qmc"] == 1.0 and p["blind"]
    for v in p["pred"].values():
        assert math.isfinite(v)


def test_probelauf_konstante_aufgabe():
    t = A.Task("konst", "konstant", 2, "glatt", 1.0, lambda u: np.ones(len(u)), lambda u: u[:, 0], 0.5)
    p = A.pilot(t, 100, 1)
    assert p["pred"]["cv"] == 1.0 and p["pred"]["lhs"] == 1.0 and p["pred"]["qmc"] == 1.0


def test_jede_informationslage_endet_an_der_richtigen_frage():
    erwartet = {"ziehen": 0, "daten": 1, "anhalten": 2, "gradient": 3, "einfluss": 3, "geschachtelt": 4, "zeit": 4, "modelle": 4}
    for info, nr in erwartet.items():
        steps = S.tree_steps(info)
        assert steps[-1]["nr"] == nr and steps[-1]["antwort"] == "Ja" and len(steps) == nr + 1
        assert all(s["antwort"] == "Nein" for s in steps[:-1]) and steps[-1]["verweis"] and "Stück" in steps[-1]["zahl_aus"] + steps[-1]["verweis"] or info == "ziehen"
    alle_nein = S.tree_steps("schaetzer")
    assert len(alle_nein) == 5 and all(s["antwort"] == "Nein" for s in alle_nein)
    with pytest.raises(ValueError):
        S.tree_steps("unbekannt")


def test_weg_durch_den_baum_bei_nur_dem_schaetzer_hat_sieben_stufen_sonst_weniger():
    s = ev.Settings(task="glatt12")
    a = ev.analyse(s, 12)
    assert [x["nr"] for x in ev.tree_path(s, a)] == [0, 1, 2, 3, 4, 5, 6]
    s2 = ev.Settings(task="glatt12", info="anhalten")
    assert [x["nr"] for x in ev.tree_path(s2, ev.analyse(s2, 12))] == [0, 1, 2]


def test_ausfallwahrscheinlichkeit_der_netze():
    assert S.net_pi("3x3", 0.9) == pytest.approx(0.0530152, abs=5e-8)
    assert S.net_pi("3x4", 0.95) == pytest.approx(0.0120679, abs=5e-8)


def test_zahlenformate():
    assert ev.de(0.12345, 3) == "0,123" and ev.zahl(1024) == "1.024" and ev.pr(1.0) == "1,00" and ev.pr(0.0) == "0,000"
    assert ev.fac(2.169) == "2,17" and ev.fac(0.2310) == "0,231" and ev.fac(0.00187) == "0,0019" and ev.fac(float("nan")) == "–"
    assert ev.pi_text(0.01184) == "0,0118" and ev.pi_text(0.1497) == "0,150" and ev.n1(54.29) == "54,3"


def test_budgets_sind_zweierpotenzen_durch_acht_teilbar():
    for N in C.N_CHOICES:
        assert N % 8 == 0 and (N & (N - 1)) == 0


def test_auswertung_ohne_ereignisse_im_probelauf_rechnet():
    a = ev.analyse(ev.Settings(task="netz33_p99", n0=200), 12)
    assert a.rec == "is" and a.pilot["blind"]
    kind, text = ev.verdict_main(a, ev.Settings(task="netz33_p99", n0=200))
    assert kind in ("success", "info", "warning") and "Importance Sampling" in text
