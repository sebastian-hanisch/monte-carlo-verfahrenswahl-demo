"""Eigenschaften (ORACLE-PLAYBOOK §5, Schicht „Literatur und Eigenschaften“): was bei jeder Wiederholung gelten muss, mit Bändern, die das Rauschen abdecken."""

import math

import numpy as np
import pytest

import mcw_algorithm as A
import mcw_constants as C
import mcw_evaluation as ev

SEED = 40


def test_der_faktor_des_einfachen_verfahrens_ist_immer_eins():
    for k in ("glatt12", "netz33_p90", "schwanz2"):
        assert ev.measure(k, 256, SEED, 40)["mc"]["factor"] == 1.0


def test_antithetisch_verliert_bei_symmetrischer_aufgabe():
    f = ev.measure("symmetrisch12", 1024, SEED, 100)
    assert f["anti"]["factor"] > 1.3                       # Theorie 2; mit 100 Wiederholungen etwa ±15 %
    assert f["qmc"]["factor"] < 0.05                       # Sobol dagegen fast exakt


def test_sobol_gewinnt_bei_glatter_aufgabe_klar_gegen_alle_anderen():
    f = ev.measure("glatt12", 1024, SEED, 100)
    assert f["qmc"]["factor"] < 0.2 and f["qmc"]["factor"] < min(f[m]["factor"] for m in ("anti", "cv", "lhs"))


def test_importance_sampling_gewinnt_bei_seltenem_ausfall_um_zwei_groessenordnungen():
    a = ev.analyse(ev.Settings(task="netz33_p99"), 100)
    assert a.best == "is" and a.factor["is"] < 0.1 and a.factor["qmc8"] > 0.5


def test_trefferregel_stimmt_mit_der_messung_bei_klarem_fall_ueberein():
    a = ev.analyse(ev.Settings(task="netz33_p99"), 60)
    assert a.rec == "is" and a.npi < C.NPI_THRESHOLD
    b = ev.analyse(ev.Settings(task="glatt12"), 60)
    assert b.rec == "qmc8" and math.isnan(b.npi)


def test_ehrlichkeit_naiver_sobol_balken_zu_breit_bloecke_decken():
    cv = ev.cover("glatt12", 1024, SEED, 100)
    assert cv["qmc"]["se_ratio"] > C.WIDE_RATIO and cv["qmc"]["cover"] > 0.97
    assert cv["qmc8"]["cover"] > 0.85 and 0.7 < cv["qmc8"]["se_ratio"] < 1.3


def test_bei_seltenem_ausfall_bricht_jede_fehlerangabe_ausser_importance_sampling():
    cv = ev.cover("netz33_p99", 1024, SEED, 100)
    assert cv["is"]["cover"] > 0.85
    for m in ("mc", "anti", "cv", "lhs", "qmc", "qmc8"):
        assert cv[m]["cover"] < 0.6, m


def test_schwerer_schwanz_bricht_die_einfache_fehlerangabe():
    cv = ev.cover("schwanz2", 1024, SEED, 200)
    assert cv["mc"]["cover"] < 0.92 and cv["qmc8"]["cover"] < 0.9


def test_kippschwelle_importance_sampling_bei_wenigen_treffern_sobol_bei_vielen():
    rows = ev.cross("3x3", 1024, SEED, 60)
    by_p = {r["p"]: r for r in rows}
    assert by_p[0.99]["factor"]["is"] < 0.2 * by_p[0.99]["factor"]["qmc8"]
    assert by_p[0.8]["factor"]["qmc8"] < 1.0 and by_p[0.8]["npi"] > 200
    npis = [r["npi"] for r in rows]
    assert npis == sorted(npis, reverse=True)              # größeres p: seltener
    assert all(r["rule"] == ("is" if r["npi"] < C.NPI_THRESHOLD else "qmc8") for r in rows)


def test_sobol_gewinn_waechst_mit_dem_budget():
    f1 = ev.measure("glatt12", 256, SEED, 60)["qmc"]["factor"]
    f3 = ev.measure("glatt12", 4096, SEED, 60)["qmc"]["factor"]
    assert f3 < f1 / 2


def test_probelauf_bei_seltenem_ausfall_ist_blind_und_waehlt_importance_sampling():
    pl = ev.pilots("netz33_p99", 1024, 1000, SEED, 60, 40)
    assert pl["blind"] == 1.0 and pl["picks"]["is"] == 40 and pl["regret"] == pytest.approx(1.0)


def test_probelauf_kosten_sind_zwei_n0_plus_512():
    p = A.pilot(A.make_task("glatt12"), 200, 3)
    assert p["cost"] == 912 and ev.pilots("glatt12", 1024, 200, SEED, 30, 10)["cost"] == 912


def test_probelauf_sagt_kontrollvariate_und_antithetisch_ungefaehr_voraus():
    meas = ev.measure("engpass33", 1024, SEED, 150)
    p = A.pilot(A.make_task("engpass33"), 1000, 5)
    assert abs(p["pred"]["cv"] - meas["cv"]["factor"]) < 0.25 and abs(p["pred"]["anti"] - meas["anti"]["factor"]) < 0.25


def test_gleiche_einstellung_gleiche_zahlen_anderer_strom_andere_zahlen():
    a = ev.measure("glatt12", 256, 5, 20)
    assert a == ev.measure("glatt12", 256, 5, 20)
    assert a != ev.measure("glatt12", 256, 6, 20)


def test_jede_aufgabe_hat_eine_endliche_wahrheit_und_dimension_im_bereich():
    for k in C.TASK_KEYS:
        t = A.make_task(k)
        assert np.isfinite(t.truth) and 1 <= t.d <= C.MAX_DIM
        y = t.f(A.points("mc", 50, t.d, 1))
        assert y.shape == (50,) and np.all(np.isfinite(y))
