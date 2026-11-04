"""Kern von Hand (DEMO-PLAYBOOK §1b): jede Regel an kleinen Zahlen, die man im Kopf nachrechnet."""

import math

import numpy as np
import pytest

import mcw_algorithm as A
import mcw_constants as C


def _linear():
    """f(u) = u in einer Dimension, Wahrheit 1/2, Kontrolle u mit bekanntem Mittel 1/2."""
    return A.Task("lin", "linear", 1, "glatt", 0.5, lambda u: u[:, 0], lambda u: u[:, 0], 0.5)


def _sym():
    """f(u) = (u - 1/2)^2, Wahrheit 1/12, symmetrisch unter u -> 1 - u."""
    return A.Task("sym", "symmetrisch", 1, "symmetrisch", 1 / 12, lambda u: (u[:, 0] - 0.5) ** 2, lambda u: u[:, 0], 0.5)


# --- Strom ----------------------------------------------------------------------------------------------------------------------------------

def test_uniforms_gegen_die_python_ganzzahl_referenz():
    seed = A.stream_seed(40, 1, 2)
    ref = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert np.allclose(A.uniforms(seed, 5), ref, rtol=0, atol=0)
    assert np.all((A.uniforms(seed, 1000) >= 0) & (A.uniforms(seed, 1000) < 1))


# --- Sobol, Latin Hypercube -------------------------------------------------------------------------------------------------------------------

def test_sobol_die_ersten_vier_punkte_in_zwei_dimensionen_von_hand():
    x = A.sobol_ints(4, 2).astype(np.float64) / float(1 << A.BITS)
    assert np.array_equal(x, np.array([[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]))


def test_sobol_netz_eigenschaft_ein_punkt_je_intervall():
    N = 64
    x = A.sobol_ints(N, 6).astype(np.float64) / float(1 << A.BITS)
    for j in range(6):
        assert sorted(np.floor(x[:, j] * N).astype(int)) == list(range(N))


def test_owen_scrambling_erhaelt_die_stratifizierung_und_bleibt_in_0_1():
    p = A.points("sobol_owen", 64, 5, 123)
    assert p.min() >= 0 and p.max() < 1
    for j in range(5):
        assert sorted(np.floor(p[:, j] * 64).astype(int)) == list(range(64))


def test_latin_hypercube_ein_punkt_je_stratum_und_dimension():
    N, d = 50, 4
    p = A.points("lhs", N, d, 9)
    for j in range(d):
        assert sorted(np.floor(p[:, j] * N).astype(int)) == list(range(N))


# --- Irwin-Hall, Normalverteilung --------------------------------------------------------------------------------------------------------------

def test_irwin_hall_von_hand():
    assert A.irwin_hall_sf(2, 1.0) == pytest.approx(0.5)                          # Dreiecksverteilung, symmetrisch um 1
    assert A.irwin_hall_sf(3, 1.5) == pytest.approx(0.5)
    assert A.irwin_hall_sf(2, 1.5) == pytest.approx(0.125)                        # P(S > 3/2) = (2 - 3/2)^2 / 2
    assert A.irwin_hall_pos_part(1) == pytest.approx(1 / 8)                       # E[(U - 1/2)+] = ∫_{1/2}^1 (u - 1/2) du
    assert A.irwin_hall_pos_part(2) == pytest.approx(1 / 6)                       # ∫_1^2 (x - 1)(2 - x) dx


def test_ndtri_array_bekannte_werte():
    z = A.ndtri_array(np.array([0.5, 0.975, 0.025, 0.8413447460685429]))
    assert z == pytest.approx([0.0, 1.959964, -1.959964, 1.0], abs=2e-8)


# --- Netze, exakte Wahrheit ---------------------------------------------------------------------------------------------------------------------

def test_netz_3x3_zuverlaessigkeit_und_ausfall():
    tab, m, Nk = A.grid_tab(3, 3)
    assert m == 12 and len(Nk) == 13 and sum(Nk) == int(tab.sum())
    assert A.make_task("netz33_p90").truth == pytest.approx(0.0530152, abs=5e-8)
    assert A.make_task("netz33_p99").truth == pytest.approx(0.00041578, abs=5e-9)
    assert A.make_task("netz34_p95").truth == pytest.approx(0.0120679, abs=5e-8)


def test_gitter_kantenzahl_und_ausfallzaehlung():
    assert [len(A.grid_edges(r, c)[1]) for r, c in ((3, 3), (3, 4), (2, 3))] == [12, 17, 7]
    tab, m, _ = A.grid_tab(3, 3)
    Nf = A.failure_counts(tab, m)
    assert Nf.sum() == (~tab).sum() and Nf[0] == 0 and Nf[m] == 1            # keine Kante fällt aus: zusammenhängend; alle fallen aus: genau eine Menge, nicht zusammenhängend
    assert Nf[1] == 0                                                         # eine ausgefallene Kante trennt das 3×3-Gitter nie (jeder Knoten hat mindestens zwei Kanten)
    assert Nf[2] == 4                                                         # die vier Ecken: beide Kanten der Ecke ausgefallen


def test_engpass_wahrheit_und_wertebereich():
    t = A.make_task("engpass33")
    u = A.points("mc", 2000, t.d, 5)
    y = t.f(u)
    assert 0 <= y.min() and y.max() <= 1
    assert abs(y.mean() - t.truth) < 4 * y.std() / math.sqrt(2000)


# --- Importance Sampling ---------------------------------------------------------------------------------------------------------------------------

def test_importance_sampling_mit_q_gleich_q_hat_die_varianz_der_einfachen_ziehung():
    tab, m, _ = A.grid_tab(3, 3)
    Nf = A.failure_counts(tab, m)
    q = 0.1
    P = A.make_task("netz33_p90").truth
    cv2 = float(A.is_second_moment(Nf, q, q)[0])
    assert cv2 == pytest.approx((1 - P) / P, rel=1e-9)                       # Bernoulli: Var/P² = (1-P)/P


def test_optimale_hilfsverteilung_ist_besser_als_keine():
    t = A.make_task("netz33_p99")
    Nf = A.failure_counts(t.net["tab"], t.net["m"])
    q, qopt = t.net["q"], t.net["qopt"]
    assert A.is_second_moment(Nf, q, qopt)[0] < 0.05 * A.is_second_moment(Nf, q, q)[0]
    assert qopt > q


def test_importance_sampling_ist_erwartungstreu_und_hat_gewichtsmittel_eins():
    t = A.make_task("netz33_p99")
    est = np.array([A.estimate(t, "is", 1024, A.stream_seed(5, r)) for r in range(300)])
    assert abs(est.mean() - t.truth) < 4 * est.std() / math.sqrt(300)
    m, q, qp = t.net["m"], t.net["q"], t.net["qopt"]
    k = (A.points("mc", 50000, m, 3) < qp).sum(axis=1)
    w = (q / qp) ** k * ((1 - q) / (1 - qp)) ** (m - k)
    assert abs(w.mean() - 1) < 0.05


# --- Schätzer von Hand -------------------------------------------------------------------------------------------------------------------------

def test_antithetisch_und_kontrollvariate_sind_bei_linearer_aufgabe_exakt():
    t = _linear()
    for s in range(5):
        assert A.estimate(t, "anti", 2, s) == pytest.approx(0.5, abs=1e-12)  # (u + 1 - u)/2
        assert A.estimate(t, "cv", 8, s) == pytest.approx(0.5, abs=1e-12)    # y = c, β = 1, Schätzer = μ_c


def test_latin_hypercube_bei_zwei_punkten_hat_ein_viertel_der_varianz():
    t = _linear()
    lhs = np.array([A.estimate(t, "lhs", 2, A.stream_seed(1, r)) for r in range(20000)])
    mc = np.array([A.estimate(t, "mc", 2, A.stream_seed(2, r)) for r in range(20000)])
    assert lhs.var() == pytest.approx(1 / 96, rel=0.06) and mc.var() == pytest.approx(1 / 24, rel=0.06)


def test_antithetisch_bei_symmetrischer_aufgabe_hat_den_doppelten_faktor():
    t = _sym()
    N = 64
    an = np.array([A.estimate(t, "anti", N, A.stream_seed(3, r)) for r in range(6000)])
    mc = np.array([A.estimate(t, "mc", N, A.stream_seed(4, r)) for r in range(6000)])
    assert an.var() / mc.var() == pytest.approx(2.0, rel=0.12)


def test_estimate_se_liefert_standardfehler_und_quantile():
    t = A.make_task("glatt12")
    for m, zq in (("mc", C.Z95), ("qmc", C.Z95), ("qmc8", C.T975[8]), ("anti", C.Z95), ("cv", C.Z95), ("lhs", C.Z95)):
        e, se, q = A.est_se(t, m, 256, 7)
        assert se > 0 and q == zq and abs(e - t.truth) < 8 * se


def test_verfahren_je_aufgabe():
    assert A.methods_for(A.make_task("netz33_p90")) == C.METHODS
    assert "is" not in A.methods_for(A.make_task("glatt12"))


# --- Probelauf ----------------------------------------------------------------------------------------------------------------------------------

def test_probelauf_symmetrische_aufgabe_sagt_antithetisch_zwei_voraus():
    p = A.pilot(_sym(), 200, 11)
    assert p["rho_a"] == pytest.approx(1.0, abs=1e-12) and p["pred"]["anti"] == pytest.approx(2.0) and p["cost"] == 2 * 200 + 512


def test_probelauf_lineare_aufgabe_sagt_kontrollvariate_null_voraus():
    p = A.pilot(_linear(), 200, 11)
    assert p["rho"] == pytest.approx(1.0) and p["pred"]["cv"] == pytest.approx(0.0, abs=1e-12) and p["pred"]["anti"] == pytest.approx(0.0, abs=1e-9)


def test_additiver_anteil_additiv_und_reine_wechselwirkung():
    u = A.points("mc", 4000, 4, 21)
    assert A.additive_share(u, u.sum(axis=1)) > 0.9
    inter = np.prod(u[:, :2] - 0.5, axis=1)                                   # Produkt zweier zentrierter Größen: keine Haupteffekte
    assert A.additive_share(u, inter) < 0.15


def test_probelauf_blind_bei_weniger_als_zehn_treffern():
    p = A.pilot(A.make_task("netz33_p99"), 200, 5)
    assert p["blind"] and p["hits"] < C.RARE_HITS
    q = A.pilot(A.make_task("glatt12"), 200, 5)
    assert not q["blind"] and q["hits"] is None


def test_wahl_nach_dem_probelauf():
    blind = dict(pred={"mc": 1.0, "cv": 0.5, "anti": 0.9, "lhs": 0.8, "qmc": 0.1}, blind=True)
    assert A.pilot_choice(blind, 0.014) == "is" and A.pilot_choice(blind, None) == "mc"
    sehend = dict(pred={"mc": 1.0, "cv": 0.5, "anti": 0.9, "lhs": 0.8, "qmc": 0.1}, blind=False)
    assert A.pilot_choice(sehend, None) == "qmc" and A.pilot_choice(sehend, 0.05) == "is" and A.pilot_choice(sehend, 0.3) == "qmc"


# --- Regel --------------------------------------------------------------------------------------------------------------------------------------

def test_trefferzahl_regel_an_den_schwellen():
    net = A.make_task("netz33_p90")
    assert A.recommend(net, 1024, 0.01)[0] == "is"                           # 10,24 < 50
    assert A.recommend(net, 1024, 0.0530)[0] == "qmc8"                       # 54,3 > 50
    assert A.recommend(net, 1024, 50 / 1024)[0] == "qmc8"                    # genau 50 ist nicht darunter
    assert A.recommend(net, 4096, 0.0118)[0] == "is"                         # 48,3 < 50
    assert A.recommend(net, 4096, 0.0125)[0] == "qmc8"                       # 51,2 > 50
    assert A.recommend(net, 256, 0.0118)[0] == "is"
    rec, npi = A.recommend(A.make_task("glatt12"), 1024, None)
    assert rec == "qmc8" and math.isnan(npi)                                 # kein Ereignis, keine Schätzung
    assert A.recommend(A.make_task("sprung32"), 1024, 0.01)[0] == "qmc8"     # kein Importance Sampling für diese Aufgabe
