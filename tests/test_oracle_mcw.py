"""Orakel (ORACLE-PLAYBOOK §5): unabhängige Rechenwege gegen den Kern. scipy kommt nur hier vor, nie zur Laufzeit."""

import math
from itertools import product

import numpy as np
import pytest
from scipy import special, stats
from scipy.stats import qmc

import mcw_algorithm as A
import mcw_constants as C


def _zusammenhang_dfs(n, edges, present):
    """Unabhängiger Zusammenhangstest: Tiefensuche über die vorhandenen Kanten."""
    adj = {i: [] for i in range(n)}
    for (u, v), ok in zip(edges, present):
        if ok:
            adj[u].append(v)
            adj[v].append(u)
    seen, stack = {0}, [0]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == n


# --- Sobol und Normalverteilung -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [2, 7, 21, 64])
def test_sobol_ganzzahlig_gegen_scipy(d):
    N = 128
    ours = A.sobol_ints(N, d).astype(np.float64) / float(1 << A.BITS)
    ref = qmc.Sobol(d, scramble=False).random(N)
    assert np.array_equal(ours, ref)


def test_ndtri_array_gegen_scipy():
    p = np.concatenate([np.geomspace(1e-12, 0.02, 30), np.linspace(0.021, 0.979, 60), 1 - np.geomspace(1e-12, 0.02, 30)])
    assert np.allclose(A.ndtri_array(p), special.ndtri(p), rtol=2e-8, atol=1e-9)


def test_owen_scrambling_ist_gleichverteilt_und_erwartungstreu():
    pts = np.concatenate([A.points("sobol_owen", 64, 3, s) for s in range(200)])
    assert stats.kstest(pts[:, 0], "uniform").pvalue > 0.01 and stats.kstest(pts[:, 2], "uniform").pvalue > 0.01
    t = A.make_task("glatt12")
    est = np.array([A.estimate(t, "qmc", 64, A.stream_seed(8, r)) for r in range(400)])
    assert abs(est.mean() - t.truth) < 4 * est.std() / math.sqrt(400)


# --- Netze: Zusammenhang, Wahrheit -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rows,cols", [(2, 3), (3, 3)])
def test_zusammenhangstabelle_gegen_tiefensuche(rows, cols):
    n, edges = A.grid_edges(rows, cols)
    tab, m, _ = A.grid_tab(rows, cols)
    rng = np.random.default_rng(1)
    for mask in rng.integers(0, 1 << m, 400):
        present = [(int(mask) >> e) & 1 for e in range(m)]
        assert bool(tab[int(mask)]) == _zusammenhang_dfs(n, edges, present)


def test_ausfallwahrscheinlichkeit_gegen_direkte_summe_ueber_alle_mengen():
    n, edges = A.grid_edges(2, 3)
    m = len(edges)
    for p in (0.5, 0.9):
        s = 0.0
        for bits in product((0, 1), repeat=m):
            if not _zusammenhang_dfs(n, edges, bits):
                k = sum(bits)
                s += p**k * (1 - p) ** (m - k)
        _, _, Nk = A.grid_tab(2, 3)
        assert 1 - A.r_exact(Nk, p) == pytest.approx(s, rel=1e-12)


def test_netz_ausfall_gegen_simulation():
    t = A.make_task("netz33_p90")
    y = t.f(A.points("mc", 400000, t.d, 77))
    assert abs(y.mean() - t.truth) < 4 * y.std() / math.sqrt(len(y))


def test_irwin_hall_gegen_scipy_und_simulation():
    d = 32
    thr = d / 2 + 1.6449 * math.sqrt(d / 12)
    assert A.irwin_hall_sf(d, thr) == pytest.approx(float(stats.irwinhall(d).sf(thr)), rel=1e-6)
    s = A.points("mc", 500000, d, 3).sum(axis=1)
    assert A.irwin_hall_pos_part(d) == pytest.approx(float(np.maximum(s - d / 2, 0).mean()), abs=4 * s.std() / math.sqrt(len(s)))


def test_exakte_wahrheiten_der_glatten_aufgaben():
    a = np.ones(12)
    assert A.exp_exact(a) == pytest.approx((math.e - 1) ** 12)
    assert A.make_task("schwanz2").truth == pytest.approx(math.exp(2.0))
    assert A.make_task("symmetrisch12").truth == pytest.approx(1.0)               # Var(Σ uᵢ) = 12/12
    assert A.make_task("produkt8").truth == 1.0


# --- Importance Sampling: zweites Moment ---------------------------------------------------------------------------------------------------------

def test_importance_sampling_zweites_moment_gegen_direkte_summe():
    n, edges = A.grid_edges(2, 3)
    m = len(edges)
    q, qp = 0.1, 0.3
    P, M2 = 0.0, 0.0
    for bits in product((0, 1), repeat=m):                   # 1 = Kante hält
        if _zusammenhang_dfs(n, edges, bits):
            continue
        f = m - sum(bits)
        pq = q**f * (1 - q) ** (m - f)
        pqp = qp**f * (1 - qp) ** (m - f)
        P += pq
        M2 += pq * pq / pqp
    tab, mm, _ = A.grid_tab(2, 3)
    Nf = A.failure_counts(tab, mm)
    assert float(A.is_second_moment(Nf, q, qp)[0]) == pytest.approx(M2 / P**2 - 1, rel=1e-10)


def test_importance_sampling_streuung_gegen_die_exakte_formel():
    t = A.make_task("netz33_p90")
    Nf = A.failure_counts(t.net["tab"], t.net["m"])
    cv2 = float(A.is_second_moment(Nf, t.net["q"], t.net["qopt"])[0])
    N = 2000
    est = np.array([A.estimate(t, "is", N, A.stream_seed(6, r)) for r in range(400)])
    assert est.std() == pytest.approx(t.truth * math.sqrt(cv2 / N), rel=0.12)


# --- Schätzer sind erwartungstreu (die Kontrollvariate mit kleiner Verzerrung) -------------------------------------------------------------

@pytest.mark.parametrize("method", ["mc", "anti", "lhs", "qmc8"])
def test_schaetzer_erwartungstreu(method):
    t = A.make_task("glatt12")
    est = np.array([A.estimate(t, method, 256, A.stream_seed(12, r)) for r in range(500)])
    assert abs(est.mean() - t.truth) < 4 * est.std() / math.sqrt(500)


def test_kontrollvariate_verzerrung_ist_klein():
    t = A.make_task("glatt12")
    est = np.array([A.estimate(t, "cv", 1024, A.stream_seed(13, r)) for r in range(600)])
    assert abs(est.mean() - t.truth) < 5 * est.std() / math.sqrt(600)


# --- Vorhersage des Probelaufs gegen exakte Formeln -------------------------------------------------------------------------------------------

def test_kontrollvariate_vorhersage_gleich_eins_minus_rho_quadrat():
    t = A.make_task("glatt12")
    u = A.points("mc", 5000, t.d, 4)
    y, c = t.f(u), t.control(u)
    rho = np.corrcoef(y, c)[0, 1]
    p = A.pilot(t, 5000, 4)
    assert p["pred"]["cv"] == pytest.approx(1 - rho**2, abs=1e-12)


def test_antithetische_vorhersage_gegen_gemessenen_faktor():
    t = A.make_task("engpass33")
    N = 512
    an = np.array([A.estimate(t, "anti", N, A.stream_seed(14, r)) for r in range(1200)])
    mc = np.array([A.estimate(t, "mc", N, A.stream_seed(15, r)) for r in range(1200)])
    p = A.pilot(t, 20000, 6)
    assert p["pred"]["anti"] == pytest.approx(an.var() / mc.var(), abs=0.08)
