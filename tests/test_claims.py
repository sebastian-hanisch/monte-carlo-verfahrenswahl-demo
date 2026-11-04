"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Alle Messungen mit den Werten der App: Strom 40, 300 Wiederholungen (Probelauf: 100 Wiederholungen der Wahl). Die Bänder decken das Rauschen ab: Varianzfaktoren aus 300 Wiederholungen sind auf etwa ±8 % (glatte Aufgaben)
bis ±35 % (Ereignisindikatoren) genau, Abdeckungen auf etwa ±0,015. Zeiten werden nie getestet. Die Zahlen der Stücke 1 bis 15 im Baum werden nicht nachgerechnet, nur auf Vorhandensein in der README geprüft."""

import math
from pathlib import Path

import numpy as np
import pytest

import mcw_algorithm as A
import mcw_constants as C
import mcw_evaluation as ev
import mcw_scenario as S

SEED = C.DEFAULT_SEED
README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def _in(x, lo, hi, what=""):
    assert lo <= x <= hi, f"{what}: {x} nicht in [{lo}, {hi}]"


def _m(task, N=1024):
    return ev.measure(task, N, SEED)


def _g(values):
    return float(np.exp(np.mean(np.log(values))))


# --- exakte Werte ----------------------------------------------------------------------------------------------------------------------------

def test_exakte_wahrheiten():
    for key, wahr in (("netz33_p90", 0.0530152), ("netz33_p99", 0.00041578), ("netz34_p95", 0.0120679), ("engpass33", 0.313287), ("sprung32", 0.0500752), ("knick32", 0.652494)):
        assert A.make_task(key).truth == pytest.approx(wahr, abs=6e-7 if wahr > 0.001 else 5e-9), key
    assert A.make_task("glatt12").truth == pytest.approx(662.419, abs=5e-4) and A.make_task("glatt64").truth == pytest.approx(149.872, abs=5e-4)
    assert A.make_task("schwanz2").truth == pytest.approx(7.38906, abs=5e-6) and A.make_task("symmetrisch12").truth == 1.0 and A.make_task("produkt8").truth == 1.0
    assert S.net_pi("3x3", 0.95) == pytest.approx(0.01184, abs=5e-6) and S.net_pi("3x4", 0.9) == pytest.approx(0.05549, abs=5e-6)


def test_kosten_des_probelaufs_sind_arithmetik():
    assert 2 * 200 + 512 == 912 and 2 * 1000 + 512 == 2512
    assert (912 / 1024, 2512 / 1024) == (pytest.approx(0.89, abs=0.005), pytest.approx(2.45, abs=0.005))
    assert (912 / 4096, 2512 / 4096) == (pytest.approx(0.22, abs=0.005), pytest.approx(0.61, abs=0.005))


# --- Experiment 1: wer gewinnt wo -------------------------------------------------------------------------------------------------------------

def test_sieger_bei_n_1024():
    best = {k: ev.analyse(ev.Settings(task=k)).best for k in C.TASK_KEYS}
    assert best["netz33_p99"] == "is" and best["netz34_p95"] == "is"
    for k in ("netz33_p90", "engpass33", "glatt12", "glatt64", "symmetrisch12", "produkt8", "sprung32", "knick32"):
        assert best[k] == "qmc", k
    assert sum(b == "qmc" for b in best.values()) >= 8 and sum(b == "is" for b in best.values()) == 2
    assert "anti" not in best.values()


@pytest.mark.parametrize("task,lo,hi", [("netz33_p90", 0.10, 0.30), ("engpass33", 0.06, 0.20), ("glatt12", 0.005, 0.07), ("glatt64", 0.0005, 0.006), ("symmetrisch12", 0.0005, 0.005),
                                         ("produkt8", 0.03, 0.14), ("sprung32", 0.25, 0.55), ("knick32", 0.02, 0.09)])
def test_sobol_faktoren_bei_n_1024(task, lo, hi):
    _in(_m(task)["qmc"]["factor"], lo, hi, task)


def test_importance_sampling_faktoren_bei_n_1024():
    _in(_m("netz33_p99")["is"]["factor"], 0.008, 0.035, "p99")
    _in(_m("netz34_p95")["is"]["factor"], 0.15, 0.45, "3x4")
    _in(_m("netz33_p90")["is"]["factor"], 0.22, 0.50, "p90")


def test_antithetisch_gewinnt_nie_und_verliert_bei_symmetrie():
    for k in C.TASK_KEYS:
        assert min(_m(k), key=lambda m: _m(k)[m]["factor"]) != "anti", k
    for N in C.N_CHOICES:
        _in(_m("symmetrisch12", N)["anti"]["factor"], 1.5, 3.2, f"N={N}")
    _in(_m("symmetrisch12")["cv"]["factor"], 0.7, 1.6, "Kontrollvariate symmetrisch")


def test_rangwechsel_beim_budget_netz_3x4():
    f1, f3 = _m("netz34_p95", 1024), _m("netz34_p95", 4096)
    assert f1["is"]["factor"] < 0.6 * f1["qmc"]["factor"]
    _in(f1["qmc"]["factor"], 0.45, 0.95, "Sobol 1024")
    _in(f3["qmc"]["factor"], 0.10, 0.35, "Sobol 4096")
    _in(f3["is"]["factor"], 0.15, 0.35, "IS 4096")
    assert f3["qmc"]["factor"] < f1["qmc"]["factor"] / 2


def test_sobol_gewinn_waechst_mit_dem_budget():
    f = [_m("glatt12", N)["qmc"]["factor"] for N in C.N_CHOICES]
    assert f[0] > f[1] > f[2]
    _in(f[0], 0.03, 0.12, "256")
    _in(f[2], 0.006, 0.03, "4096")


def test_seltener_ausfall_alle_ausser_importance_sampling_bei_eins():
    f = _m("netz33_p99")
    for m in ("anti", "cv", "lhs", "qmc", "qmc8"):
        _in(f[m]["factor"], 0.6, 1.5, m)


# --- Experiment 2: Ist der Balken ehrlich? ---------------------------------------------------------------------------------------------------

def _cv(task):
    return ev.cover(task, 1024, SEED)


def test_abdeckung_seltener_ausfall():
    cv = _cv("netz33_p99")
    for m in ("mc", "anti", "cv", "lhs", "qmc", "qmc8"):
        _in(cv[m]["cover"], 0.28, 0.45, m)
    _in(cv["is"]["cover"], 0.90, 0.99, "Importance Sampling")


def test_abdeckung_schwerer_schwanz():
    cv = _cv("schwanz2")
    for m in ("mc", "anti", "cv"):
        _in(cv[m]["cover"], 0.76, 0.92, m)
    _in(cv["qmc8"]["cover"], 0.62, 0.86, "Sobol Blöcke")
    assert cv["lhs"]["cover"] >= 0.97 and cv["qmc"]["cover"] >= 0.97
    _in(cv["lhs"]["se_ratio"], 1.2, 2.2, "Latin Hypercube")
    _in(cv["qmc"]["se_ratio"], 0.9, 1.5, "Sobol Folge")


@pytest.mark.parametrize("task,lo,hi", [("netz33_p90", 1.5, 2.6), ("engpass33", 2.0, 4.0), ("glatt12", 4.0, 9.0), ("glatt64", 14.0, 32.0), ("symmetrisch12", 14.0, 33.0),
                                         ("produkt8", 2.5, 5.5), ("sprung32", 1.2, 2.0), ("knick32", 3.0, 6.5)])
def test_naiver_sobol_balken_zu_breit(task, lo, hi):
    cv = _cv(task)
    _in(cv["qmc"]["se_ratio"], lo, hi, task)
    assert cv["qmc"]["cover"] >= 0.97


@pytest.mark.parametrize("task", ["netz33_p90", "netz34_p95", "engpass33", "glatt12", "glatt64", "symmetrisch12", "produkt8", "sprung32", "knick32"])
def test_sobol_bloecke_decken_und_sind_ehrlich(task):
    cv = _cv(task)
    _in(cv["qmc8"]["cover"], 0.89, 1.0, task)
    _in(cv["qmc8"]["se_ratio"], 0.8, 1.25, task)
    _in(cv["mc"]["cover"], 0.88, 1.0, task)


def test_bloecke_kosten_genauigkeit():
    for task, lo in (("netz33_p90", 1.2), ("engpass33", 1.5), ("glatt12", 1.8), ("glatt64", 3.0), ("symmetrisch12", 10.0), ("produkt8", 2.0), ("knick32", 1.4), ("sprung32", 1.05)):
        cv = _cv(task)
        assert cv["qmc8"]["factor"] / cv["qmc"]["factor"] >= lo, task
    cv = _cv("symmetrisch12")
    _in(cv["qmc8"]["factor"], 0.02, 0.12, "Blöcke symmetrisch")
    _in(cv["qmc"]["factor"], 0.0005, 0.006, "Folge symmetrisch")


# --- Experiment 3: Kippschwelle -------------------------------------------------------------------------------------------------------------

def _cells():
    return [r for netz in C.NETZE for N in (1024, 4096) for r in ev.cross(netz, N, SEED)]


def test_regel_in_23_von_24_zellen_und_regret():
    cells = _cells()
    assert len(cells) == 24
    ok = sum(r["rule"] == r["best"] for r in cells)
    assert ok >= 22
    regret = lambda pick: [r["factor"][pick(r)] / min(r["factor"]["qmc8"], r["factor"]["is"]) for r in cells]
    rule = regret(lambda r: r["rule"])
    _in(_g(rule), 1.0, 1.1, "Regel")
    assert max(rule) < 1.8
    _in(_g(regret(lambda r: "qmc8")), 2.5, 4.0, "immer Sobol-Blöcke")
    assert max(regret(lambda r: "qmc8")) > 20
    _in(_g(regret(lambda r: "is")), 1.05, 1.45, "immer Importance Sampling")
    assert max(regret(lambda r: "is")) < 3.0


def test_kippschwelle_liegt_bei_etwa_50_und_schwelle_20_ist_schlechter():
    cells = _cells()
    is_n = [r["npi"] for r in cells if r["best"] == "is"]
    q_n = [r["npi"] for r in cells if r["best"] == "qmc8"]
    _in(max(is_n), 40, 60, "größtes N·π mit Importance Sampling")
    _in(min(q_n), 40, 70, "kleinstes N·π mit Sobol-Blöcken")

    def right(thr):
        return sum(("is" if r["npi"] < thr else "qmc8") == r["best"] for r in cells)
    assert right(50.0) > right(20.0) and right(50.0) >= right(100.0) - 1


def test_regel_an_beiden_netzen_gleich_gut():
    for netz in C.NETZE:
        rows = [r for N in (1024, 4096) for r in ev.cross(netz, N, SEED)]
        assert sum(r["rule"] == r["best"] for r in rows) >= 11
    rows = ev.cross("3x3", 1024, SEED)
    p90 = next(r for r in rows if r["p"] == 0.9)
    _in(p90["npi"], 54.0, 54.6, "N·π")
    _in(p90["factor"]["is"], 0.22, 0.50, "IS")
    _in(p90["factor"]["qmc8"], 0.30, 0.65, "Blöcke")
    r99 = next(r for r in rows if r["p"] == 0.99)
    assert r99["factor"]["is"] < 0.05 and r99["factor"]["qmc8"] > 0.7


# --- Experiment 4: Probelauf -----------------------------------------------------------------------------------------------------------------

def _pl(task, n0, N=1024):
    return ev.pilots(task, N, n0, SEED)


def test_probelauf_sammelwerte_bei_n_1024():
    for n0, reg, within in ((200, (1.4, 2.5), (0.55, 0.80)), (1000, (1.0, 1.6), (0.70, 0.92))):
        regs = [_pl(k, n0)["regret"] for k in C.TASK_KEYS]
        ins = [_pl(k, n0)["within"] for k in C.TASK_KEYS]
        _in(_g(regs), *reg, f"Regret n0={n0}")
        _in(float(np.mean(ins)), *within, f"innerhalb 1,5 n0={n0}")


def test_immer_ein_verfahren_regret():
    true = {k: {m: v["factor"] for m, v in _m(k).items() if m != "qmc8"} for k in C.TASK_KEYS}
    best = {k: min(v.values()) for k, v in true.items()}
    always = {m: _g([true[k][m] / best[k] for k in C.TASK_KEYS]) for m in ("mc", "anti", "cv", "lhs", "qmc")}
    _in(always["qmc"], 1.3, 1.9, "immer Sobol")
    _in(always["mc"], 12.0, 28.0, "immer einfach")
    _in(always["anti"], 9.0, 21.0, "immer antithetisch")
    _in(always["cv"], 8.0, 19.0, "immer Kontrollvariate")
    _in(always["lhs"], 7.0, 15.0, "immer Latin Hypercube")
    nets = [k for k in C.TASK_KEYS if "is" in _m(k)]
    _in(_g([_m(k)["is"]["factor"] / best[k] for k in nets]), 1.0, 1.6, "immer Importance Sampling")
    assert len(nets) == 3


def test_probelauf_blind_bei_seltenem_ausfall():
    assert _pl("netz33_p99", 200)["blind"] == 1.0 and _pl("netz33_p99", 1000)["blind"] == 1.0
    assert _pl("netz34_p95", 200)["blind"] == 1.0
    _in(_pl("netz34_p95", 1000)["blind"], 0.1, 0.45, "blind 3×4 n0=1000")
    assert _pl("netz33_p99", 200)["picks"]["is"] == 100 and _pl("netz33_p99", 200)["regret"] == pytest.approx(1.0)


def test_latin_hypercube_vorhersage_versagt_bei_64_dimensionen_und_kleinem_probelauf():
    pl = _pl("glatt64", 200)
    assert pl["picks"]["lhs"] >= 70 and pl["regret"] > 20
    _in(pl["pred"]["lhs"], 0.005, 0.03, "vorhergesagt")
    _in(pl["true"]["lhs"], 0.10, 0.20, "gemessen")
    assert _pl("glatt64", 1000)["picks"]["qmc"] >= 85
    _in(_pl("glatt64", 1000)["pred"]["lhs"], 0.07, 0.14, "vorhergesagt n0=1000")


def test_vorhersage_kontrollvariate_antithetisch_und_sobol():
    pl = _pl("symmetrisch12", 1000)
    assert pl["pred"]["anti"] == pytest.approx(2.0, abs=1e-6)
    _in(pl["true"]["anti"], 1.5, 3.2, "gemessen")
    g = _pl("glatt12", 1000)
    _in(g["pred"]["cv"], 0.30, 0.45, "cv vorhergesagt")
    _in(g["true"]["cv"], 0.22, 0.40, "cv gemessen")
    assert abs(g["pred"]["cv"] - g["true"]["cv"]) < 0.15
    _in(g["pred"]["qmc"], 0.08, 0.20, "Sobol vorhergesagt")
    _in(g["true"]["qmc"], 0.01, 0.07, "Sobol gemessen")
    assert g["pred"]["qmc"] > 2 * g["true"]["qmc"]                  # der Sobol-Gewinn wird unterschätzt


# --- Presets ---------------------------------------------------------------------------------------------------------------------------------

def test_preset_standardfall():
    a = ev.analyse(ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]))
    assert a.task == "netz33_p90" and a.N == 1024
    _in(a.npi, 35.0, 90.0, "N·π̂")
    assert a.factor["is"] / a.factor["qmc8"] < 1.0 + 0.5 and a.factor["qmc8"] / a.factor["is"] < 1.0 + 0.7       # eng beieinander


def test_preset_seltener_ausfall():
    a = ev.analyse(ev.Settings(**C.PRESETS["Seltener Ausfall"]))
    assert a.best == "is" and a.rec == "is" and a.factor["is"] < 0.04 and min(v for k, v in a.factor.items() if k != "is") > 0.5


def test_preset_glatt_symmetrie_schwanz_rau():
    g = ev.analyse(ev.Settings(**C.PRESETS["Glatt und hochdimensional"]))
    assert g.factor["qmc"] < 0.006 and ev.cover("glatt64", 1024, SEED)["qmc"]["se_ratio"] > 14
    s = ev.analyse(ev.Settings(**C.PRESETS["Symmetrie"]))
    assert s.factor["anti"] > 1.5 and s.factor["qmc"] < 0.005
    w = ev.cover("schwanz2", 1024, SEED)
    assert max(w["mc"]["cover"], w["anti"]["cover"], w["cv"]["cover"]) < 0.92 and w["qmc8"]["cover"] < 0.86
    r = ev.analyse(ev.Settings(**C.PRESETS["Rau und hochdimensional"]))
    _in(r.factor["qmc"], 0.25, 0.55, "Sobol rau")
    for m in ("anti", "cv", "lhs"):
        _in(r.factor[m], 0.55, 1.1, m)


# --- Baum und README ---------------------------------------------------------------------------------------------------------------------------

def test_zahlen_des_baums_stehen_in_der_readme():
    for frag in ("Maximum Abdeckung 0,000", "Standardabweichung 0,586", "50 bis 81 %", "0,009 bis 0,019", "Verzerrung −100 %", "18- bis 31-mal", "2- bis 30-mal", "ε⁻³", "14- bis 25-mal", "57-mal"):
        assert any(frag in s["folge"] for info in C.INFO for s in S.tree_steps(info)), frag
    for frag in ("bootstrap-demo", "abbruchregeln-demo", "monte-carlo-gradient-demo", "sobol-indizes-demo", "geschachtelte-erwartungen-demo", "multilevel-monte-carlo-demo", "mehrere-modelle-demo", "posterior-sampling-demo"):
        assert frag in README, frag
    for info in C.INFO:                                                       # jeder Eintrag des Baums steht wörtlich in der README
        for s in S.tree_steps(info):
            if s["antwort"] == "Ja":
                assert s["folge"] in README, s["folge"]


# --- Tabellen der README, Zelle für Zelle -------------------------------------------------------------------------------------------------------

MATRIX_1024 = {   # antithetisch, Kontrollvariate, Latin Hypercube, Sobol 1, Sobol 8, Importance Sampling
    "netz33_p90": (0.811, 0.754, 0.714, 0.189, 0.372, 0.361), "netz33_p99": (1.01, 0.891, 0.884, 1.13, 1.08, 0.0177), "netz34_p95": (0.851, 0.848, 0.841, 0.692, 0.736, 0.298),
    "engpass33": (0.622, 0.513, 0.442, 0.116, 0.280, None), "glatt12": (0.493, 0.307, 0.306, 0.0297, 0.0720, None), "glatt64": (0.322, 0.919, 0.154, 0.0025, 0.0136, None),
    "symmetrisch12": (2.44, 1.14, 1.07, 0.0019, 0.0488, None), "schwanz2": (0.624, 0.734, 0.549, 0.289, 0.803, None), "produkt8": (0.931, 0.859, 0.659, 0.0823, 0.272, None),
    "sprung32": (0.714, 0.733, 0.758, 0.406, 0.521, None), "knick32": (0.552, 0.275, 0.280, 0.0497, 0.121, None),
}
COVER_1024 = {    # einfach, antithetisch, Kontrollvariate, Latin Hypercube, Sobol 1, Sobol 8, Importance Sampling
    "netz33_p90": (0.97, 0.96, 0.94, 0.99, 1.00, 0.96, 0.94), "netz33_p99": (0.36, 0.37, 0.36, 0.33, 0.35, 0.35, 0.95), "netz34_p95": (0.92, 0.92, 0.94, 0.94, 0.95, 0.93, 0.94),
    "engpass33": (0.94, 0.92, 0.96, 1.00, 1.00, 0.95, None), "glatt12": (0.95, 0.96, 0.92, 1.00, 1.00, 0.96, None), "glatt64": (0.92, 0.94, 0.95, 1.00, 1.00, 0.94, None),
    "symmetrisch12": (0.97, 0.94, 0.94, 0.96, 1.00, 0.98, None), "schwanz2": (0.86, 0.84, 0.82, 1.00, 1.00, 0.74, None), "produkt8": (0.94, 0.94, 0.91, 0.97, 1.00, 0.94, None),
    "sprung32": (0.96, 0.96, 0.95, 0.97, 1.00, 0.95, None), "knick32": (0.95, 0.96, 0.95, 1.00, 1.00, 0.96, None),
}


def test_tabelle_faktoren_bei_n_1024_zelle_fuer_zelle():
    for task, row in MATRIX_1024.items():
        f = _m(task)
        for method, soll in zip(("anti", "cv", "lhs", "qmc", "qmc8", "is"), row):
            if soll is None:
                assert method not in f, (task, method)
                continue
            if task == "schwanz2":                                              # schwerer Schwanz: Fehler von Lauf zu Lauf zu groß für enge Bänder
                _in(f[method]["factor"], soll / 3.0, soll * 3.0, f"{task} {method}")
            else:
                _in(f[method]["factor"], soll * 0.7 - 0.02 * (soll > 0.1), soll * 1.4 + 0.02 * (soll > 0.1), f"{task} {method}")


def test_tabelle_abdeckung_bei_n_1024_zelle_fuer_zelle():
    for task, row in COVER_1024.items():
        cv = _cv(task)
        for method, soll in zip(("mc", "anti", "cv", "lhs", "qmc", "qmc8", "is"), row):
            if soll is None:
                assert method not in cv
                continue
            _in(cv[method]["cover"], soll - 0.05, soll + 0.05, f"{task} {method}")


def test_tabelle_kosten_der_bloecke():
    soll = {"netz33_p90": (0.246, 0.472), "engpass33": (0.114, 0.303), "glatt12": (0.032, 0.100), "glatt64": (0.0017, 0.0116), "symmetrisch12": (0.0020, 0.0561),
            "produkt8": (0.065, 0.273), "knick32": (0.046, 0.097), "sprung32": (0.405, 0.536)}
    for task, (folge, bloecke) in soll.items():
        cv = _cv(task)
        _in(cv["qmc"]["factor"], folge * 0.6, folge * 1.6 + 0.01, f"{task} Folge")
        _in(cv["qmc8"]["factor"], bloecke * 0.6, bloecke * 1.6 + 0.01, f"{task} Blöcke")


def test_tabelle_kippschwelle_zelle_fuer_zelle():
    soll = {("3x3", 1024): {0.8: (236.7, 0.598, 0.370, 0.223), 0.85: (130.3, 0.435, 0.384, 0.260), 0.9: (54.3, 0.331, 0.461, 0.241), 0.95: (12.1, 0.226, 0.702, 0.366), 0.97: (4.1, 0.103, 0.976, 0.346), 0.99: (0.4, 0.0181, 0.974, 0.979)},
            ("3x4", 4096): {0.9: (227.3, 0.525, 0.409, 0.192), 0.95: (49.4, 0.282, 1.01, 0.198), 0.99: (1.7, 0.0321, 1.00, 0.611)}}
    for (netz, N), rows in soll.items():
        for r in ev.cross(netz, N, SEED):
            if r["p"] in rows:
                npi, f_is, f_q8, f_q1 = rows[r["p"]]
                assert r["npi"] == pytest.approx(npi, abs=0.06)
                for got, want, name in ((r["factor"]["is"], f_is, "IS"), (r["factor"]["qmc8"], f_q8, "Blöcke"), (r["factor"]["qmc"], f_q1, "Folge")):
                    _in(got, want * 0.6 - 0.005, want * 1.6 + 0.005, f"{netz} {N} p={r['p']} {name}")


def test_sobol_faktor_gegen_budget_in_der_readme():
    f = [_m("glatt12", N)["qmc"]["factor"] for N in C.N_CHOICES]
    for got, want in zip(f, (0.0644, 0.0297, 0.0134)):
        _in(got, want * 0.6, want * 1.6, "glatt12")
