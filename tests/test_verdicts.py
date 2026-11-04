"""Verdikte (reine Funktionen in mcw_evaluation.py) an von Hand gesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import mcw_constants as C
import mcw_evaluation as ev


def _a(**kw) -> ev.Analysis:
    f = {"mc": 1.0, "anti": 1.052, "cv": 0.80, "lhs": 0.95, "qmc": 0.231, "qmc8": 0.380, "is": 0.340}
    basis = ev.Analysis(task="netz33_p90", label="Netz 3×3, p = 0,9: Ausfall", N=1024, R=300, truth=0.0530152, d=12, mse={m: v * 1e-5 for m, v in f.items()}, factor=f, bias={m: 0.0 for m in f},
                        best="qmc", best_honest="is", rec="qmc8", npi=54.3, pilot={})
    return dataclasses.replace(basis, **kw)


S = ev.Settings()


def test_hauptverdikt_zahlen():
    kind, text = ev.verdict_main(_a(), S)
    assert "Netz 3×3, p = 0,9: Ausfall, N = 1.024 Auswertungen, Wahrheit 0,0530, 300 Wiederholungen" in text
    assert "einfach 1,00, antithetisch 1,05, Kontrollvariate 0,800, Latin Hypercube 0,950, Sobol (eine Folge) 0,231, Sobol (8 Blöcke) 0,380, Importance Sampling 0,340" in text
    assert "Am besten: Sobol (eine Folge) (0,231), mit ehrlicher Fehlerangabe Importance Sampling (0,340)" in text
    assert "N·π̂ = 54,3 (Schwelle 50) → Sobol (8 Blöcke), Faktor 0,380, das 1,12-Fache des Besten mit ehrlichem Balken" in text
    assert kind == "success" and "Schlechter als einfach" not in text


def test_hauptverdikt_schwellen():
    f = dict(_a().factor)
    assert ev.verdict_main(_a(rec="qmc8", best_honest="is"), S)[0] == "success"                     # 0,38 / 0,34 = 1,12
    schlecht = dict(f, qmc8=0.60)                                                                   # 0,60 / 0,34 = 1,76 > 1,5
    assert ev.verdict_main(_a(factor=schlecht), S)[0] == "warning"
    kante = dict(f, qmc8=0.51)                                                                      # 1,5 genau: noch in Ordnung
    assert ev.verdict_main(_a(factor=kante), S)[0] == "success"
    verlierer = dict(f, anti=1.4)
    kind, text = ev.verdict_main(_a(factor=verlierer), S)
    assert kind == "info" and "Schlechter als einfach (Faktor über 1,10): antithetisch 1,40" in text


def test_hauptverdikt_ohne_ereignis_und_mit_importance_sampling_als_regel():
    f = {"mc": 1.0, "anti": 0.5, "cv": 0.4, "lhs": 0.3, "qmc": 0.03, "qmc8": 0.07}
    kind, text = ev.verdict_main(_a(factor=f, mse=f, bias={m: 0.0 for m in f}, best="qmc", best_honest="qmc8", npi=float("nan")), S)
    assert "kein seltenes Ereignis gemessen → Sobol (8 Blöcke), Faktor 0,070, das 1,00-Fache" in text and kind == "success"
    g = dict(_a().factor, **{"is": 0.0145, "qmc8": 1.0775})
    kind, text = ev.verdict_main(_a(factor=g, best="is", best_honest="is", rec="is", npi=0.4), S)
    assert "N·π̂ = 0,4 (Schwelle 50) → Importance Sampling, Faktor 0,015, das 1,00-Fache" in text


def _mat():
    mk = lambda **k: {m: {"factor": v} for m, v in k.items()}
    return {256: mk(mc=1.0, anti=1.1, cv=0.9, lhs=0.9, qmc=0.7, qmc8=1.06, **{"is": 0.32}), 1024: mk(mc=1.0, anti=0.85, cv=0.85, lhs=0.84, qmc=0.69, qmc8=0.74, **{"is": 0.30}),
            4096: mk(mc=1.0, anti=0.95, cv=0.95, lhs=0.88, qmc=0.205, qmc8=0.88, **{"is": 0.24})}


def test_matrixverdikt_rangwechsel_und_verlierer():
    kind, text = ev.verdict_matrix(_mat(), "Netz 3×4, p = 0,95", 300)
    assert kind == "info"
    assert "N = 256: Importance Sampling (0,320); N = 1.024: Importance Sampling (0,300); N = 4.096: Sobol (eine Folge) (0,205)" in text
    assert "Die Rangfolge hängt am Budget: der Sieger wechselt." in text
    assert "Schlechter als einfach (Faktor über 1,10) in mindestens einem Budget: antithetisch 1,10" not in text         # 1,10 ist nicht über 1,10
    m = _mat()
    m[256]["anti"] = {"factor": 1.22}
    assert "antithetisch 1,22" in ev.verdict_matrix(m, "x", 300)[1]
    ruhig = {N: {k: {"factor": v["factor"]} for k, v in mm.items()} for N, mm in _mat().items()}
    for mm in ruhig.values():
        mm["qmc"]["factor"] = 0.05
        mm["anti"]["factor"] = 0.9
        mm["qmc8"]["factor"] = 0.9
    ruhig[256]["lhs"]["factor"] = ruhig[256]["is"]["factor"] = 0.9
    assert ev.verdict_matrix(ruhig, "x", 300)[0] == "success"


def _cv(**over):
    cell = lambda c, r, f: {"cover": c, "se_ratio": r, "factor": f}
    cv = {"mc": cell(0.363, 0.61, 1.0), "anti": cell(0.380, 0.59, 1.3), "cv": cell(0.360, 0.58, 1.1), "lhs": cell(0.343, 0.56, 1.2), "qmc": cell(0.370, 0.62, 1.06), "qmc8": cell(0.357, 0.58, 1.1), "is": cell(0.937, 0.94, 0.02)}
    cv.update(over)
    return cv


def test_ehrlichkeitsverdikt():
    kind, text = ev.verdict_cover(_cv(), "Netz 3×3, p = 0,99", 1024, 300)
    assert kind == "warning"
    assert "Abdeckung: einfach 0,363, antithetisch 0,380, Kontrollvariate 0,360, Latin Hypercube 0,343, Sobol (eine Folge) 0,370, Sobol (8 Blöcke) 0,357, Importance Sampling 0,937" in text
    assert "Unter 0,90 und damit gebrochen: einfach 0,363" in text
    assert "Importance Sampling" not in text.split("gebrochen:")[1].split(". ")[0]
    ok = {m: {"cover": 0.95, "se_ratio": 1.0, "factor": 0.5} for m in ("mc", "qmc", "qmc8")}
    assert ev.verdict_cover(ok, "x", 1024, 300)[0] == "success"
    breit = dict(ok, qmc={"cover": 1.0, "se_ratio": 5.6, "factor": 0.035}, qmc8={"cover": 0.95, "se_ratio": 0.95, "factor": 0.101})
    kind, text = ev.verdict_cover(breit, "x", 1024, 300)
    assert kind == "info" and "Sobol (eine Folge) 5,6" in text and "eine Folge hat Varianzfaktor 0,035 mit naivem Balken, acht Blöcke 0,101 mit ehrlichem" in text


def _rows():
    mk = lambda p, pi, npi, i, q8, best: dict(p=p, pi=pi, npi=npi, factor={"mc": 1.0, "qmc": 0.3, "qmc8": q8, "is": i}, best=best, rule="is" if npi < C.NPI_THRESHOLD else "qmc8")
    return [mk(0.8, 0.2312, 236.8, 0.75, 0.57, "qmc8"), mk(0.9, 0.0530, 54.3, 0.45, 0.51, "is"), mk(0.95, 0.0118, 12.1, 0.15, 0.59, "is"), mk(0.99, 0.0004, 0.4, 0.02, 1.3, "is")]


def test_kippschwellenverdikt():
    kind, text = ev.verdict_cross(_rows(), "3x3", 1024, 300)
    assert "Netz 3x3, N = 1.024, 300 Wiederholungen" in text and "Importance Sampling gewinnt in 3 von 4 Zellen (größtes N·π: 54,3), Sobol in Blöcken in 1 (kleinstes N·π: 236,8)" in text
    assert "Die Regel N·π < 50 trifft in 3 von 4 Zellen." in text and kind == "info"
    alle = _rows()
    alle[1]["best"] = "qmc8"
    assert ev.verdict_cross(alle, "3x3", 1024, 300)[0] == "success"
    schlecht = _rows()
    schlecht[0]["best"] = schlecht[2]["best"] = "x"
    assert ev.verdict_cross(schlecht, "3x3", 1024, 300)[0] == "warning"


def _pl(**over):
    pl = dict(picks={"mc": 0, "cv": 0, "anti": 0, "lhs": 2, "qmc": 63, "is": 35}, regret=1.39, within=0.43, blind=0.0, cost=2512, cost_ratio=2.45, RP=100, best="qmc", pred={}, true={})
    pl.update(over)
    return pl


def test_probelaufverdikt():
    kind, text = ev.verdict_pilots(_pl(), "Netz 3×3, p = 0,9", 1000, 1024)
    assert kind == "warning"
    for frag in ("Probelauf n₀ = 1.000 (Kosten 2.512 Auswertungen = 2,45 × Endbudget N = 1.024), 100 Wiederholungen der Wahl", "Gewählt: Latin Hypercube 2 %, Sobol (eine Folge) 63 %, Importance Sampling 35 %", "Bestes Verfahren: Sobol (eine Folge)",
                 "Innerhalb des 1,5-Fachen des Besten: 43 % der Wahlen, mittleres Verhältnis zum Besten 1,39", "Der Probelauf kostet mehr als der Endlauf."):
        assert frag in text, frag
    kind, text = ev.verdict_pilots(_pl(within=0.95, cost_ratio=0.22, cost=912, blind=0.3), "x", 200, 4096)
    assert kind == "success" and "blind (weniger als 10 Treffer)" in text and "kostet mehr" not in text
    assert ev.verdict_pilots(_pl(within=0.75, cost_ratio=0.5), "x", 200, 4096)[0] == "info"


def test_einzelner_probelauf_verdikt():
    p = dict(n0=1000, pred={"mc": 1.0, "cv": 0.783, "anti": 0.979, "lhs": 0.814, "qmc": 0.145}, blind=False, hits=54.0)
    kind, text = ev.verdict_pilot_single(p, _a(factor={"mc": 1.0, "cv": 0.688, "anti": 0.819, "lhs": 0.725, "qmc": 0.246, "qmc8": 0.377, "is": 0.438}))
    assert kind == "info" and "Kontrollvariate vorhergesagt 0,783, gemessen 0,688" in text and "Sobol (eine Folge) vorhergesagt 0,145, gemessen 0,246" in text
    kind, text = ev.verdict_pilot_single(dict(p, blind=True, hits=0.0), _a())
    assert kind == "warning" and "nur 0,0 Treffer (unter 10)" in text
