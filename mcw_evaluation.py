"""Messungen und Verdikte der Demo: Varianzfaktoren, Ehrlichkeit der Fehlerangabe, Kippschwelle, Probelauf, Weg durch den Baum.

Jede Messung ist eine reine Funktion ihrer Eingaben (Aufgabe, Budget, Strom, Wiederholungen) und zwischengespeichert; die Zahl der Wiederholungen wird erst im öffentlichen Aufruf aus
mcw_constants gelesen (die Tests setzen sie klein). Verdikte sind reine Funktionen der Messergebnisse: jede Zahl im Text stammt aus der Messung, keine feste Aussage über Zahlen.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import mcw_algorithm as A
import mcw_constants as C
from mcw_scenario import tree_steps

HONEST = tuple(m for m in C.METHODS if m != "qmc")     # „ehrliche“ Verfahren: die eine Sobol-Folge hat nur den naiven Balken


# --- Formatierung -----------------------------------------------------------------------------------------------------------------------------

def de(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def pr(x: float) -> str:
    return de(x, 2) if x >= 0.995 or x < 0 else de(x, 3)


def fac(x: float) -> str:
    """Varianzfaktor mit sinnvoller Genauigkeit: ab 1 zwei, ab 0,01 drei, darunter vier Nachkommastellen."""
    if x != x:
        return "–"
    return de(x, 2) if x >= 1 else de(x, 3) if x >= 0.01 else de(x, 4)


def zahl(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def pi_text(x: float) -> str:
    return de(x, 4) if x < 0.1 else de(x, 3)


def n1(x: float) -> str:
    return de(x, 1)


def mlabel(m: str) -> str:
    return C.METHOD_LABELS[m]


# --- Einstellungen ----------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    task: str = C.DEFAULT_TASK
    N: int = C.DEFAULT_N
    n0: int = C.DEFAULT_N0
    info: str = C.DEFAULT_INFO
    netz: str = C.DEFAULT_NETZ
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    task: str
    label: str
    N: int
    R: int
    truth: float
    d: int
    mse: dict           # Verfahren -> mittlerer quadratischer Fehler
    factor: dict        # Verfahren -> Varianzfaktor gegen einfach
    bias: dict
    best: str           # kleinster Faktor überhaupt
    best_honest: str    # kleinster Faktor unter den Verfahren mit ehrlicher Standard-Fehlerangabe
    rec: str            # Empfehlung der Trefferzahl-Regel
    npi: float          # N·π̂ aus dem Probelauf (NaN, wenn die Aufgabe kein Ereignis ist)
    pilot: dict


def _tindex(key: str) -> int:
    return C.TASK_KEYS.index(key)


def _R(R):
    return C.R if R is None else int(R)


# --- Messung 1: Varianzfaktoren -----------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _measure(task_key: str, N: int, seed: int, R: int):
    t = A.make_task(task_key)
    ti = _tindex(task_key)
    mse, bias = {}, {}
    for mi, m in enumerate(C.METHODS):
        if m not in A.methods_for(t):
            continue
        x = np.array([A.estimate(t, m, N, A.stream_seed(seed, 1, ti, N, mi, r)) for r in range(R)])
        mse[m] = float(np.mean((x - t.truth) ** 2))
        bias[m] = float(np.mean(x) - t.truth)
    return mse, bias


def measure(task_key: str, N: int, seed: int, R: int | None = None) -> dict:
    """Je Verfahren: mittlerer quadratischer Fehler gegen die Wahrheit, Faktor gegen einfach, Verzerrung."""
    mse, bias = _measure(task_key, N, seed, _R(R))
    base = mse["mc"]
    return {m: dict(mse=mse[m], factor=mse[m] / base, bias=bias[m]) for m in mse}


def matrix(task_key: str, seed: int, R: int | None = None) -> dict:
    """Messung 1 für alle Budgets: {N: measure}."""
    return {N: measure(task_key, N, seed, R) for N in C.N_CHOICES}


def _best(m: dict, among) -> str:
    return min((k for k in m if k in among), key=lambda k: m[k]["factor"])


@lru_cache(maxsize=None)
def _analyse(settings: Settings, R: int) -> Analysis:
    t = A.make_task(settings.task)
    m = measure(settings.task, settings.N, settings.seed, R)
    p = A.pilot(t, settings.n0, A.stream_seed(settings.seed, 9, _tindex(settings.task), settings.n0))
    rec, npi = A.recommend(t, settings.N, p["pi_hat"])
    return Analysis(settings.task, t.label, settings.N, R, t.truth, t.d, {k: v["mse"] for k, v in m.items()}, {k: v["factor"] for k, v in m.items()}, {k: v["bias"] for k, v in m.items()},
                    _best(m, C.METHODS), _best(m, HONEST), rec, npi, p)


def analyse(settings: Settings, R: int | None = None) -> Analysis:
    return _analyse(settings, _R(R))


# --- Messung 2: ist die Fehlerangabe ehrlich? ---------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cover(task_key: str, N: int, seed: int, R: int):
    t = A.make_task(task_key)
    ti = _tindex(task_key)
    out = {}
    for mi, m in enumerate(C.METHODS):
        if m not in A.methods_for(t):
            continue
        hit, ses, ests = [], [], []
        for r in range(R):
            e, se, zq = A.est_se(t, m, N, A.stream_seed(seed, 2, ti, N, mi, r))
            hit.append(abs(e - t.truth) <= zq * se)
            ses.append(se)
            ests.append(e)
        sd = float(np.std(ests, ddof=1))
        out[m] = dict(cover=float(np.mean(hit)), se_ratio=float(np.mean(ses) / sd) if sd > 0 else float("nan"), sd=sd)
    base = out["mc"]["sd"]
    for m in out:
        out[m]["factor"] = (out[m]["sd"] / base) ** 2 if base > 0 else float("nan")
    return out


def cover(task_key: str, N: int, seed: int, R: int | None = None) -> dict:
    """Je Verfahren: Abdeckung des 95-%-Intervalls, Standardfehler geteilt durch tatsächliche Streuung, Varianzfaktor (aus der Streuung)."""
    return _cover(task_key, N, seed, _R(R))


# --- Messung 3: wo kippt Importance Sampling gegen Sobol in Blöcken? -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _cross(netz: str, N: int, seed: int, R: int):
    rows_, cols_ = C.NETZE[netz]
    tab, m, Nk = A.grid_tab(rows_, cols_)
    Nf = A.failure_counts(tab, m)
    rows = []
    for pi_, p in enumerate(C.P_GRID):
        q = 1.0 - p
        truth = 1.0 - A.r_exact(Nk, p)
        pw = 1 << np.arange(m, dtype=np.int64)
        t = A.Task(f"{netz}_{p}", f"Netz {netz}, p = {p}", m, "sprung", truth,
                   lambda u, pw=pw, tab=tab, p=p: 1.0 - tab[((u < p).astype(np.int64) * pw).sum(axis=1)].astype(np.float64),
                   lambda u, p=p: (u >= p).sum(axis=1).astype(np.float64), m * q, dict(m=m, q=q, qopt=A.optimal_tilt(Nf, q), tab=tab))
        mse = {}
        for mi, meth in enumerate(("mc", "qmc", "qmc8", "is")):
            x = np.array([A.estimate(t, meth, N, A.stream_seed(seed, 3, 1 if netz == "3x4" else 0, pi_, N, mi, r)) for r in range(R)])
            mse[meth] = float(np.mean((x - truth) ** 2))
        fac_ = {k: v / mse["mc"] for k, v in mse.items()}
        rows.append(dict(p=p, pi=truth, npi=N * truth, factor=fac_, best=min(("qmc8", "is"), key=lambda k: fac_[k]), rule="is" if N * truth < C.NPI_THRESHOLD else "qmc8"))
    return rows


def cross(netz: str, N: int, seed: int, R: int | None = None) -> list:
    return _cross(netz, N, seed, _R(R))


# --- Messung 4: der Probelauf -------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _pilots(task_key: str, N: int, n0: int, seed: int, R: int, RP: int):
    t = A.make_task(task_key)
    ti = _tindex(task_key)
    meas = measure(task_key, N, seed, R)
    is_f = meas["is"]["factor"] if "is" in meas else None
    true = {m: meas[m]["factor"] for m in meas if m != "qmc8"}      # wie in der Vormessung: „Sobol“ ist die eine Folge (naiver Balken); die Blockversion zählt hier nicht als eigene Wahl
    best = min(true, key=true.get)
    picks = {m: 0 for m in true}
    regrets, preds = [], {m: [] for m in true if m != "is"}
    blind = 0
    for r in range(RP):
        p = A.pilot(t, n0, A.stream_seed(seed, 4, ti, N, n0, r))
        blind += int(p["blind"])
        c = A.pilot_choice(p, is_f)
        picks[c] += 1
        regrets.append(true[c] / true[best])
        for m in preds:
            preds[m].append(p["pred"][m])
    return dict(picks=picks, regret=float(np.exp(np.mean(np.log(regrets)))), within=float(np.mean(np.array(regrets) <= 1.5)), blind=blind / RP,
                pred={m: float(np.mean(v)) for m, v in preds.items()}, true=true, best=best, cost=2 * n0 + 8 * 64, cost_ratio=(2 * n0 + 8 * 64) / N, RP=RP)


def pilots(task_key: str, N: int, n0: int, seed: int, R: int | None = None, RP: int | None = None) -> dict:
    return _pilots(task_key, N, n0, seed, _R(R), C.R_PILOTS if RP is None else int(RP))


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis, s: Settings) -> tuple[str, str]:
    """Kernfrage „Welche Bauart für diese Aufgabe?“: gemessene Faktoren, die Regel und ihr Abstand zum Besten."""
    f = a.factor
    rec_f = f[a.rec]
    best_f = f[a.best_honest]
    gap = rec_f / best_f
    losers = [m for m in C.METHODS if m in f and f[m] > C.LOSS_FACTOR]
    head = (f"{a.label}, N = {zahl(a.N)} Auswertungen, Wahrheit {de(a.truth, 4)}, {a.R} Wiederholungen. Varianzfaktor gegen einfach (kleiner ist besser): "
            + ", ".join(f"{mlabel(m)} {fac(f[m])}" for m in C.METHODS if m in f) + ". ")
    best_txt = f"Am besten: {mlabel(a.best)} ({fac(f[a.best])})"
    if a.best == "qmc":
        best_txt += f", mit ehrlicher Fehlerangabe {mlabel(a.best_honest)} ({fac(best_f)})"
    best_txt += ". "
    if a.npi == a.npi:
        regel = f"Regel: erwartete Trefferzahl N·π̂ = {n1(a.npi)} (Schwelle {zahl(C.NPI_THRESHOLD)}) → {mlabel(a.rec)}, Faktor {fac(rec_f)}, das {de(gap, 2)}-Fache des Besten mit ehrlichem Balken. "
    else:
        regel = f"Regel: kein seltenes Ereignis gemessen → {mlabel(a.rec)}, Faktor {fac(rec_f)}, das {de(gap, 2)}-Fache des Besten mit ehrlichem Balken. "
    loss_txt = (f"Schlechter als einfach (Faktor über {de(C.LOSS_FACTOR, 2)}): {', '.join(f'{mlabel(m)} {fac(f[m])}' for m in losers)}. " if losers else "")
    kind = "success" if gap <= 1.5 else "warning"
    if losers and kind == "success":
        kind = "info"
    return kind, head + best_txt + regel + loss_txt.rstrip()


def verdict_matrix(mat: dict, label: str, R: int) -> tuple[str, str]:
    """Experiment „Wer gewinnt wo?“: Sieger je Budget und Rangwechsel."""
    parts, winners = [], {}
    for N, m in mat.items():
        b = min(m, key=lambda k: m[k]["factor"])
        winners[N] = b
        parts.append(f"N = {zahl(N)}: {mlabel(b)} ({fac(m[b]['factor'])})")
    flips = len(set(winners.values())) > 1
    txt = f"{label} ({R} Wiederholungen). Sieger je Budget: " + "; ".join(parts) + ". "
    if flips:
        txt += "Die Rangfolge hängt am Budget: der Sieger wechselt. "
    loser = sorted({(mlabel(k), fac(v["factor"])) for N, m in mat.items() for k, v in m.items() if v["factor"] > C.LOSS_FACTOR and k not in ("mc",)})
    if loser:
        txt += "Schlechter als einfach (Faktor über " + de(C.LOSS_FACTOR, 2) + ") in mindestens einem Budget: " + ", ".join(f"{k} {v}" for k, v in loser) + "."
    return ("info" if flips or loser else "success"), txt.rstrip()


def verdict_cover(cv: dict, label: str, N: int, R: int) -> tuple[str, str]:
    """Experiment „Ist der Balken ehrlich?“: Abdeckung und Standardfehler/Streuung je Verfahren."""
    broken = [m for m in cv if cv[m]["cover"] < C.HONEST_COVER]
    wide = [m for m in cv if cv[m]["se_ratio"] > C.WIDE_RATIO]
    txt = (f"{label}, N = {zahl(N)}, {R} Wiederholungen, Intervall zum Niveau 95 %. Abdeckung: " + ", ".join(f"{mlabel(m)} {pr(cv[m]['cover'])}" for m in cv) + ". ")
    if broken:
        txt += f"Unter {de(C.HONEST_COVER, 2)} und damit gebrochen: " + ", ".join(f"{mlabel(m)} {pr(cv[m]['cover'])}" for m in broken) + ". "
    if wide:
        txt += "Zu breite Balken (Standardfehler geteilt durch tatsächliche Streuung über " + de(C.WIDE_RATIO, 1) + "): " + ", ".join(f"{mlabel(m)} {n1(cv[m]['se_ratio'])}" for m in wide) + ". "
    if "qmc" in cv and "qmc8" in cv:
        txt += f"Sobol: eine Folge hat Varianzfaktor {fac(cv['qmc']['factor'])} mit naivem Balken, acht Blöcke {fac(cv['qmc8']['factor'])} mit ehrlichem."
    return ("warning" if broken else "info" if wide else "success"), txt.rstrip()


def verdict_cross(rows: list, netz: str, N: int, R: int) -> tuple[str, str]:
    """Experiment „Wo kippt es?“: Gewinner je Haltewahrscheinlichkeit gegen die Regel N·π̂ < Schwelle."""
    ok = sum(1 for r in rows if r["rule"] == r["best"])
    is_n = [r["npi"] for r in rows if r["best"] == "is"]
    q_n = [r["npi"] for r in rows if r["best"] == "qmc8"]
    txt = f"Netz {netz}, N = {zahl(N)}, {R} Wiederholungen, Fehlerangabe ehrlich (Sobol in 8 Blöcken). Importance Sampling gewinnt in {len(is_n)} von {len(rows)} Zellen"
    txt += f" (größtes N·π: {n1(max(is_n))})" if is_n else ""
    txt += f", Sobol in Blöcken in {len(q_n)}" + (f" (kleinstes N·π: {n1(min(q_n))})" if q_n else "") + ". "
    txt += f"Die Regel N·π < {zahl(C.NPI_THRESHOLD)} trifft in {ok} von {len(rows)} Zellen."
    return ("success" if ok == len(rows) else "info" if ok >= len(rows) - 1 else "warning"), txt


def verdict_pilots(pl: dict, label: str, n0: int, N: int) -> tuple[str, str]:
    """Experiment „Der Probelauf“: Wahlverteilung, Abstand zum Besten, Kosten gegen Endbudget."""
    top = max(pl["picks"], key=pl["picks"].get)
    txt = (f"{label}, Probelauf n₀ = {zahl(n0)} (Kosten {zahl(pl['cost'])} Auswertungen = {de(pl['cost_ratio'], 2)} × Endbudget N = {zahl(N)}), {pl['RP']} Wiederholungen der Wahl. "
           f"Gewählt: " + ", ".join(f"{mlabel(m)} {zahl(100 * c / pl['RP'])} %" for m, c in pl["picks"].items() if c) + f". Bestes Verfahren: {mlabel(pl['best'])}. "
           f"Innerhalb des 1,5-Fachen des Besten: {zahl(100 * pl['within'])} % der Wahlen, mittleres Verhältnis zum Besten {de(pl['regret'], 2)}. ")
    if pl["blind"] > 0:
        txt += f"Der Probelauf war in {zahl(100 * pl['blind'])} % der Läufe blind (weniger als {C.RARE_HITS} Treffer). "
    if pl["cost_ratio"] >= 1:
        txt += "Der Probelauf kostet mehr als der Endlauf."
    return ("success" if pl["within"] >= 0.9 and pl["cost_ratio"] < 1 else "info" if pl["within"] >= 0.7 else "warning"), txt.rstrip()


def verdict_pilot_single(p: dict, a: Analysis) -> tuple[str, str]:
    """Spielwiese: ein einzelner Probelauf, seine Vorhersage gegen die Messung."""
    pr_ = p["pred"]
    parts = []
    for m in ("cv", "anti", "lhs", "qmc"):
        meas = a.factor["qmc" if m == "qmc" else m]
        parts.append(f"{mlabel(m)} vorhergesagt {fac(pr_[m])}, gemessen {fac(meas)}")
    txt = f"Probelauf n₀ = {zahl(p['n0'])}: " + "; ".join(parts) + ". "
    if p["blind"]:
        txt += f"Der Probelauf sieht nur {n1(p['hits'])} Treffer (unter {C.RARE_HITS}) und ist für alle Varianzschätzungen blind."
    return ("warning" if p["blind"] else "info"), txt.rstrip()


# --- Weg durch den Baum -----------------------------------------------------------------------------------------------------------------------

def tree_path(s: Settings, a: Analysis) -> list[dict]:
    """Die Fragen des Baums bis zur ersten, die für die gewählte Informationslage „Ja“ ist; bei „nur der Schätzer“ bis Frage 5 mit der gemessenen Empfehlung."""
    steps = tree_steps(s.info)
    if s.info == "schaetzer":
        f = a.factor
        npi_txt = f"N·π̂ = {n1(a.npi)}" if a.npi == a.npi else "kein seltenes Ereignis"
        steps = steps + [dict(nr=5, frage="Der Schätzer selbst: erwartete Trefferzahl unter der Schwelle?", antwort="Ja" if a.rec == "is" else "Nein",
                              folge=f"{mlabel(a.rec)} ({npi_txt}, Schwelle {zahl(C.NPI_THRESHOLD)}); gemessener Faktor {fac(f[a.rec])}, bester ehrlicher {mlabel(a.best_honest)} {fac(f[a.best_honest])}",
                              verweis="Stück 7 und 8 (Importance Sampling), Stück 9 (Sobol)", zahl_aus="dieses Stück, gemessen"),
                         dict(nr=6, frage="Ist die Fehlerangabe ehrlich?", antwort="siehe Experiment 2", folge="Seltenes Ereignis und schwerer Schwanz brechen die Fehlerangabe aller Verfahren außer (bei seltenem Ereignis) Importance Sampling.",
                              verweis="Experiment 2", zahl_aus="dieses Stück, gemessen")]
    return steps
