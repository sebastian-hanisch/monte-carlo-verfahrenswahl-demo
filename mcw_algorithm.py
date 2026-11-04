"""Kern der Demo (numpy und Standardbibliothek, kein scipy zur Laufzeit).

Stück 16 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": die ENTSCHEIDUNG zwischen den Schätzer-Bauarten wird gemessen. Jede Regel ist eine eigene Funktion
mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom    stream_seed, uniforms          SplitMix64, zählerbasiert
  Punktmengen     points (mc, lhs, sobol_owen)    Sobol: Joe-Kuo-Richtungszahlen (64 Dimensionen eingebettet), Owen-Scrambling (aus Stück 9 übernommen)
  Netze           connected, conn_table, conn_counts, r_exact    Zusammenhang aller 2^m Kantenmengen (Aufzählung), exakte Wahrheit
  Aufgaben        make_task                      elf Aufgaben auf dem Einheitswürfel mit exakter Wahrheit
  Schätzer        estimate, est_se               einfach, antithetisch, Kontrollvariate, Latin Hypercube, Sobol (Folge, Blöcke), Importance Sampling
  Probelauf       pilot, pilot_choice            Diagnosen aus einer kleinen einfachen Stichprobe
  Regel           recommend                      Trefferzahl N·π̂ unter der Schwelle: Importance Sampling, sonst Sobol in Blöcken
Zufall kommt nur über einen übergebenen Strom, nie aus einem globalen Generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable

import numpy as np

import mcw_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
BITS = C.BITS


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


# --- Sobol-Folge --------------------------------------------------------------------------------------------------------------------------
# Richtungszahlen der Koordinaten 2 bis 64 nach Joe und Kuo (2008, Tabelle new-joe-kuo-6.21201): (Polynom als Bitmuster x^s + a_1 x^(s-1) + ... + 1, Anfangswerte m_1 ... m_s).
# Koordinate 1 ist die van-der-Corput-Folge. Die Tabelle ist ein Auszug aus der Tabelle in scipy und im Test ganzzahlig gegen scipy.stats.qmc.Sobol geprüft.

JOE_KUO = (
    (3, (1,)),
    (7, (1, 3)),
    (11, (1, 3, 1)),
    (13, (1, 1, 1)),
    (19, (1, 1, 3, 3)),
    (25, (1, 3, 5, 13)),
    (37, (1, 1, 5, 5, 17)),
    (41, (1, 1, 5, 5, 5)),
    (47, (1, 1, 7, 11, 19)),
    (55, (1, 1, 5, 1, 1)),
    (59, (1, 1, 1, 3, 11)),
    (61, (1, 3, 5, 5, 31)),
    (67, (1, 3, 3, 9, 7, 49)),
    (91, (1, 1, 1, 15, 21, 21)),
    (97, (1, 3, 1, 13, 27, 49)),
    (103, (1, 1, 1, 15, 7, 5)),
    (109, (1, 3, 1, 15, 13, 25)),
    (115, (1, 1, 5, 5, 19, 61)),
    (131, (1, 3, 7, 11, 23, 15, 103)),
    (137, (1, 3, 7, 13, 13, 15, 69)),
    (143, (1, 1, 3, 13, 7, 35, 63)),
    (145, (1, 3, 5, 9, 1, 25, 53)),
    (157, (1, 3, 1, 13, 9, 35, 107)),
    (167, (1, 3, 1, 5, 27, 61, 31)),
    (171, (1, 1, 5, 11, 19, 41, 61)),
    (185, (1, 3, 5, 3, 3, 13, 69)),
    (191, (1, 1, 7, 13, 1, 19, 1)),
    (193, (1, 3, 7, 5, 13, 19, 59)),
    (203, (1, 1, 3, 9, 25, 29, 41)),
    (211, (1, 3, 5, 13, 23, 1, 55)),
    (213, (1, 3, 7, 3, 13, 59, 17)),
    (229, (1, 3, 1, 3, 5, 53, 69)),
    (239, (1, 1, 5, 5, 23, 33, 13)),
    (241, (1, 1, 7, 7, 1, 61, 123)),
    (247, (1, 1, 7, 9, 13, 61, 49)),
    (253, (1, 3, 3, 5, 3, 55, 33)),
    (285, (1, 3, 1, 15, 31, 13, 49, 245)),
    (299, (1, 3, 5, 15, 31, 59, 63, 97)),
    (301, (1, 3, 1, 11, 11, 11, 77, 249)),
    (333, (1, 3, 1, 11, 27, 43, 71, 9)),
    (351, (1, 1, 7, 15, 21, 11, 81, 45)),
    (355, (1, 3, 7, 3, 25, 31, 65, 79)),
    (357, (1, 3, 1, 1, 19, 11, 3, 205)),
    (361, (1, 1, 5, 9, 19, 21, 29, 157)),
    (369, (1, 3, 7, 11, 1, 33, 89, 185)),
    (391, (1, 3, 3, 3, 15, 9, 79, 71)),
    (397, (1, 3, 7, 11, 15, 39, 119, 27)),
    (425, (1, 1, 3, 1, 11, 31, 97, 225)),
    (451, (1, 1, 1, 3, 23, 43, 57, 177)),
    (463, (1, 3, 7, 7, 17, 17, 37, 71)),
    (487, (1, 3, 1, 5, 27, 63, 123, 213)),
    (501, (1, 1, 3, 5, 11, 43, 53, 133)),
    (529, (1, 3, 5, 5, 29, 17, 47, 173, 479)),
    (539, (1, 3, 3, 11, 3, 1, 109, 9, 69)),
    (545, (1, 1, 1, 5, 17, 39, 23, 5, 343)),
    (557, (1, 3, 1, 5, 25, 15, 31, 103, 499)),
    (563, (1, 1, 1, 11, 11, 17, 63, 105, 183)),
    (601, (1, 1, 5, 11, 9, 29, 97, 231, 363)),
    (607, (1, 1, 5, 15, 19, 45, 41, 7, 383)),
    (617, (1, 3, 7, 7, 31, 19, 83, 137, 221)),
    (623, (1, 1, 1, 3, 23, 15, 111, 223, 83)),
    (631, (1, 1, 5, 13, 31, 15, 55, 25, 161)),
    (637, (1, 1, 3, 13, 25, 47, 39, 87, 257)),
)


@lru_cache(maxsize=None)
def direction_ints(d: int) -> np.ndarray:
    """(BITS, d) uint64: V[k, j] = Richtungszahl k+1 der Koordinate j, links ausgerichtet (m_k * 2^(BITS - k)). Zwischengespeichert, schreibgeschützt."""
    if not 1 <= d <= C.MAX_DIM:
        raise ValueError(f"Dimension {d}: eingebettet sind 1 bis {C.MAX_DIM}")
    V = np.zeros((BITS, d), dtype=np.uint64)
    for j in range(d):
        if j == 0:
            for k in range(BITS):
                V[k, 0] = 1 << (BITS - 1 - k)
            continue
        poly, init = JOE_KUO[j - 1]
        s = poly.bit_length() - 1
        a = [(poly >> (s - 1 - i)) & 1 for i in range(s - 1)]     # a_1 ... a_{s-1}
        m = list(init)
        for k in range(s, BITS):
            new = m[k - s] ^ (m[k - s] << s)
            for i in range(1, s):
                if a[i - 1]:
                    new ^= m[k - i] << i
            m.append(new)
        for k in range(BITS):
            V[k, j] = m[k] << (BITS - 1 - k)
    V.setflags(write=False)
    return V


def sobol_ints(N: int, d: int) -> np.ndarray:
    """(N, d) uint64 der ersten N Sobol-Punkte in Gray-Code-Reihenfolge, links ausgerichtet (Wert = x / 2^BITS)."""
    if N < 1:
        raise ValueError("N muss mindestens 1 sein")
    V = direction_ints(d)
    idx = np.arange(N, dtype=np.uint64)
    gray = idx ^ (idx >> np.uint64(1))
    out = np.zeros((N, d), dtype=np.uint64)
    for k in range(max(1, int(N - 1).bit_length())):
        bit = ((gray >> np.uint64(k)) & np.uint64(1)).astype(bool)
        out[bit] ^= V[k][None, :]
    return out


# --- Randomisierung -----------------------------------------------------------------------------------------------------------------------

def scramble_depth(N: int) -> int:
    """Tiefe, bis zu der gescrambelt werden muss: ceil(log2 N), mindestens 1 (ab dort sind die Präfixe einer Koordinate paarweise verschieden)."""
    return max(1, int(N - 1).bit_length())


def digital_shift(x: np.ndarray, seed: int) -> np.ndarray:
    """Zufällige digitale Verschiebung: XOR mit einer BITS-Bit-Zufallszahl je Koordinate."""
    r = (uniforms(seed, x.shape[1]) * float(1 << BITS)).astype(np.uint64)
    return x ^ r[None, :]


def owen_scramble(x: np.ndarray, seed: int, depth: int) -> np.ndarray:
    """Verschachteltes Owen-Scrambling der obersten `depth` Bits: Bit j (von oben) wird mit einem Bit gekippt, das aus (Koordinate, j, Präfix der Bits darüber) per Mischfunktion kommt.

    Rückgabe: die obersten `depth` Bits als Ganzzahl (Wert = Ergebnis / 2^depth)."""
    d = x.shape[1]
    out = np.zeros_like(x)
    col = (np.arange(d, dtype=np.uint64) + np.uint64(1))[None, :] * np.uint64(0xD1342543DE82EF95)
    base = np.uint64(seed)
    for j in range(depth):
        prefix = (x >> np.uint64(BITS - j)) if j > 0 else np.zeros_like(x)
        with np.errstate(over="ignore"):
            key = _mix_arr(base + col + np.uint64(j + 1) * np.uint64(GAMMA) + _mix_arr(prefix + np.uint64(1) + (np.uint64(j) << np.uint64(40))))
        flip = (key >> np.uint64(63)) & np.uint64(1)
        bit = (x >> np.uint64(BITS - 1 - j)) & np.uint64(1)
        out |= (bit ^ flip) << np.uint64(depth - 1 - j)
    return out


def lhs_points(seed: int, N: int, d: int) -> np.ndarray:
    """Latin Hypercube: erste N*d Zahlen = Zitter, zweite N*d Zahlen sortiert = Permutationen je Spalte (wie in Stück 5)."""
    v = uniforms(seed, 2 * N * d)
    jitter = v[: N * d].reshape(N, d)
    perm = np.argsort(v[N * d:].reshape(N, d), axis=0)
    return (perm + jitter) / N


POINT_METHODS = ("mc", "lhs", "sobol_ds", "sobol_owen")


def points(method: str, N: int, d: int, seed: int) -> np.ndarray:
    """(N, d) Punkte aus [0,1)^d: 'mc' (unabhängig), 'lhs', 'sobol_roh' (nicht randomisiert, Zellmitte), 'sobol_ds' (digitale Verschiebung), 'sobol_owen'."""
    if method == "mc":
        return uniforms(seed, N * d).reshape(N, d)
    if method == "lhs":
        return lhs_points(seed, N, d)
    x = sobol_ints(N, d)
    if method == "sobol_roh":
        return (x.astype(np.float64) + 0.5) / float(1 << BITS)
    fill = uniforms(stream_seed(seed, 1), N * d).reshape(N, d)
    if method == "sobol_ds":
        return (digital_shift(x, stream_seed(seed, 2)).astype(np.float64) + fill) / float(1 << BITS)
    if method == "sobol_owen":
        depth = scramble_depth(N)
        return (owen_scramble(x, stream_seed(seed, 3), depth).astype(np.float64) + fill) / float(1 << depth)
    raise ValueError(f"unbekannte Methode {method}")


# --- Netze: Zusammenhang, Aufzählung ------------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile (`present` ist (B, m) bool): Kleinste-Marke-Ausbreitung, zusammenhängend genau dann, wenn alle Marken 0 sind."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def conn_table(n: int, edges, chunk: int = 1 << 15) -> np.ndarray:
    """Zusammenhang für alle 2^m Kantenmengen in Maskenreihenfolge (Kante e = Bit e)."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    tab = np.empty(1 << m, dtype=bool)
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        tab[start:start + len(masks)] = connected(((masks[:, None] >> shifts) & 1).astype(bool), n, edges)
    return tab


def popcount(m: int) -> np.ndarray:
    """Zahl der gesetzten Bits für alle Masken 0 ... 2^m - 1."""
    k = np.zeros(1 << m, dtype=np.int64)
    masks = np.arange(1 << m, dtype=np.int64)
    for e in range(m):
        k += (masks >> e) & 1
    return k


def conn_counts(tab: np.ndarray, m: int) -> list[int]:
    """N_k = Zahl der zusammenhängenden Kantenmengen mit k Kanten (k = 0 ... m)."""
    return np.bincount(popcount(m)[tab], minlength=m + 1).astype(np.int64).tolist()


def r_exact(Nk, p: float) -> float:
    """Zuverlässigkeit R(p) = Σ N_k p^k (1-p)^(m-k) bei Haltewahrscheinlichkeit p je Kante."""
    m = len(Nk) - 1
    return float(sum(c * p**k * (1 - p) ** (m - k) for k, c in enumerate(Nk)))


# --- Integranden mit exakter Wahrheit -----------------------------------------------------------------------------------------------------

def f_sprung(u: np.ndarray, p: float, tab: np.ndarray) -> np.ndarray:
    """1, wenn das Netz mit den Kanten {u_e < p} zusammenhängt, sonst 0."""
    m = u.shape[1]
    mask = ((u < p).astype(np.int64) << np.arange(m, dtype=np.int64)).sum(axis=1)
    return tab[mask].astype(np.float64)


def f_engpass(u: np.ndarray, tab: np.ndarray) -> np.ndarray:
    """Größtes t, bei dem das Netz mit den Kanten {u_e >= t} zusammenhängt (stetig, stückweise linear)."""
    N, m = u.shape
    order = np.argsort(-u, axis=1)
    us = np.take_along_axis(u, order, axis=1)
    cum = np.cumsum(np.left_shift(1, order.astype(np.int64)), axis=1)
    first = tab[cum].argmax(axis=1)
    return us[np.arange(N), first]


def engpass_exact(Nk) -> float:
    """E B = ∫ R(p) dp = Σ_k N_k k!(m-k)!/(m+1)!."""
    m = len(Nk) - 1
    return float(sum(c * math.factorial(k) * math.factorial(m - k) / math.factorial(m + 1) for k, c in enumerate(Nk)))
def f_exp(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.exp(u @ a)


def exp_exact(a: np.ndarray) -> float:
    return float(np.prod(np.expm1(a) / a))



# --- Normalverteilung ---------------------------------------------------------------------------------------------------------------------

_ACK_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_ACK_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_ACK_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_ACK_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def ndtri_array(p: np.ndarray) -> np.ndarray:
    """Umkehrfunktion der Standardnormalverteilung für ein Feld in (0, 1) (Acklam, relative Genauigkeit etwa 1e-9; für Ziehungen weit genug)."""
    p = np.asarray(p, dtype=np.float64)
    a, b, c, d = _ACK_A, _ACK_B, _ACK_C, _ACK_D
    x = np.empty_like(p)
    lo = p < 0.02425
    hi = p > 1 - 0.02425
    mid = ~(lo | hi)
    q = p[mid] - 0.5
    r = q * q
    x[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return x


# --- Irwin-Hall: Summe gleichverteilter Zahlen, exakt -------------------------------------------------------------------------------------

def irwin_hall_sf(d: int, x: float) -> float:
    """P(Σ_{i<=d} Uᵢ > x) durch die alternierende Summe, mit Brüchen gerechnet (für d = 32 stabil)."""
    xf = Fraction(x).limit_denominator(10**9)
    s = Fraction(0)
    for k in range(0, int(math.floor(xf)) + 1):
        s += (-1) ** k * math.comb(d, k) * (xf - k) ** d
    return float(1 - s / math.factorial(d))


def irwin_hall_pos_part(d: int) -> float:
    """E[(S - d/2)⁺] mit S Irwin-Hall(d): ∫ P(S > x) dx von d/2 bis d, stückweise exakt (Brüche)."""
    c = Fraction(d, 2)
    edges = [c] + [Fraction(p) for p in range(int(math.ceil(c)), d + 1) if Fraction(p) > c]
    if edges[-1] != d:
        edges.append(Fraction(d))
    total = Fraction(0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        kmax = int(math.floor(lo))

        def G(x, kmax=kmax):
            return sum((-1) ** k * math.comb(d, k) * (x - k) ** (d + 1) / (d + 1) for k in range(0, kmax + 1)) / math.factorial(d)
        total += (hi - lo) - (G(hi) - G(lo))
    return float(total)


# --- Aufgaben ---------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """Eine Aufgabe auf [0,1]^d mit exakter Wahrheit, einer Kontrollvariate mit bekanntem Mittel und (nur Netz-Ausfälle) einem Importance-Sampling-Ansatz."""
    key: str
    label: str
    d: int
    kind: str
    truth: float
    f: Callable = None
    control: Callable = None
    control_mean: float = 0.0
    net: dict | None = None            # m, q, qopt, tab (nur Netz-Ausfälle)


def grid_edges(rows: int, cols: int) -> tuple[int, list[tuple[int, int]]]:
    """Gitter rows x cols: Knotenzahl und Kantenliste (waagerecht, dann senkrecht je Zeile)."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    return rows * cols, edges


@lru_cache(maxsize=None)
def grid_tab(rows: int, cols: int):
    """Zusammenhangstabelle, Kantenzahl und Zählungen N_k des Gitternetzes (zwischengespeichert)."""
    n, edges = grid_edges(rows, cols)
    tab = conn_table(n, edges)
    tab.setflags(write=False)
    return tab, len(edges), conn_counts(tab, len(edges))


def failure_counts(tab: np.ndarray, m: int) -> np.ndarray:
    """N_f: Zahl der Kantenmengen mit f AUSGEFALLENEN Kanten, bei denen das Netz NICHT zusammenhängt (f = 0 ... m)."""
    pc = popcount(m)
    return np.bincount((m - pc)[~tab], minlength=m + 1).astype(np.float64)


def is_second_moment(Nf: np.ndarray, q: float, qp) -> np.ndarray:
    """Exaktes relatives Varianzquadrat CV² = M2/P² - 1 des Importance-Sampling-Schätzers je Hilfsverteilung q' (jede Kante fällt mit q' aus)."""
    m = len(Nf) - 1
    f = np.arange(m + 1)
    pq = q**f * (1 - q) ** (m - f)
    P = float(np.sum(Nf * pq))
    qp = np.atleast_1d(np.asarray(qp, dtype=np.float64))
    pqp = qp[:, None] ** f[None, :] * (1 - qp[:, None]) ** (m - f)[None, :]
    M2 = (Nf[None, :] * pq[None, :] ** 2 / pqp).sum(axis=1)
    return M2 / P**2 - 1.0


def optimal_tilt(Nf: np.ndarray, q: float) -> float:
    """Hilfsverteilung q' mit kleinster Varianz (Gittersuche, dann Feinsuche): in der Demo als bekannt angenommen (sie zu LERNEN ist Stück 8)."""
    grid = np.concatenate([np.geomspace(1e-4, 0.5, 600), np.linspace(0.5, 0.999, 300)])
    best = grid[int(np.argmin(is_second_moment(Nf, q, grid)))]
    g2 = np.linspace(best * 0.7, min(best * 1.4, 0.999), 400)
    return float(g2[int(np.argmin(is_second_moment(Nf, q, g2)))])


@lru_cache(maxsize=None)
def make_task(key: str) -> Task:
    if key not in C.TASKS:
        raise ValueError(f"unbekannte Aufgabe {key}")
    spec = C.TASKS[key]
    kind = spec["kind"]
    if kind == "netz":
        tab, m, Nk = grid_tab(spec["rows"], spec["cols"])
        p = spec["p"]
        q = 1.0 - p
        pw = 1 << np.arange(m, dtype=np.int64)
        truth = 1.0 - r_exact(Nk, p)

        def f(u, pw=pw, tab=tab, p=p):                       # 1 = Netz fällt aus; Kante hält, wenn u < p
            return 1.0 - tab[((u < p).astype(np.int64) * pw).sum(axis=1)].astype(np.float64)

        def control(u, p=p):                                 # Zahl der ausgefallenen Kanten, Mittel m·q
            return (u >= p).sum(axis=1).astype(np.float64)
        net = dict(m=m, q=q, qopt=optimal_tilt(failure_counts(tab, m), q), tab=tab)
        return Task(key, spec["label"], m, "sprung", truth, f, control, m * q, net)
    if kind == "engpass":
        tab, m, Nk = grid_tab(spec["rows"], spec["cols"])
        return Task(key, spec["label"], m, "knick", engpass_exact(Nk), lambda u, tab=tab: f_engpass(u, tab), lambda u: u.sum(axis=1), m / 2.0)
    if kind == "glatt":
        d = spec["d"]
        a = (2.0 / np.arange(1, d + 1)) if spec["decay"] else np.ones(d)
        return Task(key, spec["label"], d, "glatt", exp_exact(a), lambda u, a=a: f_exp(u, a), lambda u: u.sum(axis=1), d / 2.0)
    if kind == "symmetrisch":
        d = spec["d"]
        return Task(key, spec["label"], d, "symmetrisch", d / 12.0, lambda u: (u - 0.5).sum(axis=1) ** 2, lambda u: ((u - 0.5) ** 2).sum(axis=1), d / 12.0)
    if kind == "schwanz":
        s = spec["sigma"]
        return Task(key, spec["label"], 1, "schwanz", math.exp(s * s / 2), lambda u, s=s: np.exp(s * ndtri_array(u[:, 0])), lambda u: ndtri_array(u[:, 0]), 0.0)
    if kind == "produkt":
        d = spec["d"]
        return Task(key, spec["label"], d, "wechselwirkung", 1.0, lambda u: np.prod(2.0 * u, axis=1), lambda u: u.sum(axis=1), d / 2.0)
    if kind == "sprung_hoch":
        d = spec["d"]
        thr = d / 2.0 + 1.6449 * math.sqrt(d / 12.0)
        return Task(key, spec["label"], d, "sprung_hoch", irwin_hall_sf(d, thr), lambda u, thr=thr: (u.sum(axis=1) > thr).astype(np.float64), lambda u: u.sum(axis=1), d / 2.0)
    if kind == "knick_hoch":
        d = spec["d"]
        return Task(key, spec["label"], d, "knick_hoch", irwin_hall_pos_part(d), lambda u, d=d: np.maximum(u.sum(axis=1) - d / 2.0, 0.0), lambda u: u.sum(axis=1), d / 2.0)
    raise ValueError(f"unbekannte Aufgabe {key}")


# --- Schätzer ---------------------------------------------------------------------------------------------------------------------------------

def estimate(t: Task, method: str, N: int, seed: int) -> float:
    """Ein Schätzwert mit genau N Auswertungen von f. Verfahren: mc, anti, cv, lhs, qmc (eine Sobol-Folge), qmc8 (acht Blöcke zu N/8), is (nur Netz-Ausfälle)."""
    return est_se(t, method, N, seed)[0]


def est_se(t: Task, method: str, N: int, seed: int) -> tuple[float, float, float]:
    """(Schätzwert, Standardfehler, Quantil) mit der Fehlerangabe, die man dem Verfahren üblicherweise gibt (Messung „Ist der Balken ehrlich?“)."""
    d = t.d
    if method == "mc":
        y = t.f(points("mc", N, d, seed))
        return float(y.mean()), float(y.std(ddof=1) / math.sqrt(N)), C.Z95
    if method == "lhs":
        y = t.f(points("lhs", N, d, seed))
        return float(y.mean()), float(y.std(ddof=1) / math.sqrt(N)), C.Z95
    if method == "qmc":
        y = t.f(points("sobol_owen", N, d, seed))
        return float(y.mean()), float(y.std(ddof=1) / math.sqrt(N)), C.Z95
    if method == "qmc8":
        if N % 8:
            raise ValueError("Sobol in Blöcken braucht N durch 8 teilbar")
        bl = np.array([t.f(points("sobol_owen", N // 8, d, stream_seed(seed, b))).mean() for b in range(8)])
        return float(bl.mean()), float(bl.std(ddof=1) / math.sqrt(8)), C.T975[8]
    if method == "anti":
        u = points("mc", N // 2, d, seed)
        y = 0.5 * (t.f(u) + t.f(1.0 - u))
        return float(y.mean()), float(y.std(ddof=1) / math.sqrt(N // 2)), C.Z95
    if method == "cv":
        u = points("mc", N, d, seed)
        y, c = t.f(u), t.control(u)
        cc = c - c.mean()
        var = float((cc * cc).mean())
        beta = float((cc * (y - y.mean())).mean() / var) if var > 0 else 0.0
        r = y - beta * c
        return float(r.mean() + beta * t.control_mean), float(r.std(ddof=1) / math.sqrt(N)), C.Z95
    if method == "is":
        if t.net is None:
            raise ValueError("Importance Sampling gibt es nur für die Netz-Ausfälle")
        m, q, qp = t.net["m"], t.net["q"], t.net["qopt"]
        fail = points("mc", N, m, seed) < qp
        k = fail.sum(axis=1)
        mask = ((~fail).astype(np.int64) * (1 << np.arange(m, dtype=np.int64))).sum(axis=1)
        w = (1.0 - t.net["tab"][mask].astype(np.float64)) * (q / qp) ** k * ((1 - q) / (1 - qp)) ** (m - k)
        return float(w.mean()), float(w.std(ddof=1) / math.sqrt(N)), C.Z95
    raise ValueError(f"unbekanntes Verfahren {method}")


def methods_for(t: Task) -> tuple[str, ...]:
    return tuple(m for m in C.METHODS if m != "is" or t.net is not None)


# --- Probelauf ----------------------------------------------------------------------------------------------------------------------------------

def additive_share(u: np.ndarray, y: np.ndarray, K: int = 8) -> float:
    """Geschätzter Anteil der additiven Haupteffekte an der Varianz: Summe der um das Binrauschen bereinigten Varianzen der Binmittel je Koordinate."""
    n, d = u.shape
    v = float(y.var(ddof=1))
    if v <= 0:
        return 0.0
    idx = np.minimum((u * K).astype(int), K - 1)
    tot = 0.0
    for j in range(d):
        cnt = np.bincount(idx[:, j], minlength=K)
        s = np.bincount(idx[:, j], weights=y, minlength=K)
        ok = cnt > 1
        mean = s[ok] / cnt[ok]
        w = cnt[ok] / cnt[ok].sum()
        between = float(np.sum(w * (mean - np.sum(w * mean)) ** 2))
        noise = float(v * (ok.sum() - 1) / n)
        tot += max(between - noise, 0.0)
    return min(tot / v, 1.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def pilot(t: Task, n0: int, seed: int) -> dict:
    """Probelauf aus n0 einfachen Auswertungen, n0/2 antithetischen Paaren und acht Sobol-Wiederholungen zu 64 Punkten.

    Vorhersage des Varianzfaktors: Kontrollvariate 1 - rho², antithetisch 1 + rho_a (gleiche Kosten), Latin Hypercube 1 - additiver Anteil, Sobol Varianz der Wiederholungen / (sigma²/64).
    Kosten: 2·n0 + 8·64 Auswertungen. `hits`: Summe der Werte (bei Netz-Ausfällen die Trefferzahl); `blind`: weniger als RARE_HITS Treffer."""
    u = points("mc", n0, t.d, seed)
    y = t.f(u)
    v = float(y.var(ddof=1))
    rho = _corr(y, t.control(u))
    ua = points("mc", n0 // 2, t.d, stream_seed(seed, 1))
    rho_a = _corr(t.f(ua), t.f(1.0 - ua))
    hits = float(y.sum()) if t.net is not None else None
    reps = [float(t.f(points("sobol_owen", 64, t.d, stream_seed(seed, 2, r))).mean()) for r in range(8)]
    add = additive_share(u, y)
    pred = {"mc": 1.0, "cv": 1.0 - rho * rho, "anti": 1.0 + rho_a, "lhs": max(1.0 - add, 0.01), "qmc": float(np.var(reps, ddof=1)) / (v / 64) if v > 0 else 1.0}
    blind = hits is not None and hits < C.RARE_HITS
    return dict(n0=n0, rho=rho, rho_a=rho_a, hits=hits, pi_hat=(hits / n0 if hits is not None else None), v=v, add=add, pred=pred, blind=blind, cost=2 * n0 + 8 * 64)


def pilot_choice(p: dict, is_factor: float | None) -> str:
    """Wahl nach dem Probelauf: kleinster vorhergesagter Faktor; bei blindem Probelauf Rückfall auf Importance Sampling (Netz-Aufgabe) oder einfach.
    `is_factor`: gemessener Faktor des Importance Sampling (Hilfsverteilung als bekannt angenommen), None wenn es das Verfahren für die Aufgabe nicht gibt."""
    est = dict(p["pred"])
    if is_factor is not None:
        est["is"] = is_factor
    if p["blind"]:
        est = {"mc": 1.0, **({"is": is_factor} if is_factor is not None else {})}
    return min(est, key=est.get)


# --- Regel -----------------------------------------------------------------------------------------------------------------------------------

def recommend(t: Task, N: int, pi_hat: float | None) -> tuple[str, float]:
    """Trefferzahl-Regel: bei einer Netz-Aufgabe (Importance Sampling verfügbar) und N·π̂ < NPI_THRESHOLD: 'is', sonst 'qmc8'. Rückgabe (Verfahren, N·π̂); ohne Schätzung N·π̂ = NaN."""
    npi = float("nan") if pi_hat is None else N * pi_hat
    if t.net is not None and pi_hat is not None and npi < C.NPI_THRESHOLD:
        return "is", npi
    return "qmc8", npi
