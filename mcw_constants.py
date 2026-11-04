"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

BITS = 32                                  # Stellen der Sobol-Ganzzahlen
MAX_DIM = 64                               # eingebettete Joe-Kuo-Richtungszahlen reichen bis hierher
Z95 = 1.959963984540054
T975 = {4: 3.182446, 8: 2.364624, 16: 2.131450, 32: 2.039513}   # t-Quantil 0,975 mit K-1 Freiheitsgraden (K Blöcke)

# --- Aufgaben (elf, alle auf dem Einheitswürfel mit exakter Wahrheit) ------------------------------------------------------------------------
TASKS = {
    "netz33_p90": {"kind": "netz", "rows": 3, "cols": 3, "p": 0.9, "label": "Netz 3×3, p = 0,9: Ausfall (seltenes Ereignis, π ≈ 0,053)"},
    "netz33_p99": {"kind": "netz", "rows": 3, "cols": 3, "p": 0.99, "label": "Netz 3×3, p = 0,99: Ausfall (sehr selten, π ≈ 0,0004)"},
    "netz34_p95": {"kind": "netz", "rows": 3, "cols": 4, "p": 0.95, "label": "Netz 3×4, p = 0,95: Ausfall (selten, π ≈ 0,012)"},
    "engpass33": {"kind": "engpass", "rows": 3, "cols": 3, "label": "Netz 3×3: Engpass-Schwelle (stetig, mit Knick)"},
    "glatt12": {"kind": "glatt", "d": 12, "decay": False, "label": "glatt: exp(Σ aᵢ uᵢ), d = 12"},
    "glatt64": {"kind": "glatt", "d": 64, "decay": True, "label": "glatt: exp(Σ aᵢ uᵢ), d = 64, abfallende Gewichte"},
    "symmetrisch12": {"kind": "symmetrisch", "d": 12, "label": "symmetrisch: (Σ(uᵢ − ½))², d = 12"},
    "schwanz2": {"kind": "schwanz", "sigma": 2.0, "label": "schwerer Schwanz: exp(2z), d = 1"},
    "produkt8": {"kind": "produkt", "d": 8, "label": "Wechselwirkung: ∏ 2uᵢ, d = 8"},
    "sprung32": {"kind": "sprung_hoch", "d": 32, "label": "rau, hochdimensional: 1[Σ uᵢ > Schwelle], d = 32"},
    "knick32": {"kind": "knick_hoch", "d": 32, "label": "Knick, hochdimensional: (Σ uᵢ − d/2)⁺, d = 32"},
}
TASK_KEYS = tuple(TASKS)
DEFAULT_TASK = "netz33_p90"

# --- Verfahren --------------------------------------------------------------------------------------------------------------------------------
METHODS = ("mc", "anti", "cv", "lhs", "qmc", "qmc8", "is")
METHOD_LABELS = {"mc": "einfach", "anti": "antithetisch", "cv": "Kontrollvariate", "lhs": "Latin Hypercube", "qmc": "Sobol (eine Folge)", "qmc8": "Sobol (8 Blöcke)", "is": "Importance Sampling"}
METHOD_COLORS = {"mc": "#222222", "anti": "#8E6BBF", "cv": "#E08E0B", "lhs": "#4c78a8", "qmc": "#7FB5A8", "qmc8": "#2F6B65", "is": "#C0392B"}
COLOR_BAD = "#C0392B"
COLOR_OK = "#2F6B65"

# --- Regler -----------------------------------------------------------------------------------------------------------------------------------
N_CHOICES = (256, 1024, 4096)              # Budget: Zahl der Auswertungen je Schätzer (durch 8 teilbar, Zweierpotenz)
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 1024
N0_CHOICES = (200, 1000)                   # Größe des Probelaufs
N0_MIN, N0_MAX, DEFAULT_N0 = N0_CHOICES[0], N0_CHOICES[-1], 1000
INFO = {
    "schaetzer": "Nur der Schätzer (alles andere ist geklärt)",
    "daten": "Die Daten liegen schon vor",
    "anhalten": "Ich muss zwischendurch anhalten dürfen",
    "gradient": "Ich brauche Gradienten nach Parametern",
    "einfluss": "Ich will wissen, welche Eingabe die Varianz treibt",
    "geschachtelt": "Erwartung über Erwartung",
    "zeit": "Fehler durch Zeitdiskretisierung",
    "modelle": "Es gibt ein billiges zweites Modell",
    "ziehen": "Die Verteilung lässt sich nicht ziehen",
}
DEFAULT_INFO = "schaetzer"
NETZE = {"3x3": (3, 3), "3x4": (3, 4)}     # Experiment 3
DEFAULT_NETZ = "3x3"
SEED_MAX, DEFAULT_SEED = 999999, 40

# --- Messungen --------------------------------------------------------------------------------------------------------------------------------
R = 300                                    # Wiederholungen je Zelle (Varianzfaktoren auf etwa ±8 % bis ±35 % genau, je nach Aufgabe)
R_PILOTS = 100                             # Wiederholungen der Wahl nach dem Probelauf
P_GRID = (0.8, 0.85, 0.9, 0.95, 0.97, 0.99)    # Experiment 3: Haltewahrscheinlichkeiten je Kante
NPI_THRESHOLD = 50.0                       # Regel: erwartete Trefferzahl N·π̂ darunter -> Importance Sampling
RARE_HITS = 10                             # Probelauf mit weniger Treffern ist blind
HONEST_COVER = 0.90                        # darunter gilt eine Fehlerangabe als gebrochen
WIDE_RATIO = 1.5                           # Standardfehler/Streuung darüber gilt als zu breit
LOSS_FACTOR = 1.10                         # Faktor darüber: schlechter als einfach

# --- Presets ----------------------------------------------------------------------------------------------------------------------------------
PRESETS = {
    "Standardfall (Voreinstellung)": {"task": DEFAULT_TASK, "N": DEFAULT_N, "n0": DEFAULT_N0, "info": DEFAULT_INFO, "netz": DEFAULT_NETZ, "seed": DEFAULT_SEED},
    "Seltener Ausfall": {"task": "netz33_p99", "N": DEFAULT_N, "n0": DEFAULT_N0, "info": DEFAULT_INFO, "netz": DEFAULT_NETZ, "seed": DEFAULT_SEED},
    "Glatt und hochdimensional": {"task": "glatt64", "N": DEFAULT_N, "n0": DEFAULT_N0, "info": DEFAULT_INFO, "netz": DEFAULT_NETZ, "seed": DEFAULT_SEED},
    "Symmetrie": {"task": "symmetrisch12", "N": DEFAULT_N, "n0": DEFAULT_N0, "info": DEFAULT_INFO, "netz": DEFAULT_NETZ, "seed": DEFAULT_SEED},
    "Schwerer Schwanz": {"task": "schwanz2", "N": DEFAULT_N, "n0": DEFAULT_N0, "info": DEFAULT_INFO, "netz": DEFAULT_NETZ, "seed": DEFAULT_SEED},
    "Rau und hochdimensional": {"task": "sprung32", "N": DEFAULT_N, "n0": DEFAULT_N0, "info": DEFAULT_INFO, "netz": DEFAULT_NETZ, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Netz 3×3, p = 0,9, N = 1024: die erwartete Trefferzahl liegt mit etwa 54 knapp über der Schwelle von 50; Importance Sampling und Sobol in Blöcken liegen nah beieinander, Importance Sampling ist etwas besser (Faktor 0,33 bis 0,36 gegen 0,37 bis 0,46, je nach Messung).",
    "Seltener Ausfall": "Netz 3×3, p = 0,99: erwartet wird kaum ein Treffer (N·π = 0,4). Importance Sampling senkt die Varianz auf rund 2 %; alle anderen Verfahren bleiben bei etwa 1 und decken nur etwa ein Drittel.",
    "Glatt und hochdimensional": "exp(Σ aᵢ uᵢ) in 64 Dimensionen mit abfallenden Gewichten: Sobol senkt die Varianz auf 0,25 %, der naive Balken ist rund 23-mal zu breit; Blöcke sind ehrlich und kosten Genauigkeit.",
    "Symmetrie": "(Σ(uᵢ − ½))²: antithetische Paare verdoppeln die Varianz (Faktor 2,4), Sobol senkt sie auf 0,2 %.",
    "Schwerer Schwanz": "exp(2z): einfach, antithetisch und Kontrollvariate decken nur 0,82 bis 0,86, Sobol in Blöcken 0,74; nur die sehr breiten Balken von Latin Hypercube und Sobol (Folge) decken 1,00.",
    "Rau und hochdimensional": "Indikator in 32 Dimensionen: Sobol (eine Folge) erreicht 0,41, die anderen Verfahren 0,7 bis 0,8; der Gewinn bleibt klein.",
}
