"""Szenario: der Baum der Entscheidungsfragen (mit den Zahlen der Stücke 1 bis 15, hier NICHT neu gemessen) und die exakte Ausfallwahrscheinlichkeit der Netze."""

from __future__ import annotations

import mcw_algorithm as A
import mcw_constants as C

# Fragen 0 bis 4: je Frage die Informationslagen, bei denen sie mit „Ja“ beantwortet wird, und was daraus folgt. Die Zahlen stammen aus den READMEs der genannten Stücke.
QUESTIONS = (
    dict(nr=0, frage="Lässt sich die Verteilung nicht ziehen?", ja=("ziehen",),
         folge={"ziehen": "Ketten statt unabhängiger Ziehungen: MCMC (Burn-in, Mischzeit, R̂). Gehört nicht zu dieser Linie."},
         verweis={"ziehen": "Inverse Probleme 8 (posterior-sampling-demo)"}, zahl_aus={"ziehen": "Verweis, nicht gemessen"}),
    dict(nr=1, frage="Liegen die Daten schon vor, und ist die Statistik beliebig?", ja=("daten",),
         folge={"daten": "Bootstrap. Aber: Maximum Abdeckung 0,000, Standardabweichung 0,586, seltene Ereignisse (π = 0,0118, n = 30) 0,300."},
         verweis={"daten": "Stück 15 (bootstrap-demo)"}, zahl_aus={"daten": "aus Stück 15, nicht neu gemessen"}),
    dict(nr=2, frage="Muss man zwischen den Ziehungen anhalten dürfen?", ja=("anhalten",),
         folge={"anhalten": "Jederzeit gültige Konfidenzfolgen. Hinsehen bei jedem n lässt die Wahrheit in 50 bis 81 % der Läufe irgendwann außerhalb von Wald und Wilson liegen; die Beta-Mischungs-Folge 0,009 bis 0,019."},
         verweis={"anhalten": "Stück 14 (abbruchregeln-demo)"}, zahl_aus={"anhalten": "aus Stück 14, nicht neu gemessen"}),
    dict(nr=3, frage="Will man Gradienten nach Parametern oder Einflüsse der Eingaben?", ja=("gradient", "einfluss"),
         folge={"gradient": "Gradientenschätzer. Pfadweise ist bei einem Sprung exakt null (Verzerrung −100 %); der maßwertige Schätzer ist bei seltenen Ausfällen 18- bis 31-mal besser als der Score.",
                "einfluss": "Sobol-Indizes. Owens Drei-Matrizen-Schätzer ist bei kleinen Erstordnungsindizes 2- bis 30-mal genauer; bei wenigen diskreten Eingaben ist die Aufzählung billiger."},
         verweis={"gradient": "Stück 11 (monte-carlo-gradient-demo)", "einfluss": "Stück 10 (sobol-indizes-demo)"},
         zahl_aus={"gradient": "aus Stück 11, nicht neu gemessen", "einfluss": "aus Stück 10, nicht neu gemessen"}),
    dict(nr=4, frage="Hat die Aufgabe Struktur über den Schätzer hinaus?", ja=("geschachtelt", "zeit", "modelle"),
         folge={"geschachtelt": "Erwartung über Erwartung kostet geschachtelt ε⁻³; Multilevel hilft nur mit antithetischer Kopplung. Ist die innere Erwartung bekannt, ist bedingtes Monte Carlo 14- bis 25-mal billiger als Multilevel beim Knick.",
                "zeit": "Multilevel mit gekoppelten Ebenen. Bei Milstein 57-mal billiger bei ε = 0,000625; bei Euler mit Sprung teurer als gewöhnliches Monte Carlo.",
                "modelle": "Billiges zweites Modell als Kontrollvariate. Der Gewinn hängt an Korrelation und Kostenverhältnis; ein hoch korreliertes, aber nicht billiges Modell schadet."},
         verweis={"geschachtelt": "Stück 13 (geschachtelte-erwartungen-demo) und Stück 6", "zeit": "Stück 12 (multilevel-monte-carlo-demo)", "modelle": "Stück 4 (mehrere-modelle-demo)"},
         zahl_aus={"geschachtelt": "aus Stück 13, nicht neu gemessen", "zeit": "aus Stück 12, nicht neu gemessen", "modelle": "aus Stück 4, nicht neu gemessen"}),
)


def tree_steps(info: str) -> list[dict]:
    """Die Fragen 0 bis 4 für die gewählte Informationslage: „Nein“ bis zur ersten Frage, die mit „Ja“ beantwortet wird; dort endet der Baum. Bei „nur der Schätzer“ sind alle fünf „Nein“."""
    if info not in C.INFO:
        raise ValueError(f"unbekannte Informationslage {info}")
    steps = []
    for q in QUESTIONS:
        if info in q["ja"]:
            steps.append(dict(nr=q["nr"], frage=q["frage"], antwort="Ja", folge=q["folge"][info], verweis=q["verweis"][info], zahl_aus=q["zahl_aus"][info]))
            return steps
        steps.append(dict(nr=q["nr"], frage=q["frage"], antwort="Nein", folge="weiter zur nächsten Frage", verweis="", zahl_aus=""))
    return steps


def net_pi(netz: str, p: float) -> float:
    """Exakte Ausfallwahrscheinlichkeit 1 − R(p) des Gitternetzes aus der Aufzählung aller Kantenmengen."""
    rows, cols = C.NETZE[netz]
    _, _, Nk = A.grid_tab(rows, cols)
    return 1.0 - A.r_exact(Nk, p)
