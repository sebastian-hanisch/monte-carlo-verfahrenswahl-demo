# Verfahrenswahl: welcher Schätzer wann, und ist die Fehlerangabe ehrlich? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-monte-carlo-verfahrenswahl-demo.streamlit.app/)** -->

Sechzehntes und letztes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [bootstrap-demo](https://github.com/sebastian-hanisch/bootstrap-demo) (Stück 15) und Abschluss nach dem Muster von [matroid-verfahrenswahl-demo](https://github.com/sebastian-hanisch/matroid-verfahrenswahl-demo) und [kegel-verfahrenswahl-demo](https://github.com/sebastian-hanisch/kegel-verfahrenswahl-demo): Hier wird **kein neues Verfahren** gebaut, sondern die **Entscheidung** gemessen. Welche Bauart des Schätzers (einfach, antithetisch, Kontrollvariate, Latin Hypercube, Sobol, Importance Sampling) gewinnt bei welcher Aufgabe, wann kippt die Wahl, ist die Fehlerangabe danach ehrlich, und wie gut sagt ein kleiner Probelauf das voraus? Vehikel sind elf Aufgaben mit exakter Wahrheit, darunter die **Zuverlässigkeit der Netze der Reihe** (Hantel gibt es hier nicht; Gitter 3×3 und 3×4).

## Warum dieses Problem

Fünfzehn Stücke haben Werkzeuge angesammelt, und jedes senkt bei gleichem Aufwand den Fehler *irgendwo*. Wer eine neue Aufgabe vor sich hat, braucht eine Regel. Die Demo misst, wie viel die richtige Wahl bringt, wie viel die falsche kostet und ob man die Wahl mit einem kleinen Probelauf treffen kann. Jede Bauart bekommt **genau N Auswertungen** der Aufgabe; gemessen wird der mittlere quadratische Fehler gegen die Wahrheit, relativ zum einfachen Monte Carlo (**Faktor unter 1 = Gewinn**).

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`, Dateien `s16_*`). Was sich geändert hat oder beim Bau dazukam:

- **„Kein Verfahren gewinnt überall“ stimmt nur halb.** Bei N = 1024 gewinnt **randomisiertes Sobol 9 von 11** Aufgaben, **Importance Sampling die beiden seltenen Netz-Ausfälle**; antithetisch gewinnt nie. Der Befund ist, **woran es kippt** (erwartete Trefferzahl) und dass die **Fehlerangabe** vor der Genauigkeit entscheidet.
- **Die Kippschwelle ist die erwartete Trefferzahl N·π ≈ 50.** Importance Sampling gegen Sobol in acht Blöcken (ehrliche Fehlerangabe) über 24 Zellen (Netz 3×3 und 3×4, p von 0,8 bis 0,99, N = 1024 und 4096): Importance Sampling gewinnt bis N·π = 54,3, Sobol in Blöcken ab 56,8. Die Regel „N·π̂ < 50 → Importance Sampling, sonst Sobol in Blöcken“ trifft in **23 von 24** Zellen.
- **Die Fehlerangabe ist das zweite Kriterium.** Der naive Balken einer einzelnen Sobol-Folge ist bei acht von elf Aufgaben **1,6- bis 23-mal zu breit**; acht Blöcke machen ihn ehrlich und kosten Genauigkeit. Bei **sehr seltenen Ereignissen** (π = 0,0004) decken **alle** Intervalle außer dem von Importance Sampling nur **0,33 bis 0,36**; beim **schweren Schwanz** (exp(2z)) decken einfach, antithetisch und Kontrollvariate **0,82 bis 0,86**, Sobol in Blöcken **0,74**.
- **Der Probelauf ersetzt die Regel nicht.** Er sagt Kontrollvariate (1 − ρ̂²) und antithetisch (1 + ρ̂) gut voraus, den Latin-Hypercube-Gewinn bei kleinem Probelauf und 64 Dimensionen **völlig falsch**, den Sobol-Gewinn um das **2- bis 20-Fache zu klein**; bei seltenen Ereignissen ist er **blind**.
- **Rangfolgen nur bei doppeltem Abstand.** Varianzfaktoren aus 300 Wiederholungen sind auf etwa ±8 % bis ±35 % genau (antithetisch bei der rauen Aufgabe lieferte in der Vormessung 0,83 und 1,35, die Theorie sagt 0,95). Der Plan nannte für den Probelauf 61 bis 89 % der Wahlen innerhalb des 1,5-Fachen des Besten; mit den Werten der App sind es **67 % / 81 %** (n₀ = 200 / 1000, N = 1024).
- **Kein scipy zur Laufzeit:** Sobol mit eingebetteten Joe–Kuo-Richtungszahlen (aus Stück 9 übernommen und dort gegen `scipy.stats.qmc` geprüft), Normalverteilung nach Acklam, Irwin–Hall mit Brüchen.

## Modell

Gesucht ist μ = E f(U), U gleichverteilt auf [0, 1]^d, für **elf Aufgaben mit exakter Wahrheit**: **Netz-Ausfall** 3×3 mit p = 0,9 (π = 0,0530), 3×3 mit p = 0,99 (π = 0,00042), 3×4 mit p = 0,95 (π = 0,0121; Wahrheit aus der Aufzählung aller 2^m Kantenmengen); **Engpass-Schwelle** 3×3 (stetig mit Knick, 0,3133); **glatt** exp(Σ aᵢ uᵢ) mit d = 12 (662,4) und d = 64 mit abfallenden Gewichten (149,9); **symmetrisch** (Σ(uᵢ − ½))² mit d = 12 (Wahrheit 1); **schwerer Schwanz** exp(2z) mit d = 1 (e² = 7,389); **Wechselwirkung** ∏ 2uᵢ mit d = 8 (Wahrheit 1); **rau, hochdimensional** 1[Σ uᵢ > Schwelle] mit d = 32 (π = 0,0501, Irwin–Hall exakt); **Knick, hochdimensional** (Σ uᵢ − d/2)⁺ mit d = 32 (0,6525, exakt). Der 📐-Expander der App rechnet die Formeln vor.

## Methodik

**Sieben Verfahren, je N Auswertungen** (N = 256, 1024, 4096): **einfach**; **antithetisch** (N/2 Paare u, 1 − u); **Kontrollvariate** (geschätztes β, Kontrolle: Summe der Eingaben beziehungsweise Zahl der ausgefallenen Kanten); **Latin Hypercube**; **Sobol (eine Folge)** mit Owen-Scrambling und naivem Balken s/√N; **Sobol (8 Blöcke)** zu N/8 Punkten mit t-Intervall über die acht Blockmittel (ehrlicher Balken); **Importance Sampling** (nur Netz-Ausfälle; jede Kante fällt mit dem varianzoptimalen q′ aus, **als bekannt angenommen**, es zu lernen ist Stück 8). **Experiment 1:** Varianzfaktor je Aufgabe und Budget, 300 Wiederholungen. **Experiment 2:** Abdeckung des 95-%-Intervalls mit der Fehlerangabe, die man dem Verfahren üblicherweise gibt, und Standardfehler geteilt durch die tatsächliche Streuung (N = 1024, 300 Wiederholungen). **Experiment 3:** Importance Sampling gegen Sobol in Blöcken über die Haltewahrscheinlichkeit p (0,8 bis 0,99) beider Netze, aufgetragen gegen N·π. **Experiment 4:** der Probelauf (n₀ einfache Auswertungen, n₀/2 antithetische Paare, acht Sobol-Wiederholungen zu 64 Punkten; Kosten 2·n₀ + 512): vorhergesagt werden Kontrollvariate 1 − ρ̂², antithetisch 1 + ρ̂ₐ, Latin Hypercube 1 − (um das Binrauschen bereinigter additiver Anteil), Sobol Varianz der Wiederholungen/(σ̂²/64); gewählt wird der kleinste vorhergesagte Faktor, bei weniger als 10 Treffern (blinder Probelauf) Importance Sampling. 100 Wiederholungen der Wahl.

## Der Baum (Zahlen aus den Stücken, hier nicht neu gemessen)

Vor der Schätzerfrage stehen fünf Fragen zur Informationslage; sie verweisen auf die Stücke der Linie. Die Zahlen stammen aus deren READMEs. **Frage 5** (erwartete Trefferzahl unter der Schwelle?) und **Frage 6** (ist die Fehlerangabe ehrlich?) sind in diesem Stück gemessen.

| Frage (Informationslage) | Folge | Wo |
|---|---|---|
| 0. Lässt sich die Verteilung nicht ziehen? (Die Verteilung lässt sich nicht ziehen) | Ketten statt unabhängiger Ziehungen: MCMC (Burn-in, Mischzeit, R̂). Gehört nicht zu dieser Linie. | Inverse Probleme 8 (posterior-sampling-demo) |
| 1. Liegen die Daten schon vor, und ist die Statistik beliebig? (Die Daten liegen schon vor) | Bootstrap. Aber: Maximum Abdeckung 0,000, Standardabweichung 0,586, seltene Ereignisse (π = 0,0118, n = 30) 0,300. | Stück 15 (bootstrap-demo) |
| 2. Muss man zwischen den Ziehungen anhalten dürfen? (Ich muss zwischendurch anhalten dürfen) | Jederzeit gültige Konfidenzfolgen. Hinsehen bei jedem n lässt die Wahrheit in 50 bis 81 % der Läufe irgendwann außerhalb von Wald und Wilson liegen; die Beta-Mischungs-Folge 0,009 bis 0,019. | Stück 14 (abbruchregeln-demo) |
| 3. Will man Gradienten nach Parametern oder Einflüsse der Eingaben? (Ich brauche Gradienten nach Parametern) | Gradientenschätzer. Pfadweise ist bei einem Sprung exakt null (Verzerrung −100 %); der maßwertige Schätzer ist bei seltenen Ausfällen 18- bis 31-mal besser als der Score. | Stück 11 (monte-carlo-gradient-demo) |
| 3. Will man Gradienten nach Parametern oder Einflüsse der Eingaben? (Ich will wissen, welche Eingabe die Varianz treibt) | Sobol-Indizes. Owens Drei-Matrizen-Schätzer ist bei kleinen Erstordnungsindizes 2- bis 30-mal genauer; bei wenigen diskreten Eingaben ist die Aufzählung billiger. | Stück 10 (sobol-indizes-demo) |
| 4. Hat die Aufgabe Struktur über den Schätzer hinaus? (Erwartung über Erwartung) | Erwartung über Erwartung kostet geschachtelt ε⁻³; Multilevel hilft nur mit antithetischer Kopplung. Ist die innere Erwartung bekannt, ist bedingtes Monte Carlo 14- bis 25-mal billiger als Multilevel beim Knick. | Stück 13 (geschachtelte-erwartungen-demo) und Stück 6 |
| 4. Hat die Aufgabe Struktur über den Schätzer hinaus? (Fehler durch Zeitdiskretisierung) | Multilevel mit gekoppelten Ebenen. Bei Milstein 57-mal billiger bei ε = 0,000625; bei Euler mit Sprung teurer als gewöhnliches Monte Carlo. | Stück 12 (multilevel-monte-carlo-demo) |
| 4. Hat die Aufgabe Struktur über den Schätzer hinaus? (Es gibt ein billiges zweites Modell) | Billiges zweites Modell als Kontrollvariate. Der Gewinn hängt an Korrelation und Kostenverhältnis; ein hoch korreliertes, aber nicht billiges Modell schadet. | Stück 4 (mehrere-modelle-demo) |

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Strom 40, `tests/test_claims.py` rechnet sie nach. **Experiment 1**, Varianzfaktor gegen einfach bei N = 1024 (kleiner ist besser; Sobol 1 = eine Folge, Sobol 8 = acht Blöcke):

| Aufgabe | antithetisch | Kontrollvariate | Latin Hypercube | Sobol 1 | Sobol 8 | Importance Sampling |
|---|---|---|---|---|---|---|
| Netz 3×3, p = 0,9 | 0,811 | 0,754 | 0,714 | **0,189** | 0,372 | 0,361 |
| Netz 3×3, p = 0,99 | 1,01 | 0,891 | 0,884 | 1,13 | 1,08 | **0,0177** |
| Netz 3×4, p = 0,95 | 0,851 | 0,848 | 0,841 | 0,692 | 0,736 | **0,298** |
| Engpass 3×3 | 0,622 | 0,513 | 0,442 | **0,116** | 0,280 | – |
| glatt, d = 12 | 0,493 | 0,307 | 0,306 | **0,0297** | 0,0720 | – |
| glatt, d = 64 | 0,322 | 0,919 | 0,154 | **0,0025** | 0,0136 | – |
| symmetrisch, d = 12 | **2,44** | 1,14 | 1,07 | **0,0019** | 0,0488 | – |
| schwerer Schwanz | 0,624 | 0,734 | 0,549 | **0,289** | 0,803 | – |
| Wechselwirkung, d = 8 | 0,931 | 0,859 | 0,659 | **0,0823** | 0,272 | – |
| rau, d = 32 | 0,714 | 0,733 | 0,758 | **0,406** | 0,521 | – |
| Knick, d = 32 | 0,552 | 0,275 | 0,280 | **0,0497** | 0,121 | – |

- **Siege:** N = 1024: Sobol (eine Folge) 9, Importance Sampling 2; N = 4096: Sobol 10, Importance Sampling 1; antithetisch nie. Beim schweren Schwanz schwanken die Faktoren stark von Budget zu Budget (nicht belastbar).
- **Antithetisch verliert bei Symmetrie:** 1,98 / 2,44 / 1,97 bei N = 256 / 1024 / 4096 (Theorie: 2; der Wert 2,4 bei N = 1024 liegt über den beiden anderen, die Streuung zwischen den Budgets deutet auf Rauschen).
- **Rangwechsel beim Budget (Netz 3×4, p = 0,95):** N = 1024: Importance Sampling 0,298 gegen Sobol 0,692; N = 4096: Sobol 0,205 gegen Importance Sampling 0,242.
- **Rau, d = 32:** Sobol (eine Folge) erreicht 0,41, die anderen Verfahren 0,71 bis 0,76; der Gewinn bleibt klein.
- **Sobol-Gewinn wächst mit N:** glatt, d = 12: 0,0644 / 0,0297 / 0,0134 bei N = 256 / 1024 / 4096.

**Experiment 2**, Abdeckung des 95-%-Intervalls bei N = 1024 (einfach / antithetisch / Kontrollvariate / Latin Hypercube / Sobol 1 / Sobol 8 / Importance Sampling) und Standardfehler geteilt durch Streuung bei Sobol 1:

| Aufgabe | Abdeckung | SF/Streuung Sobol 1 |
|---|---|---|
| Netz 3×3, p = 0,9 | 0,97 / 0,96 / 0,94 / 0,99 / 1,00 / 0,96 / 0,94 | 2,0 |
| Netz 3×3, p = 0,99 | **0,36 / 0,37 / 0,36 / 0,33 / 0,35 / 0,35 / 0,95** | 0,6 |
| Netz 3×4, p = 0,95 | 0,92 / 0,92 / 0,94 / 0,94 / 0,95 / 0,93 / 0,94 | 1,1 |
| Engpass 3×3 | 0,94 / 0,92 / 0,96 / 1,00 / 1,00 / 0,95 / – | 3,0 |
| glatt, d = 12 | 0,95 / 0,96 / 0,92 / 1,00 / 1,00 / 0,96 / – | 6,0 |
| glatt, d = 64 | 0,92 / 0,94 / 0,95 / 1,00 / 1,00 / 0,94 / – | 22,6 |
| symmetrisch, d = 12 | 0,97 / 0,94 / 0,94 / 0,96 / 1,00 / 0,98 / – | 23,2 |
| schwerer Schwanz | **0,86 / 0,84 / 0,82** / 1,00 / 1,00 / **0,74** / – | 1,2 |
| Wechselwirkung, d = 8 | 0,94 / 0,94 / 0,91 / 0,97 / 1,00 / 0,94 / – | 3,9 |
| rau, d = 32 | 0,96 / 0,96 / 0,95 / 0,97 / 1,00 / 0,95 / – | 1,6 |
| Knick, d = 32 | 0,95 / 0,96 / 0,95 / 1,00 / 1,00 / 0,96 / – | 4,7 |

- **Kosten der ehrlichen Blöcke** (Varianzfaktor aus der Streuung, Folge → Blöcke): Netz 3×3 p = 0,9: 0,246 → 0,472; Engpass 0,114 → 0,303; glatt d = 12 0,032 → 0,100; glatt d = 64 0,0017 → 0,0116; symmetrisch 0,0020 → 0,0561; Wechselwirkung 0,065 → 0,273; Knick 0,046 → 0,097; rau 0,405 → 0,536.

**Experiment 3**, Importance Sampling gegen Sobol in Blöcken, 24 Zellen (N = 1024 und 4096, beide Netze):

| Netz, N, p | N·π | Faktor Importance Sampling | Faktor Sobol 8 | Faktor Sobol 1 |
|---|---|---|---|---|
| 3×3, 1024, 0,8 | 236,7 | 0,598 | 0,370 | 0,223 |
| 3×3, 1024, 0,85 | 130,3 | 0,435 | 0,384 | 0,260 |
| 3×3, 1024, 0,9 | 54,3 | 0,331 | 0,461 | 0,241 |
| 3×3, 1024, 0,95 | 12,1 | 0,226 | 0,702 | 0,366 |
| 3×3, 1024, 0,97 | 4,1 | 0,103 | 0,976 | 0,346 |
| 3×3, 1024, 0,99 | 0,4 | 0,0181 | 0,974 | 0,979 |
| 3×4, 4096, 0,9 | 227,3 | 0,525 | 0,409 | 0,192 |
| 3×4, 4096, 0,95 | 49,4 | 0,282 | 1,01 | 0,198 |
| 3×4, 4096, 0,99 | 1,7 | 0,0321 | 1,00 | 0,611 |

- **Regel** „N·π̂ < 50“: **23 von 24** Zellen richtig (3×3: 11 von 12, 3×4: 12 von 12), mittleres Verhältnis zur besten Varianz **1,014** (schlechtester Fall 1,39). **Immer Sobol in Blöcken** 3,09 (schlechtester Fall **53,8**), **immer Importance Sampling** 1,22 (2,19). Mit Schwelle 20: 21 von 24 (1,122), mit Schwelle 100: 23 von 24 (1,010).
- **Der eine Fehlgriff:** Netz 3×3, p = 0,9, N = 1024 (N·π = 54,3): die Regel sagt Sobol in Blöcken, Importance Sampling ist etwas besser (0,331 gegen 0,461).
- Mit der **einen Folge** (naiver Balken) wären Importance Sampling und Sobol schon bei N·π = 49,4 gemischt (Netz 3×4, N = 4096: Sobol 0,198 gegen Importance Sampling 0,282).

**Experiment 4**, der Probelauf (N = 1024, 100 Wiederholungen der Wahl, über elf Aufgaben):

- **Wahl nach Probelauf** innerhalb des **1,5-Fachen** des Besten in **67 % / 81 %** der Fälle (n₀ = 200 / 1000), mittleres Verhältnis zum Besten **1,88 / 1,25**; Kosten **0,89 / 2,45 × N** (bei N = 4096: 0,22 / 0,61 × N).
- **Immer ein Verfahren** (mittleres Verhältnis zum Besten): einfach **19,2**, antithetisch **14,3**, Kontrollvariate **12,8**, Latin Hypercube **10,2**, Sobol (eine Folge) **1,58**, Importance Sampling (drei Netz-Aufgaben) **1,24**.
- **Blind:** beim seltenen Ausfall (π = 0,00042) sieht der Probelauf in **100 %** der Läufe weniger als 10 Treffer; beim Netz 3×4 (π = 0,0121) in **100 % / 23 %** (n₀ = 200 / 1000).
- **Latin Hypercube bei d = 64:** vorhergesagt 0,011, gemessen 0,154 (n₀ = 200); der Probelauf wählt Latin Hypercube in 95 % der Fälle, das Verhältnis zum Besten liegt bei **49,7**; mit n₀ = 1000 stimmt es besser (vorhergesagt 0,101).
- **Vorhersagen** (n₀ = 1000): antithetisch bei der symmetrischen Aufgabe **2,00** vorhergesagt, **2,44** gemessen; Kontrollvariate bei glatt, d = 12: **0,377** gegen **0,307**; Sobol 0,12 gegen 0,030 (der Gewinn wird unterschätzt).

## Ehrliche Grenzen

- **Nur diese elf Aufgaben,** keine Mischformen, keine Aufgabe mit mehreren Ausgaben. Die **Schwelle 50** gilt für Gitternetze im Bereich N·π von 0,4 bis rund 1000 und N von 1024 bis 4096; sie ist keine Konstante.
- **Die Hilfsverteilung des Importance Sampling ist als optimal bekannt angenommen;** sie zu lernen kostet Probeläufe (Stück 8), die Zahlen sind Bestwerte.
- **Kontrollvariate und Latin Hypercube haben je eine Bauart,** Sobol nur Owen-Scrambling mit acht Blöcken.
- **Keine Rechenzeit gemessen** (gleiche Zahl Auswertungen, nicht gleiche Sekunden; das Erzeugen der Sobol-Punkte kostet im Python-Eigenbau ein Vielfaches).
- **Rauschen:** 300 Wiederholungen. Enge Fälle (Standardfall: Sobol in Blöcken gegen Importance Sampling) sind nicht belastbar, ebenso die Faktoren beim schweren Schwanz.
- **Die Zahlen der Stücke 1 bis 15 im Baum** sind aus deren READMEs übernommen und **hier nicht neu gemessen** (Bootstrap, Abbruchregeln, Gradienten, Sobol-Indizes, geschachtelte Erwartungen, Multilevel, mehrere Modelle).
- **Schwelle und Regel wurden an denselben Aufgaben gefunden, an denen sie gemessen werden:** abgelesen an den 3×3-Zellen der Vormessung, an frischen Wiederholungen und beiden Netzen bestätigt (3×4: 12 von 12), nicht an fremden Aufgaben.
- **Quellen:** Lehrbuchwissen und die Stücke der Linie; keine Literatur neu gelesen.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Sobol-Punkte von Hand (die ersten vier Punkte in zwei Dimensionen), Netz-Eigenschaft und Latin-Hypercube-Strata, Irwin–Hall von Hand, Ausfallzählung, antithetisch und Kontrollvariate bei f(u) = u exakt, Latin Hypercube bei N = 2 mit Varianz 1/96 gegen 1/24, antithetisch bei Symmetrie mit Faktor 2, Importance Sampling mit q′ = q gleich der Bernoulli-Varianz, Probelauf, Regel an den Schwellen.
- **Orakel** (`tests/test_oracle_mcw.py`): Sobol-Punkte **ganzzahlig gegen `scipy.stats.qmc.Sobol`** (2, 7, 21, 64 Dimensionen); Normalverteilung gegen `scipy.special`; Zusammenhang gegen **Tiefensuche**; Ausfallwahrscheinlichkeit gegen die **direkte Summe über alle Mengen**; Irwin–Hall gegen `scipy.stats.irwinhall`; Importance-Sampling-Moment gegen die direkte Summe und gegen Simulation; Erwartungstreue aller Schätzer; Owen-Scrambling gleichverteilt.
- **Eigenschaften** (`tests/test_properties.py`): Faktor des einfachen Verfahrens gleich 1, antithetisch verliert bei Symmetrie, Sobol gewinnt bei glatter Aufgabe, Importance Sampling bei seltenem Ausfall, naiver Balken zu breit und Blöcke ehrlich, seltener Ausfall und schwerer Schwanz brechen die Fehlerangabe, Kippschwelle, Gewinn wächst mit dem Budget, Probelauf blind, Reproduzierbarkeit.
- **Randfälle** (`tests/test_edge_cases.py`): Fehlermeldungen (Importance Sampling nur für Netze, Blöcke brauchen N durch 8, unbekannte Eingaben, Dimension), kleinstes Budget, Probelauf ohne Treffer und mit konstanter Aufgabe, jede Informationslage des Baums, Zahlenformate.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes rechnet; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jede Auswahl jedes Reglers, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`): jede Zahl dieser README.

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `mcw_algorithm.py` | Kern: Strom, Sobol (Owen), Latin Hypercube, Netze, elf Aufgaben, sieben Verfahren, Importance Sampling, Probelauf, Trefferzahl-Regel |
| `mcw_constants.py` | Regler-Grenzen, Aufgaben, feste Annahmen, Farben, Presets |
| `mcw_evaluation.py` | Messungen (Faktoren, Abdeckung, Kippschwelle, Probelauf), Verdikte, Weg durch den Baum |
| `mcw_layout.py` | Fester Kopf (Stylesheet) |
| `mcw_pdf_export.py` | PDF der aktuellen Einstellung |
| `mcw_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `mcw_scenario.py` | Baum der Entscheidungsfragen mit den Zahlen der Stücke 1 bis 15, exakte Netz-Ausfallwahrscheinlichkeit |
| `mcw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Verteilung nicht ziehbar (Ketten, Burn-in, R̂) | [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo) (Inverse Probleme 8), MCMC-Linie später |
| Lernen der Hilfsverteilung beim Importance Sampling | [kreuzentropie-demo](https://github.com/sebastian-hanisch/kreuzentropie-demo) (Stück 8) |
| Bootstrap, Abbruchregeln, Gradienten, Sobol-Indizes | [bootstrap-demo](https://github.com/sebastian-hanisch/bootstrap-demo) (Stück 15), [abbruchregeln-demo](https://github.com/sebastian-hanisch/abbruchregeln-demo) (14), [monte-carlo-gradient-demo](https://github.com/sebastian-hanisch/monte-carlo-gradient-demo) (11), [sobol-indizes-demo](https://github.com/sebastian-hanisch/sobol-indizes-demo) (10) |
| Multilevel, geschachtelte Erwartungen, mehrere Modelle | [multilevel-monte-carlo-demo](https://github.com/sebastian-hanisch/multilevel-monte-carlo-demo) (12), [geschachtelte-erwartungen-demo](https://github.com/sebastian-hanisch/geschachtelte-erwartungen-demo) (13), [mehrere-modelle-demo](https://github.com/sebastian-hanisch/mehrere-modelle-demo) (4) |
| Bayessche Quadratur, Stein-Kontrollvariaten | nicht gebaut (gelbe Kandidaten der Gliederung) |
| Rechenzeit der Verfahren | nicht gemessen |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
