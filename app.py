"""Verfahrenswahl: welcher Schätzer wann, und ist die Fehlerangabe ehrlich? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 16 (Abschluss) der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Kein neues Verfahren, sondern die ENTSCHEIDUNG wird gemessen. Elf Aufgaben mit exakter Wahrheit, sechs Bauarten des Schätzers mit je gleich
vielen Auswertungen: Sobol gewinnt die meisten, Importance Sampling die seltenen Ausfälle (Kippschwelle: erwartete Trefferzahl N·π ≈ 50), antithetisch gewinnt nie; bei seltenen Ereignissen und schweren Schwänzen
bricht jede einfache Fehlerangabe; ein Probelauf ersetzt die Regel nicht.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import mcw_constants as C
import mcw_evaluation as ev
from mcw_evaluation import Settings, de, fac, mlabel, n1, pr, zahl
from mcw_layout import dunkles_design, kopf_css
from mcw_pdf_export import build_pdf
from mcw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mcw_visualization import build_cover, build_cross, build_factors, build_matrix, build_pilots

st.set_page_config(page_title="Verfahrenswahl: welcher Schätzer wann? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Verfahren messen …")
def _analysis(settings, R):
    return ev.analyse(settings, R)


@st.cache_data(show_spinner="Alle drei Budgets messen …")
def _matrix(task, seed, R):
    return ev.matrix(task, seed, R)


@st.cache_data(show_spinner="Fehlerangaben messen …")
def _cover(task, N, seed, R):
    return ev.cover(task, N, seed, R)


@st.cache_data(show_spinner="Kippschwelle messen …")
def _cross(netz, N, seed, R):
    return ev.cross(netz, N, seed, R)


@st.cache_data(show_spinner="Probeläufe wiederholen …")
def _pilots(task, N, n0, seed, R):
    return ev.pilots(task, N, n0, seed, R)


def _info_label(key):
    return C.INFO[key]


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mcw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Verfahrenswahl: welcher Schätzer wann?")
st.markdown(
    """
Die Linie hat fünfzehn Verfahren angesammelt. Hier wird nichts Neues gerechnet, sondern die **Entscheidung** gemessen: Welche Bauart des Schätzers gewinnt bei welcher Aufgabe, **wann kippt die Wahl**,
ist die **Fehlerangabe** danach noch ehrlich, und wie gut sagt ein **kleiner Probelauf** das voraus? Elf Aufgaben mit exakter Wahrheit, sechs Bauarten mit **je gleich vielen Auswertungen**:
einfach, antithetisch, Kontrollvariate, Latin Hypercube, Sobol (eine Folge oder acht Blöcke) und Importance Sampling. Gemessen wird der mittlere quadratische Fehler gegen die Wahrheit, relativ zum einfachen Monte Carlo:
**ein Faktor unter 1 ist ein Gewinn**.

Weiter unten: **🌳 der Weg durch den Baum**, **🔬 vier Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wählt man den Schätzer", expanded=True):
    st.markdown(
        """
Die **Regel** dieser Demo hat zwei Fragen:

1. **Wie viele Treffer erwarten Sie?** Bei einem seltenen Ereignis (Netzausfall) ist die erwartete Trefferzahl **N·π** das Maß. Liegt sie **unter 50**, gewinnt **Importance Sampling** (die Hilfsverteilung ist hier als bekannt angenommen; sie zu lernen ist Stück 8); darüber **Sobol in acht Blöcken**.
2. **Ist die Fehlerangabe ehrlich?** Der naive Balken einer einzelnen Sobol-Folge ist oft **mehrfach zu breit**; acht unabhängige Blöcke machen ihn ehrlich und kosten Genauigkeit. Bei **seltenen Ereignissen** und **schweren Schwänzen** hält die einfache Fehlerangabe fast aller Verfahren nicht.

Die anderen Bauarten gelten **nur nach Messung**: **Kontrollvariate** bei großem ρ², **Latin Hypercube** bei additiver Struktur, **antithetisch** nur bei negativer Korrelation der Paare (bei symmetrischen Aufgaben kostet es das Doppelte).
Vor diesen beiden Fragen steht der **Baum** der Informationslage (Daten schon da? anhalten? Gradient? Erwartung in Erwartung? ziehbar?): er verweist auf die Stücke der Linie.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Aufgabe und Budget**")
    task = st.selectbox("Aufgabe", options("task_select"), format_func=lambda k: C.TASKS[k]["label"], key="task_select", help="Elf Aufgaben mit exakter Wahrheit; die Netz-Aufgaben haben als einzige ein Importance Sampling.")
    n = st.select_slider("Budget N (Auswertungen je Verfahren)", options("n_select"), key="n_select", help="Jedes Verfahren darf genau N Auswertungen der Aufgabe verbrauchen.")
    info = st.selectbox("Informationslage (für den Baum)", options("info_select"), format_func=_info_label, key="info_select", help="Was Sie über die Aufgabe wissen; bestimmt, wie weit der Baum bis zur Schätzerfrage kommt.")
    n0 = st.select_slider("Größe n₀ des Probelaufs", options("n0_select"), key="n0_select", help="Einfache Auswertungen des Probelaufs; dazu kommen n₀ antithetische und 512 Sobol-Auswertungen.")
    st.markdown("**Kippschwelle (Experiment 3)**")
    netz = st.selectbox("Netz", options("netz_select"), key="netz_select", help="Gitternetz, dessen Ausfallwahrscheinlichkeit über die Haltewahrscheinlichkeit p je Kante verändert wird.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"task_select": task, "n_select": int(n), "n0_select": int(n0), "info_select": info, "netz_select": netz, "seed_input": int(seed)})

settings = Settings(str(task), int(n), int(n0), str(info), str(netz), int(seed))
a = _analysis(settings, C.R)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"{a.label}; Wahrheit {de(a.truth, 4)}; Dimension {a.d}; N = {zahl(a.N)} Auswertungen je Verfahren, {a.R} Wiederholungen. Faktor = Varianz gegen einfaches Monte Carlo, kleiner ist besser.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit", de(a.truth, 4), help="Exakt: aus der Aufzählung aller Kantenmengen, der Irwin-Hall-Verteilung oder einer Formel.")
m2.metric("Bester (ehrlicher Balken)", mlabel(a.best_honest), delta=f"Faktor {fac(a.factor[a.best_honest])}", delta_color="off", help="Das Verfahren mit dem kleinsten Faktor, ohne die einzelne Sobol-Folge, deren Standard-Balken nicht ehrlich ist.")
m3.metric("Regel empfiehlt", mlabel(a.rec), delta=f"Faktor {fac(a.factor[a.rec])}", delta_color="off", help=f"Trefferzahl-Regel: erwartete Trefferzahl N·π̂ unter {zahl(C.NPI_THRESHOLD)} gibt Importance Sampling, sonst Sobol in Blöcken.")
m4.metric("antithetisch", fac(a.factor["anti"]), delta="schlechter als einfach" if a.factor["anti"] > C.LOSS_FACTOR else "nicht schlechter", delta_color="inverse" if a.factor["anti"] > C.LOSS_FACTOR else "off", help="Varianz der antithetischen Paare gegen einfach, bei gleichen Kosten.")
st.plotly_chart(build_factors(a), width="stretch", key="main_chart")
st.caption("Varianzfaktor je Bauart (logarithmische Achse). Alles unter der gestrichelten 1 ist besser als einfaches Monte Carlo.")

if st.button("📄 PDF der Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="verfahrenswahl.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welche Bauart für diese Aufgabe?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🌳 Der Weg durch den Baum")
st.caption("Die Fragen 0 bis 4 verweisen auf die Stücke der Linie; ihre Zahlen stammen **aus den Stücken, nicht neu gemessen**. Frage 5 und 6 sind in diesem Stück gemessen (Experimente unten).")
steps = ev.tree_path(settings, a)
st.dataframe({"Frage": [f"{s['nr']}. {s['frage']}" for s in steps], "Antwort": [s["antwort"] for s in steps], "Folge": [s["folge"] for s in steps],
              "Wo": [s["verweis"] for s in steps], "Herkunft der Zahlen": [s["zahl_aus"] for s in steps]}, hide_index=True, width="stretch", key="tree_table")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer gewinnt wo?")
st.caption(f"Die gewählte Aufgabe bei allen drei Budgets (N = {', '.join(zahl(x) for x in C.N_CHOICES)}), {C.R} Wiederholungen je Verfahren und Budget. Die Rangfolge hängt am Budget; manche Verfahren sind schlechter als einfach.")
if st.button("▶ Messung starten", key="run_matrix"):
    st.session_state["matrix_done"] = True
if st.session_state.get("matrix_done"):
    ex1 = _matrix(settings.task, settings.seed, C.R)
    st.plotly_chart(build_matrix(ex1), width="stretch", key="matrix_chart")
    ms = [m for m in C.METHODS if m in ex1[C.N_CHOICES[0]]]
    st.dataframe({"Verfahren": [mlabel(m) for m in ms], **{f"Faktor N = {zahl(N)}": [fac(ex1[N][m]["factor"]) for m in ms] for N in C.N_CHOICES}}, hide_index=True, width="stretch", key="matrix_table")
    kind, text = ev.verdict_matrix(ex1, a.label, C.R)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Ist der Balken ehrlich?")
st.caption(f"Jedes Verfahren bekommt die Fehlerangabe, die man ihm üblicherweise gibt (Sobol als eine Folge: der naive Balken; Sobol in 8 Blöcken: t-Intervall über die Blöcke). Abdeckung des 95-%-Intervalls bei N = {zahl(settings.N)}, {C.R} Wiederholungen.")
if st.button("▶ Messung starten", key="run_cover"):
    st.session_state["cover_done"] = True
if st.session_state.get("cover_done"):
    ex2 = _cover(settings.task, settings.N, settings.seed, C.R)
    st.plotly_chart(build_cover(ex2), width="stretch", key="cover_chart")
    ms = [m for m in C.METHODS if m in ex2]
    st.dataframe({"Verfahren": [mlabel(m) for m in ms], "Abdeckung": [pr(ex2[m]["cover"]) for m in ms], "Standardfehler / Streuung": [de(ex2[m]["se_ratio"], 2) for m in ms],
                  "Varianzfaktor aus der Streuung": [fac(ex2[m]["factor"]) for m in ms]}, hide_index=True, width="stretch", key="cover_table")
    kind, text = ev.verdict_cover(ex2, a.label, settings.N, C.R)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Wo kippt Importance Sampling gegen Sobol?")
st.caption(f"Netz {settings.netz}, Haltewahrscheinlichkeit p von {de(C.P_GRID[0], 2)} bis {de(C.P_GRID[-1], 2)}, N = {zahl(settings.N)}, {C.R} Wiederholungen. Die Achse ist die erwartete Trefferzahl N·π; verglichen werden Importance Sampling (Hilfsverteilung bekannt) und Sobol in Blöcken (ehrliche Fehlerangabe).")
if st.button("▶ Messung starten", key="run_cross"):
    st.session_state["cross_done"] = True
if st.session_state.get("cross_done"):
    ex3 = _cross(settings.netz, settings.N, settings.seed, C.R)
    st.plotly_chart(build_cross(ex3, settings.N), width="stretch", key="cross_chart")
    st.dataframe({"p": [de(r["p"], 2) for r in ex3], "π": [de(r["pi"], 4) for r in ex3], "N·π": [n1(r["npi"]) for r in ex3], "Faktor Importance Sampling": [fac(r["factor"]["is"]) for r in ex3],
                  "Faktor Sobol (Blöcke)": [fac(r["factor"]["qmc8"]) for r in ex3], "Faktor Sobol (Folge)": [fac(r["factor"]["qmc"]) for r in ex3], "Sieger": [mlabel(r["best"]) for r in ex3],
                  "Regel sagt": [mlabel(r["rule"]) for r in ex3]}, hide_index=True, width="stretch", key="cross_table")
    kind, text = ev.verdict_cross(ex3, settings.netz, settings.N, C.R)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Der Probelauf")
st.caption(f"Statt der Regel einen Probelauf (n₀ = {zahl(settings.n0)}) entscheiden lassen: aus einer einfachen Stichprobe, antithetischen Paaren und acht Sobol-Wiederholungen werden die Varianzfaktoren vorhergesagt und das beste Verfahren gewählt. {C.R_PILOTS} Wiederholungen der Wahl für die gewählte Aufgabe; bei weniger als {C.RARE_HITS} Treffern fällt die Wahl auf Importance Sampling zurück.")
if st.button("▶ Messung starten", key="run_pilots"):
    st.session_state["pilots_done"] = True
if st.session_state.get("pilots_done"):
    ex4 = _pilots(settings.task, settings.N, settings.n0, settings.seed, C.R)
    st.plotly_chart(build_pilots(ex4), width="stretch", key="pilots_chart")
    st.dataframe({"Verfahren": [mlabel(m) for m in ex4["picks"]], "gewählt": [f"{zahl(100 * c / ex4['RP'])} %" for c in ex4["picks"].values()], "gemessener Faktor": [fac(ex4["true"][m]) for m in ex4["picks"]],
                  "vorhergesagt (Mittel)": [fac(ex4["pred"][m]) if m in ex4["pred"] else "–" for m in ex4["picks"]]}, hide_index=True, width="stretch", key="pilots_table")
    kind, text = ev.verdict_pilots(ex4, a.label, settings.n0, settings.N)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein einzelner Probelauf"):
    st.caption(f"Ein einziger Probelauf mit n₀ = {zahl(settings.n0)} für die gewählte Aufgabe (der Strom bestimmt ihn; „Neuen Strom wählen“ zieht einen anderen). Aus den Diagnosen sagt er die Varianzfaktoren voraus; daneben steht, was die Messung bei N = {zahl(settings.N)} ergibt.")
    p = a.pilot
    st.dataframe({"Diagnose": ["Korrelation ρ̂ zur Kontrolle", "Korrelation ρ̂ₐ der antithetischen Paare", "additiver Anteil der Varianz", "Treffer im Probelauf", "geschätztes π̂", "erwartete Trefferzahl N·π̂"],
                  "Wert": [de(p["rho"], 3), de(p["rho_a"], 3), de(p["add"], 3), n1(p["hits"]) if p["hits"] is not None else "–", de(p["pi_hat"], 4) if p["pi_hat"] is not None else "–", n1(a.npi) if a.npi == a.npi else "–"]},
                 hide_index=True, width="stretch", key="pilot_diag")
    st.dataframe({"Verfahren": [mlabel(m) for m in ("cv", "anti", "lhs", "qmc")], "vorhergesagt": [fac(p["pred"][m]) for m in ("cv", "anti", "lhs", "qmc")],
                  "gemessen": [fac(a.factor[m]) for m in ("cv", "anti", "lhs", "qmc")]}, hide_index=True, width="stretch", key="pilot_pred")
    kind, text = ev.verdict_pilot_single(p, a)
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Fehlerangabe passt zum Verfahren** | Der naive Balken einer Sobol-Folge ist mehrfach zu breit; acht Blöcke sind ehrlich, kosten aber Genauigkeit (Experiment 2). | Stück 9 |
| **Es gibt genug Treffer** | Bei weniger als etwa 50 erwarteten Treffern gewinnt Importance Sampling, bei sehr seltenen Ereignissen decken alle anderen Intervalle nur etwa ein Drittel (Experiment 2 und 3). | Stück 7 und 8 |
| **Der Schwanz ist leicht** | Beim schweren Schwanz deckt keine übliche Fehlerangabe zu 95 % (Experiment 2). | Stück 7 (Pareto-k̂) |
| **Antithetisch hilft** | Bei symmetrischen Aufgaben verdoppelt es die Varianz (Experiment 1). | Stück 2 |
| **Ein Probelauf sagt voraus** | Er ist bei seltenen Ereignissen blind, unterschätzt den Sobol-Gewinn und kostet bis über das Endbudget (Experiment 4). | Stück 3, 5 und 9 |
| **Das Ziehen klappt** | Lässt sich die Verteilung nicht ziehen, hilft keines dieser Verfahren (Baum, Frage 0). | Inverse Probleme 8 |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Elf **erfundene** Aufgaben auf dem Einheitswürfel, deren Wahrheit exakt bekannt ist (Aufzählung aller Kantenmengen, Irwin–Hall-Verteilung, Formeln). Jedes Verfahren bekommt genau N Auswertungen; gemessen wird der mittlere quadratische Fehler gegen die Wahrheit, relativ zum einfachen Monte Carlo.
Zufallszahlen aus festen Strömen (SplitMix64), Sobol mit Owen-Scrambling und eingebetteten Joe-Kuo-Richtungszahlen.

**Grenzen.** Nur diese elf Aufgaben, keine Mischformen. Die **Hilfsverteilung des Importance Sampling ist als optimal bekannt angenommen**; sie zu lernen kostet Probeläufe (Stück 8), die Zahlen sind Bestwerte. Kontrollvariate und Latin Hypercube haben je eine Bauart,
Sobol nur Owen-Scrambling mit acht Blöcken. **Rechenzeit ist nicht gemessen** (gleiche Zahl Auswertungen, nicht gleiche Sekunden). Multilevel, geschachtelte Erwartungen, Gradienten, Bootstrap, Sobol-Indizes und Abbruchregeln sind **nicht neu gemessen**, ihre Zahlen im Baum stammen aus den Stücken der Linie.
Die **Schwelle 50** wurde an Gitternetzen im Bereich N·π von 0,1 bis rund 950 gefunden; sie gilt für diese Netze und Budgets, nicht als Naturgesetz.

**Bekannte Probleme und Unsicherheiten.** Varianzfaktoren aus 300 Wiederholungen sind auf etwa ±8 % (glatte Aufgaben) bis ±35 % (Ereignisindikatoren) genau; belastbar sind Rangfolgen mit mindestens doppeltem Abstand, enge Fälle (Standardfall: Sobol in Blöcken gegen Importance Sampling) sind es nicht.
Die Schwelle wurde an den 3×3-Netzen abgelesen und an den 3×4-Netzen geprüft, nicht blind festgelegt. Literatur: Lehrbuchwissen und die Stücke der Linie, keine Quelle neu gelesen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Schätzer und Faktor.** Gesucht ist $\mu=\mathbb E f(U)$, $U$ gleichverteilt auf $[0,1]^d$. Jedes Verfahren liefert einen Schätzer $\hat\mu$ aus $N$ Auswertungen; der Faktor ist $\mathrm{MSE}(\hat\mu)/\mathrm{MSE}(\hat\mu_{\text{einfach}})$, gemessen über $R$ Wiederholungen.

**Antithetisch.** Paare $(U,1-U)$, Schätzer $\frac1{N/2}\sum\frac{f(U_i)+f(1-U_i)}2$; bei gleichen Kosten ist die Varianz $\sigma^2(1+\rho_a)/N$ mit $\rho_a=\operatorname{corr}(f(U),f(1-U))$, der Faktor also $1+\rho_a$; bei symmetrischem $f$ ist $\rho_a=1$, der Faktor 2.

**Kontrollvariate.** $\hat\mu=\bar y-\hat\beta(\bar c-\mu_c)$ mit $\hat\beta=\operatorname{cov}(y,c)/\operatorname{var}(c)$; der Faktor ist etwa $1-\rho^2$ (Stück 3).

**Latin Hypercube.** Je Koordinate genau ein Punkt je Stratum; der Gewinn ist etwa der additive Anteil der Varianz (Stück 5). **Sobol.** Eine Folge aus $N=2^k$ Punkten, Owen-Scrambling; in acht **Blöcken** zu $N/8$ Punkten gibt das t-Intervall mit 7 Freiheitsgraden eine ehrliche Fehlerangabe (Stück 9).

**Importance Sampling.** Jede Kante fällt mit $q'$ statt $q$ aus; Gewicht $(q/q')^k((1-q)/(1-q'))^{m-k}$ bei $k$ Ausfällen; das relative Varianzquadrat ist exakt $\sum_f N_f\,(q^f(1-q)^{m-f})^2/(q'^f(1-q')^{m-f})/P^2-1$ (Stück 7).

**Trefferzahl-Regel.** Mit $\hat\pi$ aus dem Probelauf: Importance Sampling, wenn $N\hat\pi<50$ (und die Aufgabe ein Netz-Ausfall ist), sonst Sobol in Blöcken.

**Probelauf.** Vorhergesagte Faktoren: Kontrollvariate $1-\hat\rho^2$, antithetisch $1+\hat\rho_a$, Latin Hypercube $1-\hat s_{\text{add}}$ mit dem um das Binrauschen bereinigten additiven Anteil, Sobol $\widehat{\operatorname{Var}}(\text{8 Wiederholungen zu 64 Punkten})/(\hat\sigma^2/64)$. Kosten $2n_0+8\cdot64$ Auswertungen.

**Irwin–Hall.** $P(\sum_{i\le d}U_i>x)=1-\frac1{d!}\sum_{k\le\lfloor x\rfloor}(-1)^k\binom dk(x-k)^d$ (exakt, mit Brüchen gerechnet).

**Literatur (Lehrbuchwissen und die Stücke 1 bis 15 der Linie, hier nicht neu gelesen).** Owen 1995/1998 (Scrambling), Joe und Kuo 2008 (Richtungszahlen), McKay, Beckman und Conover 1979 (Latin Hypercube), Hesterberg 1995 und Owen 2013 (Importance Sampling).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
