"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import mcw_constants as C
from mcw_evaluation import Analysis, Settings, de, fac, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "sigma", "μ": "mu", "ρ": "rho", "α": "alpha", "β": "beta", "γ": "gamma", "θ": "theta", "τ": "tau", "ε": "eps", "π": "pi", "φ": "phi", "δ": "delta", "∞": "unendlich", "Σ": "Summe", "Δ": "Delta",
            "½": "1/2", "̂": "", "ᵢ": "_i", "∫": "Integral", "⁺": "+", "′": "'", "⁻": "-", "¹": "1", "≠": "!=", "∝": "~", "ₐ": "_a", "₀": "_0"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Verfahrenswahl: welcher Schätzer wann?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{a.label}; Wahrheit {de(a.truth, 4)}; N = {a.N} Auswertungen je Verfahren, {a.R} Wiederholungen, Strom {settings.seed}. Alles simuliert, die Wahrheit ist exakt. Varianz gegen einfaches Monte Carlo (kleiner ist besser):"),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    ms = [m for m in C.METHODS if m in a.factor]
    pdf.set_font("Helvetica", "B", 9)
    for h, w in (("Verfahren", 60), ("Faktor", 30), ("mittl. quadr. Fehler", 45)):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for m in ms:
        for text, w in ((C.METHOD_LABELS[m], 60), (fac(a.factor[m]), 30), (f"{a.mse[m]:.3e}", 45)):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: monte-carlo-verfahrenswahl-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
