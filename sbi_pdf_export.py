"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import sbi_constants as C
from sbi_evaluation import Analysis, Settings, de, dn, idx, prozent, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "α": "alpha", "β": "beta", "θ": "theta", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": "", "ϱ": "rho", "ᵢ": "_i", "∫": "Integral", "τ": "tau"}


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
    pdf.cell(0, 9, _s("Sobol-Indizes: welche Eingabe treibt die Varianz?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    last = len(a.ns) - 1
    pdf.multi_cell(0, 5, _s(f"Integrand: {a.label}, {a.p} Eingaben; bis N = {de(a.ns[last], 0)} Zeilen ({de(a.runs[last], 0)} Modellläufe); {a.reps} Wiederholungen je N, Entwurf: {C.DESIGNS[a.design]}, Strom {settings.seed}. "
                            f"Alles simuliert, die Indizes sind exakt."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (90, 60)
    pdf.cell(widths[0], 6, _s("Kennzahl"), border=1)
    pdf.cell(widths[1], 6, _s("Wert"), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for label, val in (("größter / kleinster Erstordnungsindex (exakt)", f"{idx(a.S[a.i_s_hi])} / {idx(a.S[a.i_s_lo])}"), ("größter / kleinster Gesamtindex (exakt)", f"{idx(a.T[a.i_t_hi])} / {idx(a.T[a.i_t_lo])}"),
                       ("Summe der Gesamtindizes = mittlere Dimension (exakt)", dn(a.sums["T"][0], 2)), ("Summe der Erstordnungsindizes (exakt)", dn(a.sums["S"][0], 3)),
                       ("RMSE größter S (Jansen)", idx(a.rmse_hi["S_jansen"][last])), ("RMSE kleinster S (Jansen / Janon)", f"{idx(a.rmse_lo['S_jansen'][last])} / {idx(a.rmse_lo['S_janon'][last])}"),
                       ("RMSE kleinster Gesamtindex (Jansen)", idx(a.rmse_lo["T_jansen"][last])), ("negative Schätzungen des kleinsten S (Jansen)", prozent(a.neg_lo["S_jansen"][last])),
                       ("Läufe mit Varianz", prozent(a.defined[last]))):
        pdf.cell(widths[0], 6, _s(label), border=1)
        pdf.cell(widths[1], 6, _s(val), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: sobol-indizes-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
