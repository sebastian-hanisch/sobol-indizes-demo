"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sbi_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def build_curve(a, which="lo", legend=True):
    """RMSE gegen N für den kleinsten (lo) oder größten (hi) Index; gestrichelt die Gesamtindizes; die waagerechte Linie ist der Index selbst (darüber: verschluckt)."""
    fig = go.Figure()
    data = a.rmse_lo if which == "lo" else a.rmse_hi
    for k in C.MAIN_ESTIMATORS:
        ys = [v if np.isfinite(v) and v > 0 else None for v in data[k]]
        fig.add_trace(go.Scatter(x=list(a.ns), y=ys, mode="lines+markers", name=C.LABELS[k], line=dict(color=C.COLORS[k], dash="dash" if k.startswith("T") else "solid")))
    s_val = a.S[a.i_s_lo] if which == "lo" else a.S[a.i_s_hi]
    t_val = a.T[a.i_t_lo] if which == "lo" else a.T[a.i_t_hi]
    fig.add_hline(y=max(s_val, 1e-12), line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text=f"Erstordnungsindex {s_val:.4f}".replace(".", ","), annotation_position="top left")
    if abs(t_val - s_val) > 1e-12:
        fig.add_hline(y=max(t_val, 1e-12), line=dict(color=C.COLOR_TRUTH, dash="dashdot", width=1), annotation_text=f"Gesamtindex {t_val:.4f}".replace(".", ","), annotation_position="bottom left")
    fig.update_xaxes(title_text="Zeilen N (Modellläufe (p + 2) N)", type="log", tickvals=list(a.ns), ticktext=[str(n) for n in a.ns])
    fig.update_yaxes(title_text="RMSE", type="log")
    fig = _base(fig, 420)
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, font=dict(size=10)), margin=dict(l=10, r=10, t=30, b=10))
    else:
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def build_small(ex):
    """RMSE der vier Erstordnungsschätzer bei gleichen Läufen für den kleinsten und den größten Index."""
    rows = [r for r in ex["rows"] if r["est"].startswith("S")]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[C.LABELS[r["est"]] for r in rows], y=[max(r["rmse_lo"], 1e-12) for r in rows], name=f"kleinster Index ({ex['S_lo']:.4f})".replace(".", ","), marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=[C.LABELS[r["est"]] for r in rows], y=[max(r["rmse_hi"], 1e-12) for r in rows], name=f"größter Index ({ex['S_hi']:.4f})".replace(".", ","), marker_color=C.COLOR_OK))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="RMSE", type="log")
    return _base(fig, 340)


def build_rank(ex):
    """Anteil der Läufe mit richtiger Trennung gegen die Zahl der Modellläufe."""
    rows = ex["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["runs"] for r in rows], y=[r["p_sep"] for r in rows], mode="lines+markers", name="wirksame vor unwirksamen", line=dict(color=C.COLOR_OK)))
    fig.add_trace(go.Scatter(x=[r["runs"] for r in rows], y=[r["spearman"] for r in rows], mode="lines+markers", name="Rangkorrelation (Spearman)", line=dict(color=C.COLORS["S_saltelli"], dash="dash")))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text="95 %", annotation_position="bottom right")
    fig.update_xaxes(title_text="Modellläufe (p + 1) N", type="log")
    fig.update_yaxes(title_text="Anteil bzw. Korrelation", range=[0, 1.05])
    return _base(fig, 330)


def build_sums(a):
    """Geschätzte Summen der Gesamtindizes (links) und der Erstordnungsindizes (rechts) mit RMSE-Balken gegen den exakten Wert."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Σ Gesamtindizes = mittlere Dimension", "Σ Erstordnungsindizes = additiver Anteil"), horizontal_spacing=0.12)
    for col, key, color in ((1, "T", C.COLORS["T_jansen"]), (2, "S", C.COLORS["S_janon"])):
        exact, rows = a.sums[key]
        fig.add_trace(go.Scatter(x=list(a.ns), y=[r[0] for r in rows], mode="markers+lines", error_y=dict(type="data", array=[r[1] if np.isfinite(r[1]) else 0 for r in rows], visible=True), marker=dict(color=color), line=dict(color=color),
                                 showlegend=False, name="Schätzung ± RMSE"), row=1, col=col)
        fig.add_hline(y=exact, line=dict(color=C.COLOR_TRUTH, dash="dot"), row=1, col=col)
        fig.update_xaxes(type="log", tickvals=list(a.ns), ticktext=[str(n) for n in a.ns], title_text="N", row=1, col=col)
    fig = _base(fig, 320)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_intervals(ex):
    """Überdeckung des 95-%-Bootstrap-Intervalls je Index, Nenner neu gerechnet gegen fester Nenner."""
    targets = []
    for r in ex["rows"]:
        if r["target"] not in targets:
            targets.append(r["target"])
    fig = go.Figure()
    for fixed, name, color in ((False, "Nenner neu gerechnet", C.COLOR_OK), (True, "Nenner fest", C.COLOR_BAD)):
        ys = [next(r["coverage"] for r in ex["rows"] if r["target"] == t and r["fixed"] == fixed) for t in targets]
        fig.add_trace(go.Bar(x=targets, y=ys, name=name, marker_color=color, text=[f"{y:.2f}".replace(".", ",") for y in ys], textposition="outside"))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text="95 %", annotation_position="bottom right")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Überdeckung", range=[0, 1.15])
    return _base(fig, 330)


def build_cost(rows):
    """Modellläufe: Aufzählung gegen Pick-Freeze, nur Gesamtindizes und Owen 2013."""
    fig = go.Figure()
    for key, label, color in (("enum", "Aufzählung 2^p", C.COLOR_TRUTH), ("total", "nur Gesamtindizes (p + 1) N", C.COLORS["T_jansen"]), ("saltelli", "Pick-Freeze (p + 2) N", C.COLORS["S_saltelli"]), ("owen", "Owen 2013 (2p + 2) N", C.COLORS["S_owen"])):
        fig.add_trace(go.Bar(x=[f"{r['netz']} (p = {r['p']})" for r in rows], y=[r[key] for r in rows], name=label, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Modellläufe", type="log")
    return _base(fig, 330)


def build_netz(netz):
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
