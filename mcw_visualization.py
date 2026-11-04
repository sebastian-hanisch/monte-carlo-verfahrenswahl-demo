"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import mcw_constants as C

TINY = 1e-6      # Mindesthöhe sichtbarer Balken (Beschriftungen an Höhe null lösen in Plotly „-Infinity“-Meldungen aus)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def _lbl(m):
    return C.METHOD_LABELS[m]


def build_factors(a):
    """Varianzfaktor je Verfahren gegen einfach (logarithmische Achse; gestrichelt 1 = einfach, darunter ist besser)."""
    ms = [m for m in C.METHODS if m in a.factor]
    fig = go.Figure(go.Bar(x=[_lbl(m) for m in ms], y=[max(a.factor[m], TINY) for m in ms], marker_color=[C.METHOD_COLORS[m] for m in ms], name="Varianzfaktor"))
    fig.add_hline(y=1.0, line=dict(color="#222222", dash="dash"), annotation_text="einfach = 1", annotation_position="top right")
    fig.update_yaxes(title_text="Varianz gegen einfach (log)", type="log")
    fig.update_layout(showlegend=False)
    return _base(fig, 360)


def build_matrix(mat):
    """Varianzfaktor je Verfahren bei den drei Budgets (gruppierte Balken, logarithmische Achse)."""
    fig = go.Figure()
    shades = ["#9ecae1", "#4c78a8", "#1f3b63"]
    for (N, m), color in zip(mat.items(), shades):
        ms = [k for k in C.METHODS if k in m]
        fig.add_trace(go.Bar(x=[_lbl(k) for k in ms], y=[max(m[k]["factor"], TINY) for k in ms], name=f"N = {N}", marker_color=color))
    fig.add_hline(y=1.0, line=dict(color="#222222", dash="dash"), annotation_text="einfach = 1", annotation_position="top right")
    fig.update_yaxes(title_text="Varianz gegen einfach (log)", type="log")
    fig.update_layout(barmode="group")
    return _base(fig, 380)


def build_cover(cv):
    """Abdeckung des 95-%-Intervalls je Verfahren; gestrichelt das Niveau 0,95, gepunktet die Grenze 0,90."""
    ms = [m for m in C.METHODS if m in cv]
    colors = [C.COLOR_BAD if cv[m]["cover"] < C.HONEST_COVER else C.COLOR_OK for m in ms]
    fig = go.Figure(go.Bar(x=[_lbl(m) for m in ms], y=[max(cv[m]["cover"], TINY) for m in ms], marker_color=colors, name="Abdeckung"))
    fig.add_hline(y=0.95, line=dict(color="#222222", dash="dash"), annotation_text="nominell 0,95", annotation_position="top right")
    fig.add_hline(y=C.HONEST_COVER, line=dict(color="#888888", dash="dot"))
    fig.update_yaxes(title_text="Abdeckung", range=[0, 1.08])
    fig.update_layout(showlegend=False)
    return _base(fig, 340)


def build_cross(rows, N):
    """Varianzfaktor gegen die erwartete Trefferzahl N·π: Importance Sampling, Sobol in Blöcken und als eine Folge; senkrecht die Schwelle der Regel."""
    xs = [r["npi"] for r in rows]
    fig = go.Figure()
    for key, name, color, dash in (("is", "Importance Sampling", C.METHOD_COLORS["is"], "solid"), ("qmc8", "Sobol (8 Blöcke)", C.METHOD_COLORS["qmc8"], "solid"), ("qmc", "Sobol (eine Folge)", C.METHOD_COLORS["qmc"], "dot")):
        fig.add_trace(go.Scatter(x=xs, y=[max(r["factor"][key], TINY) for r in rows], mode="lines+markers", name=name, line=dict(color=color, dash=dash)))
    fig.add_vline(x=C.NPI_THRESHOLD, line=dict(color="#222222", dash="dash"), annotation_text=f"Schwelle {C.NPI_THRESHOLD:g}", annotation_position="top")
    fig.add_hline(y=1.0, line=dict(color="#888888", dash="dot"))
    fig.update_xaxes(title_text=f"erwartete Trefferzahl N·π (N = {N}, log)", type="log")
    fig.update_yaxes(title_text="Varianz gegen einfach (log)", type="log")
    return _base(fig, 380)


def build_pilots(pl):
    """Vom Probelauf vorhergesagter (Mittel über die Wiederholungen) gegen gemessenen Varianzfaktor, je Verfahren."""
    ms = [m for m in ("cv", "anti", "lhs", "qmc") if m in pl["pred"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[_lbl(m) for m in ms], y=[max(pl["pred"][m], TINY) for m in ms], name="vorhergesagt (Probelauf)", marker_color="#E08E0B"))
    fig.add_trace(go.Bar(x=[_lbl(m) for m in ms], y=[max(pl["true"][m], TINY) for m in ms], name="gemessen", marker_color=C.COLOR_OK))
    fig.add_hline(y=1.0, line=dict(color="#222222", dash="dash"), annotation_text="einfach = 1", annotation_position="top right")
    fig.update_yaxes(title_text="Varianz gegen einfach (log)", type="log")
    fig.update_layout(barmode="group")
    return _base(fig, 340)
