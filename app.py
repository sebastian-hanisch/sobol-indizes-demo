"""Sobol-Indizes: welche Eingabe treibt die Varianz? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 10 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Der Pick-Freeze-Entwurf misst mit (p + 2) N Modellläufen, welcher Anteil der Ausgabevarianz an welcher Eingabe hängt (Erstordnungs- und Gesamtindex).
Kleine Erstordnungsindizes gehen im Rauschen unter, kleine Gesamtindizes nicht; der Schätzer muss zur Indexgröße passen (Owen 2013 bei kleinen), Rangfolge ist billiger als Wert, Σ T ist die mittlere Dimension,
bei wenigen Eingaben ist die Aufzählung billiger, und bei seltenen Ausgaben hilft kein N. Vehikel sind die Netz-Integranden der Reihe und ein Begleiter exp(Σ aᵢuᵢ); alle Indizes sind exakt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import sbi_constants as C
import sbi_evaluation as ev
from sbi_evaluation import Settings, de, de_e, dn, idx, prozent
from sbi_layout import dunkles_design, kopf_css
from sbi_pdf_export import build_pdf
from sbi_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from sbi_scenario import named_netz, parse_edges
from sbi_visualization import build_cost, build_curve, build_intervals, build_netz, build_rank, build_small, build_sums

st.set_page_config(page_title="Sobol-Indizes: welche Eingabe treibt die Varianz? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(netz_key):
    return ev.netz_table(named_netz(netz_key))


@st.cache_data(show_spinner="Genauigkeit gegen N messen …")
def _analysis(settings, table):
    return ev.analyse(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Schätzer bei gleichen Läufen vergleichen …")
def _small(settings, table):
    return ev.small_index_experiment(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Rangfolge messen …")
def _rank(settings, table):
    return ev.rank_experiment(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Intervalle prüfen …")
def _intervals(settings, table):
    return ev.interval_experiment(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _netz_label(key):
    netz = named_netz(key)
    return f"{netz.name} ({netz.m} Kanten)"


def _p_label(p):
    return f"{p:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in sbi_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎛️ Sobol-Indizes: welche Eingabe treibt die Varianz?")
st.markdown(
    """
Ein **Sobol-Index** sagt, welcher Anteil der Varianz einer Ausgabe an einer Eingabe hängt: der **Erstordnungsindex** (die Eingabe allein) und der **Gesamtindex** (die Eingabe mit allen Wechselwirkungen).
Der **Pick-Freeze-Entwurf** schätzt beide aus zwei unabhängigen Zufallsmatrizen A und B und den Mischungen AB_i mit **(p + 2) N** Modellläufen. Wie genau das ist, hängt an der Größe des Index:
**kleine Erstordnungsindizes gehen im Rauschen unter, kleine Gesamtindizes nicht**, und der **Schätzer** muss zur Indexgröße passen. Eine **Rangfolge** ist billiger als ein Wert, die **Summe der Gesamtindizes** ist die mittlere Dimension aus
Stück 9, und bei wenigen Eingaben ist die **Aufzählung** billiger als jede Stichprobe. Bei **seltenen Ausgaben** hilft kein N.

Weiter unten: **🔬 fünf Experimente**, **💰 Kosten**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Pick-Freeze-Entwurf", expanded=True):
    st.markdown(
        """
1. **Zwei Matrizen:** A und B, je N Zeilen und p Spalten (eine Spalte je Eingabe), unabhängig gleichverteilt. **AB_i** ist A, aber mit der Spalte i aus B: Eingabe i ist »frisch gezogen«, alle anderen sind »eingefroren«.
2. **Läufe:** f(A), f(B) und f(AB_1), …, f(AB_p): zusammen **(p + 2) N**. Für die Gesamtindizes allein genügen f(A) und die AB_i, also (p + 1) N.
3. **Schätzer:** Hängt f kaum von u_i ab, sind f(A) und f(AB_i) fast gleich. **Gesamtindex** (Jansen): mittleres (f(A) − f(AB_i))² geteilt durch 2 Var f. **Erstordnungsindex** (Saltelli, Jansen, Janon, Owen): Kovarianz oder Abstand von f(B) und f(AB_i), die nur durch u_i verbunden sind.
4. **Warum klein schwer ist:** Der Erstordnungsschätzer ist ein Mittelwert von Termen, deren Streuung kaum davon abhängt, wie groß der Index ist: der absolute Fehler bleibt gleich, der Index schrumpft. Der Gesamtschätzer mittelt Quadrate kleiner Differenzen: sein Fehler schrumpft mit.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Integrand**")
    integrand = st.selectbox("Integrand", options("integrand_select"), format_func=C.INTEGRANDS.get, key="integrand_select",
                             help="Sprung: das Netz hängt zusammen (0 oder 1). Glatt: Zuverlässigkeit bei unsicheren Haltewahrscheinlichkeiten (ein Polynom). Begleiter: exp(Σ aᵢuᵢ) mit wählbarer Dimension.")
    netz_key = st.selectbox("Netz", options("netz_select"), format_func=_netz_label, key="netz_select", disabled=integrand == "exp",
                            help="Gitter oder die Hantel (zwei Dreiecke, durch eine Brücke verbunden). Jede Kante ist eine Eingabe, p ist die Kantenzahl.")
    p = st.select_slider("Haltewahrscheinlichkeit p je Kante", options("p_select"), key="p_select", format_func=_p_label, disabled=integrand != "sprung", help="Nur für den Sprung: bei p = 0,99 sind Ausfälle selten.")
    d = st.select_slider("Dimension d des Begleiters", options("d_select"), key="d_select", disabled=integrand != "exp", help="Zahl der Eingaben von exp(Σ aᵢuᵢ).")
    weight = st.selectbox("Gewichte aᵢ", options("weight_select"), format_func=C.WEIGHTS.get, key="weight_select", disabled=integrand != "exp",
                          help="Abfallende Gewichte: die Eingaben unterscheiden sich stark in ihrem Index (0,63 bis 0,01 bei d = 8).")
    st.markdown("**Stichprobe**")
    n = st.select_slider("Größtes N (Zeilen)", options("n_select"), key="n_select", format_func=lambda v: f"{v:,}".replace(",", "."), help="Die Genauigkeitskurve läuft von N = 100 bis hierher; die Einzelmessungen benutzen dieses N.")
    design = st.selectbox("Entwurf von A und B", options("design_select"), format_func=C.DESIGNS.get, key="design_select",
                          help="Unabhängige Zufallszahlen oder die zwei Hälften eines Owen-Sobol-Punktes (Stück 9). Bei glatten Integranden um ein Vielfaches genauer; braucht 2p ≤ 64.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"integrand_select": integrand, "netz_select": netz_key, "p_select": float(p), "d_select": int(d), "weight_select": weight, "n_select": int(n), "design_select": design, "seed_input": int(seed)})

settings = Settings(str(integrand), str(netz_key), float(p), int(d), str(weight), int(n), str(design), int(seed))
reason = ev.available(settings)
if reason:
    st.warning(reason + " Bitte eine andere Einstellung wählen.")
    st.stop()
table = _table(settings.netz)
a = _analysis(settings, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

last = len(a.ns) - 1
st.markdown("## 🎯 Ergebnis")
st.caption(f"Integrand: {a.label}; {a.p} Eingaben; Entwurf: {C.DESIGNS[a.design]}; N von {de(a.ns[0], 0)} bis {de(a.ns[last], 0)} Zeilen, {a.reps} Wiederholungen je N.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Größter Index S", idx(a.S[a.i_s_hi]), help="Exakter Erstordnungsindex der wirksamsten Eingabe: Anteil der Varianz, der an ihr allein hängt (exakte ANOVA).")
m2.metric("Kleinster Index S", idx(a.S[a.i_s_lo]), delta=f"RMSE {idx(a.rmse_lo['S_jansen'][last])}", delta_color="off", help="Der RMSE ist der des Jansen-Schätzers bei dem größten N.")
m3.metric("Kleinster Index T", idx(a.T[a.i_t_lo]), delta=f"RMSE {idx(a.rmse_lo['T_jansen'][last])}", delta_color="off", help="Mit allen Wechselwirkungen; der Jansen-Schätzer ist nie negativ.")
m4.metric("Modellläufe", de(a.runs[last], 0), delta=f"{prozent(a.defined[last])} der Läufe mit Varianz", delta_color="off", help=f"(p + 2) N Läufe für alle Erst- und Gesamtindizes bei N = {de(a.ns[last], 0)}.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Kleinster Index**")
    st.plotly_chart(build_curve(a, "lo"), width="stretch", key="main_chart_lo")
with c2:
    st.markdown("**Größter Index**")
    st.plotly_chart(build_curve(a, "hi", legend=False), width="stretch", key="main_chart_hi")
st.caption("RMSE gegen die Zeilenzahl N, doppelt logarithmisch. Die gepunktete Linie ist der Erstordnungsindex selbst, die strichpunktierte der Gesamtindex: liegt eine Kurve darüber, ist der Index vom Rauschen verschluckt. Gestrichelt: die Gesamtindex-Schätzer.")
with st.expander("Alle exakten Indizes"):
    st.dataframe({"Eingabe": list(range(a.p)), "Erstordnungsindex S": [idx(v) for v in a.S], "Gesamtindex T": [idx(v) for v in a.T]}, hide_index=True, width="stretch", key="index_table")
    st.caption(f"Summe S = {de(sum(a.S), 3)} (additiver Anteil, Stück 5); Summe T = {de(sum(a.T), 3)} (mittlere Dimension, Stück 9).")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="sobol-indizes.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie genau und zu welchem Preis misst man, welche Eingabe die Varianz treibt?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Genauigkeit gegen die Zeilenzahl")
st.caption("Die Kurven im Bild oben: sechs Schätzer, je 100 Wiederholungen mit festen Strömen. Der Fehler fällt mit 1/√N; was sich unterscheidet, ist die Höhe, und ob sie unter dem Index liegt.")
kind, text = ev.verdict_curve(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Kleine Indizes und die Wahl des Schätzers")
st.caption(f"Bei gleicher Zahl von Modellläufen: Saltelli, Jansen und Janon mit N = {de(settings.n, 0)} Zeilen und Owens Drei-Matrizen-Schätzer (2013) mit entsprechend weniger Zeilen. {C.GRID_REPS} Wiederholungen.")
if st.button("▶ Messung starten", key="run_small"):
    st.session_state["small_done"] = True
if st.session_state.get("small_done"):
    ex2 = _small(settings, table)
    st.plotly_chart(build_small(ex2), width="stretch", key="small_chart")
    st.dataframe(
        {"Schätzer": [C.LABELS[r["est"]] for r in ex2["rows"]], "Zeilen N": [de(r["N"], 0) for r in ex2["rows"]], "Modellläufe": [de(r["runs"], 0) for r in ex2["rows"]],
         "RMSE kleinster Index": [idx(r["rmse_lo"]) for r in ex2["rows"]], "RMSE größter Index": [idx(r["rmse_hi"]) for r in ex2["rows"]], "negative Schätzungen (kleinster)": [prozent(r["neg_lo"]) if r["est"].startswith("S") else "–" for r in ex2["rows"]]},
        hide_index=True, width="stretch", key="small_table",
    )
    kind, text = ev.verdict_small(ex2)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Rangfolge statt Wert")
st.caption(f"Nach dem Jansen-Gesamtindex ((p + 1) N Läufe): wie oft stehen alle wirksamen Eingaben (Gesamtindex über {de(C.RANK_FRACTION * 100, 0)} % des größten) vor allen anderen? {C.GRID_REPS} Wiederholungen je N.")
if st.button("▶ Messung starten", key="run_rank"):
    st.session_state["rank_done"] = True
if st.session_state.get("rank_done"):
    ex3 = _rank(settings, table)
    st.plotly_chart(build_rank(ex3), width="stretch", key="rank_chart")
    st.dataframe(
        {"Zeilen N": [de(r["N"], 0) for r in ex3["rows"]], "Modellläufe": [de(r["runs"], 0) for r in ex3["rows"]], "wirksame vor allen anderen": [dn(r["p_sep"], 2) for r in ex3["rows"]],
         "Rangkorrelation": [dn(r["spearman"], 2) for r in ex3["rows"]], "Läufe mit Varianz": [prozent(r["defined"]) for r in ex3["rows"]]},
        hide_index=True, width="stretch", key="rank_table",
    )
    kind, text = ev.verdict_rank(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Die Summen der Indizes")
st.caption("Die Summe der Gesamtindizes ist die mittlere Dimension aus Stück 9, die Summe der Erstordnungsindizes der additive Anteil aus Stück 5. Geschätzt aus denselben Läufen wie oben (Gesamtindex: Jansen, Erstordnung: Janon); Balken: RMSE.")
st.plotly_chart(build_sums(a), width="stretch", key="sums_chart")
kind, text = ev.verdict_sums(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Der ehrliche Fehlerbalken")
st.caption(f"Zeilen-Bootstrap (Perzentil, {C.BOOT} Ziehungen) für den Jansen-Schätzer bei N = {de(min(settings.n, C.CI_N_MAX), 0)}: einmal mit dem ganzen Schätzer inklusive Varianz im Nenner neu gerechnet, einmal mit festem Nenner. {C.CI_REPS} Wiederholungen.")
if st.button("▶ Messung starten", key="run_intervals"):
    st.session_state["intervals_done"] = True
if st.session_state.get("intervals_done"):
    ex5 = _intervals(settings, table)
    st.plotly_chart(build_intervals(ex5), width="stretch", key="ci_chart")
    st.dataframe(
        {"Index": [r["target"] for r in ex5["rows"]], "Nenner": ["fest" if r["fixed"] else "neu gerechnet" for r in ex5["rows"]], "exakter Wert": [idx(r["truth"]) for r in ex5["rows"]],
         "Überdeckung": [dn(r["coverage"], 3) for r in ex5["rows"]], "mittlere Halbbreite": [idx(r["halfwidth"]) for r in ex5["rows"]]},
        hide_index=True, width="stretch", key="ci_table",
    )
    kind, text = ev.verdict_intervals(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Kosten --------------------------------------------------------------------------------------------------------------------------------

st.subheader("💰 Kosten: Aufzählung gegen Pick-Freeze")
st.caption("Exakt (kein Zufall): Modellläufe für die drei Netze. Bei diskreten Eingaben genügen 2^p Läufe für alle Indizes ohne jeden Fehler; Pick-Freeze lohnt sich erst, wenn 2^p die Läufe weit übersteigt.")
cost = ev.cost_rows(settings.n)
st.plotly_chart(build_cost(cost), width="stretch", key="cost_chart")
st.dataframe(
    {"Netz": [r["netz"] for r in cost], "Eingaben p": [r["p"] for r in cost], "Aufzählung 2^p": [de(r["enum"], 0) for r in cost], "nur Gesamtindizes (p + 1) N": [de(r["total"], 0) for r in cost],
     "Pick-Freeze (p + 2) N": [de(r["saltelli"], 0) for r in cost], "Owen 2013 (2p + 2) N": [de(r["owen"], 0) for r in cost], "Gleichstand bei N": [de(r["breakeven"], 0) for r in cost]},
    hide_index=True, width="stretch", key="cost_table",
)
kind, text = ev.verdict_cost(cost, settings.n)
getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; Integrand (beim Begleiter: der Sprung), p, N, Entwurf und Strom stammen aus der Seitenleiste. "
        "Es zeigt die Genauigkeitskurve und die exakten Indizes."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Größter Erstordnungsindex", idx(ca.S[ca.i_s_hi]))
        s2.metric("Kleinster Erstordnungsindex", idx(ca.S[ca.i_s_lo]))
        s3.metric("Summe der Gesamtindizes", dn(sum(ca.T), 2))
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        st.plotly_chart(build_curve(ca, "lo"), width="stretch", key="custom_curve")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Index ist nicht zu klein** | Erstordnungsindizes unter dem Schätzfehler gehen im Rauschen unter und werden zur Hälfte negativ geschätzt (Experimente 1 und 2). | – |
| **Der Schätzer passt zur Indexgröße** | Für große Indizes sind Jansen und Janon besser, für kleine Owens Drei-Matrizen-Schätzer; Homma–Saltelli ohne Zentrierung ist überall schlechter (Experiment 2). | – |
| **Eine Rangfolge reicht** | Wer nur wissen will, welche Eingaben wirken, braucht deutlich weniger Läufe als für den Wert (Experiment 3). | – |
| **Die Ausgabe variiert in der Stichprobe** | Bei seltenen Ereignissen (Gitter 3×4, p = 0,99) hat ein großer Teil der Läufe keine Varianz, die Indizes sind undefiniert und auch die definierten sind weit weg (Preset »Seltene Ausgabe«). | Stück 7 (Importance Sampling) |
| **Der Fehlerbalken rechnet den Nenner mit** | Mit festem Nenner wird der Balken für Erstordnungsindizes etwa doppelt so breit (Experiment 5). | Stück 15 (Bootstrap) |
| **Die Eingaben sind unabhängig** | Für abhängige Eingaben sind diese Indizes nicht mehr eindeutig; hier nicht behandelt. | – |
| **Wenige Eingaben, diskret** | Dann ist die Aufzählung billiger und fehlerfrei (Kosten). | Stück 9 (exakte ANOVA) |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Integranden sind **erfunden**; die Indizes sind exakt (ANOVA aus der Aufzählung aller Kantenmengen bzw. geschlossene Formeln). Die Eingaben sind unabhängig. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten; der glatte Integrand nur bis 12. Gemessen sind die Hantel, die Gitter 3×3 und 3×4 und der Begleiter mit d bis 32. Es gibt keine Indizes zweiter Ordnung, keine Gruppenindizes und keine abhängigen Eingaben.
Owens Schätzer läuft mit unabhängigen Zufallszahlen (3p Dimensionen), auch wenn A und B quasi-zufällig sind. Die Intervalle sind nur für den Jansen-Schätzer gerechnet und nur bis N = 1000.

**Bekannte Probleme und Unsicherheiten.** Messungen mit wenigen Wiederholungen (100) rauschen: ein einzelner RMSE ist auf etwa 10 % genau. Janon und Monod (2014) und Saltelli (2010) sind nur als Formeln aus dem Gedächtnis eingebaut; Owen (2013) ist in Abschnitt 3 gelesen, die übrigen
Abschnitte nicht. Die Literaturangaben im Mathematik-Expander stammen sonst aus dem Gedächtnis. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Indizes.** Für $f$ mit unabhängigen Eingaben $u_1,\dots,u_p$ ist $\operatorname{Var} f=\sum_{S\ne\emptyset}\sigma_S^2$ (ANOVA). Erstordnung: $S_i=\operatorname{Var}(\mathbb E[f\mid u_i])/\operatorname{Var}f=\sigma_{\{i\}}^2/\sigma^2$. Gesamt: $T_i=\mathbb E[\operatorname{Var}(f\mid u_{-i})]/\operatorname{Var}f=\sum_{S\ni i}\sigma_S^2/\sigma^2$.
Es gilt $S_i\le T_i$, $\sum_i S_i\le 1\le\sum_i T_i$ und $\sum_i T_i=\sum_S|S|\sigma_S^2/\sigma^2$, die **mittlere Dimension**.

**Pick-Freeze.** $A,B\in[0,1]^{N\times p}$ unabhängig, $AB_i$ = $A$ mit Spalte $i$ aus $B$. Mit $f_A=f(A)$ usw. und $V=\operatorname{Var}$ der $2N$ Werte von $f_A,f_B$:
$$\hat S_i^{\text{Saltelli}}=\tfrac1V\,\overline{f_B\,(f_{AB_i}-f_A)},\qquad \hat S_i^{\text{Jansen}}=1-\tfrac1{2V}\,\overline{(f_B-f_{AB_i})^2},\qquad \hat T_i^{\text{Jansen}}=\tfrac1{2V}\,\overline{(f_A-f_{AB_i})^2}.$$
Janon/Monod: Korrelation von $f_B$ und $f_{AB_i}$ mit symmetrisierten Mitteln; Homma–Saltelli: $(\overline{f_Bf_{AB_i}}-\bar f_A^{\,2})/V$ (nicht zentriert im Zähler). Läufe: $(p+2)N$ bzw. $(p+1)N$ für $T$ allein.

**Owen 2013.** Mit drei unabhängigen Matrizen $x,y,z$: $\hat\tau_i^2=\overline{\bigl(f(x)-f(z_i{:}x_{-i})\bigr)\bigl(f(x_i{:}y_{-i})-f(y)\bigr)}$, $\hat S_i=\hat\tau_i^2/V$; Läufe $(2p+2)N$. Ist $f$ von $u_i$ kaum abhängig, sind beide Faktoren klein: der Fehler schrumpft mit dem Index.

**Exakt.** Netz-Integranden: $f$ hängt von $u_e$ nur über $x_e=\mathbf 1[u_e<p]$ ab; in der Basis $(x_e-p)/\sqrt{p(1-p)}$ liefert ein Schmetterlingsschritt je Kante alle $2^p$ Koeffizienten $c_S$, $\sigma_S^2=c_S^2$. Glatt: $\sigma_S^2=c_S^2\,(\delta^2/(12\bar p\bar q))^{|S|}$.
Begleiter $\exp(\sum a_iu_i)$ mit $\mu_j=\mathbb E e^{a_ju}$, $m_j=\mathbb E e^{2a_ju}$: $S_i=\frac{(m_i-\mu_i^2)\prod_{j\ne i}\mu_j^2}{\prod_j m_j-\prod_j\mu_j^2}$, $T_i=1-\frac{\mu_i^2\prod_{j\ne i}m_j-\prod_j\mu_j^2}{\prod_j m_j-\prod_j\mu_j^2}$.

**Intervall.** Zeilen-Bootstrap: Zeilen $(f_A,f_B,f_{AB_i})_k$ mit Zurücklegen ziehen und den ganzen Schätzer samt $V$ neu rechnen; Perzentile 2,5 % und 97,5 %. Hält man $V$ fest, ignoriert man die Kopplung von Zähler und Nenner.

**Literatur (aus dem Gedächtnis, wenn nicht anders vermerkt).** Sobol 1993, „Sensitivity estimates for nonlinear mathematical models“; Homma und Saltelli 1996, „Importance measures in global sensitivity analysis of nonlinear models“ (Reliability Engineering and System Safety); Jansen 1999, „Analysis of variance designs for model output“ (Computer Physics Communications); Saltelli, Annoni, Azzini, Campolongo, Ratto und Tarantola 2010, „Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index“ (Computer Physics Communications); Janon, Klein, Lagnoux, Nodet und Prieur 2014, „Asymptotic normality and efficiency of two Sobol index estimators“ (ESAIM: Probability and Statistics); Liu und Owen 2006, „Estimating mean dimensionality of analysis of variance decompositions“ (Journal of the American Statistical Association); Owen 2013, „Better estimation of small Sobol' sensitivity indices“ (ACM Transactions on Modeling and Computer Simulation; arXiv 1204.4763, Abstract und Abschnitt 3 eingesehen); Azzini, Mara und Rosati 2020, „Monte Carlo estimators of first- and total-orders Sobol' indices“ (arXiv 2006.08232, nur Abstract eingesehen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
