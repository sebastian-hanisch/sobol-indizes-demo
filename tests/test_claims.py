"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung, ANOVA, geschlossene Formeln, Kosten) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen (100 Wiederholungen) und sind reproduzierbar; sie stehen hier mit Bändern, die das Rauschen
(ein einzelner RMSE ist auf etwa 10 bis 15 % genau, Anteile auf etwa ±0,05) und kleine Plattformunterschiede abdecken. RMSE und Faktoren werden im Verhältnis (mal/durch) verglichen."""

import numpy as np
import pytest

import sbi_algorithm as A
import sbi_evaluation as ev
from sbi_evaluation import Settings


def _band(wert, soll, mal=1.8):
    return soll / mal <= wert <= soll * mal


def _exakt(wert, soll):
    """Auf die in der README genannten Stellen gerundet: drei Nachkommastellen ab 0,1, sonst vier."""
    return wert == pytest.approx(soll, abs=6e-4 if soll >= 0.1 else 6e-5)


HANTEL, G33, EXP8 = Settings(netz="hantel"), Settings(netz="3x3"), Settings(integrand="exp", d=8, weight="harmonisch")


# --- exakte Indizes -----------------------------------------------------------------------------------------------------------------------

def test_exakte_indizes():
    soll = {"hantel": (HANTEL, 0.631, 0.0175, 0.668, 0.111, 0.736, 1.336), "g33": (G33, 0.0283, 0.0024, 0.225, 0.0659, 0.2365, 2.067), "exp8": (EXP8, 0.630, 0.0100, 0.658, 0.0113, 0.967, 1.034)}
    for name, (s, s_hi, s_lo, t_hi, t_lo, sum_s, sum_t) in soll.items():
        p = ev.make_problem(s)
        assert _exakt(p.S.max(), s_hi) and _exakt(p.S.min(), s_lo) and _exakt(p.T.max(), t_hi) and _exakt(p.T.min(), t_lo), name
        assert p.S.sum() == pytest.approx(sum_s, abs=0.0006) and p.T.sum() == pytest.approx(sum_t, abs=0.0006), name
    p = ev.make_problem(Settings(netz="3x4", p=0.99))
    assert p.T.max() == pytest.approx(0.244, abs=0.0006) and p.T.min() == pytest.approx(0.0026, abs=0.00006) and p.S.sum() == pytest.approx(0.0202, abs=0.00006) and p.T.sum() == pytest.approx(2.021, abs=0.0006)


def test_hantel_hat_eine_dominierende_bruecke_und_das_gitter_zwei_kantenklassen():
    p = ev.make_problem(HANTEL)
    assert p.S[6] == pytest.approx(0.631, abs=0.0006) and p.S[:6] == pytest.approx([0.0175] * 6, abs=0.0006)
    g = ev.make_problem(G33)
    assert len(set(np.round(g.T, 6))) == 2 and sorted((g.T > 0.1).sum() for _ in [0]) == [8]


# --- Genauigkeit ----------------------------------------------------------------------------------------------------------------------------

def test_fehler_faellt_mit_eins_durch_wurzel_n_auf_der_hantel():
    a = ev.analyse(Settings(netz="hantel", n=10000))
    soll = (0.111, 0.0559, 0.0320, 0.0187, 0.0103)
    for v, s in zip(a.rmse_hi["S_jansen"], soll):
        assert _band(v, s, 1.5)
    assert a.rmse_hi["S_jansen"][0] / a.rmse_hi["S_jansen"][-1] == pytest.approx(10.8, rel=0.3)


def test_kleinster_erstordnungsindex_des_gitters_wird_verschluckt():
    a = ev.analyse(Settings(netz="3x3", n=10000))
    s_lo = a.S[a.i_s_lo]
    assert s_lo == pytest.approx(0.0024, abs=0.00006)
    for k, soll in (("S_saltelli", 0.0175), ("S_jansen", 0.0134), ("S_janon", 0.0108)):
        assert _band(a.rmse_lo[k][-1], soll, 1.5) and 3.5 < a.rmse_lo[k][-1] / s_lo < 9, k
        assert 0.3 < a.neg_lo[k][-1] < 0.6, k


def test_kleine_gesamtindizes_haben_kleine_fehler_und_sind_nie_negativ():
    for s, soll, t in ((Settings(netz="3x3", n=10000), 0.0075, 0.0659), (Settings(netz="hantel", n=10000), 0.0065, 0.111), (Settings(integrand="exp", d=8, weight="harmonisch", n=10000), 0.0002, 0.0113)):
        a = ev.analyse(s)
        assert _exakt(a.T[a.i_t_lo], t)
        assert _band(a.rmse_lo["T_jansen"][-1], soll, 1.7) and a.rmse_lo["T_jansen"][-1] < 0.2 * t
    assert ev.run_estimators(ev.make_problem(G33), "mc", 300, 100, 3, 1)["T_jansen"].min() >= 0


def test_nicht_zentrierter_schaetzer_auf_der_hantel():
    a = ev.analyse(Settings(netz="hantel", n=1000))
    assert _band(a.rmse_hi["S_homma"][-1], 0.150, 1.5) and _band(a.rmse_hi["S_jansen"][-1], 0.0320, 1.5)
    assert a.rmse_hi["S_homma"][-1] / a.rmse_hi["S_jansen"][-1] == pytest.approx(4.7, rel=0.4)
    assert _band(a.rmse_hi["T_homma"][-1], 0.0976, 1.5) and _band(a.rmse_hi["T_jansen"][-1], 0.0360, 1.5)


def test_schaetzerwahl_nach_indexgroesse_am_begleiter():
    a = ev.analyse(Settings(integrand="exp", d=8, weight="harmonisch", n=1000))
    assert _band(a.rmse_hi["S_saltelli"][-1], 0.113, 1.5) and _band(a.rmse_hi["S_jansen"][-1], 0.0208, 1.5) and _band(a.rmse_hi["S_janon"][-1], 0.0197, 1.5)
    assert _band(a.rmse_lo["S_saltelli"][-1], 0.0150, 1.5) and _band(a.rmse_lo["S_jansen"][-1], 0.0289, 1.5) and _band(a.rmse_lo["S_janon"][-1], 0.0294, 1.5)
    assert a.rmse_hi["S_saltelli"][-1] > 3 * a.rmse_hi["S_jansen"][-1] and a.rmse_lo["S_saltelli"][-1] < 0.7 * a.rmse_lo["S_jansen"][-1]


# --- Owens Schätzer bei gleichen Läufen ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("s,n,lo_owen,lo_jansen,hi_owen,hi_jansen", [
    (Settings(netz="3x3", n=1000), 1000, 0.0086, 0.0432, 0.0352, 0.0589),
    (Settings(integrand="exp", d=8, weight="harmonisch", n=1000), 1000, 0.0010, 0.0292, 0.0538, 0.0202),
    (Settings(integrand="glatt", netz="hantel", n=1000), 1000, 0.0077, 0.0316, 0.0477, 0.0197),
    (Settings(netz="hantel", n=1000), 1000, 0.0152, 0.0362, 0.0829, 0.0345),
])
def test_owen_schaetzer_bei_gleichen_laeufen(s, n, lo_owen, lo_jansen, hi_owen, hi_jansen):
    r = {x["est"]: x for x in ev.small_index_experiment(s)["rows"]}
    assert _band(r["S_owen"]["rmse_lo"], lo_owen, 1.6) and _band(r["S_jansen"]["rmse_lo"], lo_jansen, 1.6) and _band(r["S_owen"]["rmse_hi"], hi_owen, 1.6) and _band(r["S_jansen"]["rmse_hi"], hi_jansen, 1.6)
    assert r["S_owen"]["rmse_lo"] < r["S_jansen"]["rmse_lo"] and r["S_owen"]["runs"] <= 1.01 * r["S_jansen"]["runs"]


def test_owen_faktoren_bei_n_10000_und_der_anteil_negativer_schaetzungen():
    for s, faktor in ((Settings(netz="3x3", n=10000), 3.9), (Settings(integrand="exp", d=8, weight="harmonisch", n=10000), 32), (Settings(netz="hantel", n=10000), 2.4)):
        r = {x["est"]: x for x in ev.small_index_experiment(s)["rows"]}
        assert r["S_jansen"]["rmse_lo"] / r["S_owen"]["rmse_lo"] == pytest.approx(faktor, rel=0.5), s
    r = {x["est"]: x for x in ev.small_index_experiment(Settings(netz="3x3", n=10000))["rows"]}
    assert r["S_owen"]["neg_lo"] < 0.05 < r["S_jansen"]["neg_lo"]


# --- Summen -------------------------------------------------------------------------------------------------------------------------------

def test_summe_der_gesamtindizes_ist_schaetzbar_die_summe_der_erstordnung_kaum():
    for s, rel_t, rel_s in ((Settings(netz="3x3", n=10000), 0.027, 0.46), (Settings(netz="hantel", n=10000), 0.013, 0.08), (Settings(integrand="exp", d=8, weight="harmonisch", n=10000), 0.012, 0.077)):
        a = ev.analyse(s)
        t_exact, t_rows = a.sums["T"]
        s_exact, s_rows = a.sums["S"]
        assert t_rows[-1][1] / t_exact == pytest.approx(rel_t, rel=0.6) and s_rows[-1][1] / s_exact == pytest.approx(rel_s, rel=0.6), s
        assert t_rows[-1][0] == pytest.approx(t_exact, abs=0.1)


# --- Rangfolge ----------------------------------------------------------------------------------------------------------------------------

def test_rangfolge_am_gitter_3x3_und_3x4():
    ex = ev.rank_experiment(G33)
    soll = (0.0, 0.04, 0.43, 0.97, 1.0)
    assert [r["runs"] for r in ex["rows"]] == [390, 1300, 3900, 13000, 39000] and ex["n_hi"] == 8 and ex["n_lo"] == 4
    for r, s in zip(ex["rows"], soll):
        assert r["p_sep"] == pytest.approx(s, abs=0.12)
    assert all(np.isnan(r["p_top1"]) for r in ex["rows"])                              # keine eindeutig wichtigste Kante
    ex = ev.rank_experiment(Settings(netz="3x4"))
    for r, s in zip(ex["rows"], (0.0, 0.02, 0.24, 0.87, 1.0)):
        assert r["p_sep"] == pytest.approx(s, abs=0.12)
    assert (ex["n_hi"], ex["n_lo"]) == (8, 9)


def test_rangfolge_am_begleiter_ist_schon_bei_30_zeilen_richtig():
    ex = ev.rank_experiment(EXP8)
    assert [r["runs"] for r in ex["rows"]][:2] == [270, 900] and all(r["p_top1"] >= 0.97 for r in ex["rows"]) and ex["n_hi"] == 1


# --- Seltene Ausgabe ----------------------------------------------------------------------------------------------------------------------

def test_seltene_ausgabe_am_gitter_3x4():
    a = ev.analyse(Settings(netz="3x4", p=0.99, n=10000))
    erwartet = (0.08, 0.22, 0.57, 0.92, 1.0)
    for d, e in zip(a.defined, erwartet):
        assert d == pytest.approx(e, abs=0.2)
    assert a.defined[0] < 0.25 and a.defined[-1] > 0.97
    assert a.T[a.i_t_hi] == pytest.approx(0.244, abs=0.0006) and a.rmse_hi["T_jansen"][-1] > 0.4 * a.T[a.i_t_hi]
    assert ev.verdict_main(a, Settings(netz="3x4", p=0.99, n=10000))[0] == "warning"


# --- Quasi-Zufallszahlen ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("s,s_faktor,t_faktor", [(Settings(netz="hantel", n=1000), 3.2, 2.8), (Settings(integrand="exp", d=8, weight="harmonisch", n=1000), 13, 4.7), (Settings(integrand="glatt", netz="hantel", n=1000), 27, 8.2)])
def test_quasi_zufallszahlen_im_entwurf(s, s_faktor, t_faktor):
    mc, ow = ev.analyse(s), ev.analyse(Settings(**{**s.__dict__, "design": "owen"}))
    assert mc.rmse_hi["S_saltelli"][-1] / ow.rmse_hi["S_saltelli"][-1] == pytest.approx(s_faktor, rel=0.5)
    assert mc.rmse_hi["T_jansen"][-1] / ow.rmse_hi["T_jansen"][-1] == pytest.approx(t_faktor, rel=0.5)


def test_quasi_zufallszahlen_helfen_beim_sprung_am_gitter_nur_wenig():
    s = Settings(netz="3x3", n=1000)
    mc, ow = ev.analyse(s), ev.analyse(Settings(netz="3x3", n=1000, design="owen"))
    assert 1.1 < mc.rmse_hi["S_saltelli"][-1] / ow.rmse_hi["S_saltelli"][-1] < 2.5


# --- Intervalle ---------------------------------------------------------------------------------------------------------------------------

def test_intervalle_am_gitter_3x3():
    ex = ev.interval_experiment(G33)
    r = {(x["target"], x["fixed"]): x for x in ex["rows"]}
    assert ex["N"] == 1000 and ex["reps"] == 100 and ex["boot"] == 100
    assert r[("größter Erstordnungsindex", False)]["coverage"] == pytest.approx(0.90, abs=0.09) and _band(r[("größter Erstordnungsindex", False)]["halfwidth"], 0.0996, 1.4)
    assert r[("größter Erstordnungsindex", True)]["coverage"] >= 0.96 and _band(r[("größter Erstordnungsindex", True)]["halfwidth"], 0.176, 1.4)
    assert r[("größter Erstordnungsindex", True)]["halfwidth"] / r[("größter Erstordnungsindex", False)]["halfwidth"] == pytest.approx(1.77, rel=0.25)
    assert r[("kleinster Erstordnungsindex", False)]["halfwidth"] > r[("kleinster Erstordnungsindex", False)]["truth"] * 10
    assert r[("kleinster Gesamtindex", False)]["coverage"] == pytest.approx(0.89, abs=0.1) and r[("größter Gesamtindex", True)]["halfwidth"] == pytest.approx(r[("größter Gesamtindex", False)]["halfwidth"], rel=0.15)


# --- Kosten (exakt) ---------------------------------------------------------------------------------------------------------------------------

def test_kosten_der_drei_netze():
    rows = {r["p"]: r for r in ev.cost_rows(1000)}
    assert (rows[7]["enum"], rows[7]["saltelli"], rows[7]["total"], rows[7]["owen"]) == (128, 9000, 8000, 16000)
    assert (rows[12]["enum"], rows[12]["saltelli"]) == (4096, 14000) and (rows[17]["enum"], rows[17]["saltelli"]) == (131072, 19000)
    assert rows[7]["breakeven"] == pytest.approx(14.2, abs=0.05) and rows[12]["breakeven"] == pytest.approx(292.6, abs=0.05) and rows[17]["breakeven"] == pytest.approx(6898.5, abs=0.05)
