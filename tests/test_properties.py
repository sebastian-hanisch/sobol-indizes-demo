"""Eigenschaften (ORACLE-PLAYBOOK §5): was für ALLE Eingaben gelten muss, geprüft an mehreren Instanzen und mit Bändern für das Rauschen."""

import numpy as np
import pytest

import sbi_algorithm as A
import sbi_constants as C
import sbi_evaluation as ev
from sbi_evaluation import Settings
from sbi_scenario import named_netz


@pytest.mark.parametrize("key", ["hantel", "3x3"])
@pytest.mark.parametrize("p", [0.7, 0.9, 0.99])
def test_erstordnung_nie_ueber_gesamt_und_summen_liegen_auf_der_richtigen_seite_von_eins(key, p):
    prob = ev.make_problem(Settings(integrand="sprung", netz=key, p=p))
    assert (prob.S <= prob.T + 1e-12).all() and prob.T.sum() >= 1 - 1e-12 and prob.S.sum() <= 1 + 1e-12 and (prob.S >= -1e-12).all()


@pytest.mark.parametrize("w", list(C.WEIGHTS))
@pytest.mark.parametrize("d", C.D_CHOICES)
def test_begleiter_indizes_sind_geordnet(d, w):
    S, T, var = A.exact_exp(A.exp_weights(w, d))
    assert (S <= T + 1e-12).all() and var > 0 and T.sum() >= 1 and S.sum() <= 1


def test_additive_funktion_hat_erst_gleich_gesamtindex():
    """Eine Eingabe: S = T = 1; für exp(a u) in einer Dimension zusammenfallend, in zwei Dimensionen ist T größer als S (Wechselwirkung)."""
    S1, T1, _ = A.exact_exp(np.array([2.0]))
    S2, T2, _ = A.exact_exp(np.array([2.0, 2.0]))
    assert S1[0] == pytest.approx(1.0) and T1[0] == pytest.approx(1.0) and (T2 > S2).all()


def test_jansen_gesamtindex_ist_nie_negativ_und_der_erstordnungsindex_wird_negativ():
    prob = ev.make_problem(Settings(integrand="sprung", netz="3x3"))
    est = ev.run_estimators(prob, "mc", 300, 60, 3, 1)
    assert (est["T_jansen"] >= 0).all() and (est["S_jansen"] < 0).any() and (est["S_saltelli"] < 0).any()


def test_der_fehler_faellt_mit_eins_durch_wurzel_n():
    a = ev.analyse(Settings(netz="hantel", n=3000))
    r = a.rmse_hi["S_jansen"]
    assert r[-1] / r[0] == pytest.approx((100 / 3000) ** 0.5, rel=0.35)


def test_nicht_zentrierter_schaetzer_ist_deutlich_schlechter_als_jansen():
    a = ev.analyse(Settings(netz="hantel", n=1000))
    assert a.rmse_hi["S_homma"][-1] > 2.5 * a.rmse_hi["S_jansen"][-1] and a.rmse_hi["T_homma"][-1] > 2 * a.rmse_hi["T_jansen"][-1]


def test_kleine_erstordnungsindizes_verschluckt_kleine_gesamtindizes_nicht():
    a = ev.analyse(Settings(netz="3x3", n=3000))
    s_lo, t_lo = a.S[a.i_s_lo], a.T[a.i_t_lo]
    assert a.rmse_lo["S_jansen"][-1] > s_lo and a.rmse_lo["T_jansen"][-1] < 0.5 * t_lo


def test_owen_schaetzer_schlaegt_jansen_bei_kleinen_indizes_und_verliert_bei_grossen():
    ex = ev.small_index_experiment(Settings(integrand="exp", d=8, weight="harmonisch", n=1000))
    r = {x["est"]: x for x in ex["rows"]}
    assert r["S_owen"]["rmse_lo"] < r["S_jansen"]["rmse_lo"] / 3 and r["S_owen"]["rmse_hi"] > r["S_jansen"]["rmse_hi"]
    assert r["S_owen"]["runs"] <= 1.01 * r["S_jansen"]["runs"]                           # bei gleichen Läufen (bis auf die Rundung der Zeilenzahl)


def test_saltelli_ist_bei_kleinen_indizes_glatter_funktionen_besser_als_jansen():
    ex = ev.small_index_experiment(Settings(integrand="exp", d=8, weight="harmonisch", n=1000))
    r = {x["est"]: x for x in ex["rows"]}
    assert r["S_saltelli"]["rmse_lo"] < r["S_jansen"]["rmse_lo"] and r["S_saltelli"]["rmse_hi"] > r["S_jansen"]["rmse_hi"]


def test_rangfolge_wird_mit_n_besser_und_ist_billiger_als_der_wert():
    ex = ev.rank_experiment(Settings(netz="3x3"))
    p = [r["p_sep"] for r in ex["rows"]]
    assert p[0] < 0.2 and p[-1] >= 0.97 and all(b >= a - 0.1 for a, b in zip(p, p[1:])) and ex["n_hi"] == 8 and ex["n_lo"] == 4
    ex = ev.rank_experiment(Settings(integrand="exp", d=8, weight="harmonisch"))
    assert ex["rows"][1]["p_top1"] >= 0.99                                              # die wichtigste Eingabe schon bei N = 100 (900 Läufe)


def test_seltene_ausgabe_hat_laeufe_ohne_varianz_gemaess_der_formel():
    a = ev.analyse(Settings(netz="3x4", p=0.99, n=1000))
    prob = ev.make_problem(Settings(netz="3x4", p=0.99))
    tab = A.conn_table(12, named_netz("3x4").edges)
    P_fail = 1 - A.r_exact(A.conn_counts(tab, 17), 0.99)
    for N, defd in zip(a.ns, a.defined):
        erwartet = 1 - (1 - P_fail) ** (2 * N) - P_fail ** (2 * N)
        assert defd == pytest.approx(erwartet, abs=0.2), N
    assert a.defined[0] < 0.25 and prob.T.max() > 0.2


def test_summe_der_gesamtindizes_wird_erwartungstreu_geschaetzt_die_summe_der_erstordnung_ungenau():
    a = ev.analyse(Settings(netz="hantel", n=3000))
    ex_t, rows_t = a.sums["T"]
    ex_s, rows_s = a.sums["S"]
    assert rows_t[-1][0] == pytest.approx(ex_t, abs=0.05) and rows_t[-1][1] / ex_t < 0.05
    assert rows_s[-1][1] / ex_s > 0.04


def test_fester_nenner_macht_den_erstordnungsbalken_zu_breit():
    ex = ev.interval_experiment(Settings(netz="3x3", n=1000))
    r = {(x["target"], x["fixed"]): x for x in ex["rows"]}
    neu, fest = r[("größter Erstordnungsindex", False)], r[("größter Erstordnungsindex", True)]
    assert fest["halfwidth"] > 1.4 * neu["halfwidth"] and fest["coverage"] >= neu["coverage"] and 0.8 <= neu["coverage"] <= 1.0


def test_quasi_zufallszahlen_helfen_bei_glatten_integranden():
    mc = ev.analyse(Settings(integrand="glatt", netz="hantel", n=1000, design="mc"))
    ow = ev.analyse(Settings(integrand="glatt", netz="hantel", n=1000, design="owen"))
    assert ow.rmse_hi["S_saltelli"][-1] < mc.rmse_hi["S_saltelli"][-1] / 5


def test_gleiche_einstellung_gleiches_ergebnis():
    s = Settings(netz="hantel", n=300)
    a, b = ev.analyse(s), ev.analyse(s)
    assert a.rmse_hi == b.rmse_hi and a.neg_lo == b.neg_lo
    assert ev.analyse(Settings(netz="hantel", n=300, seed=s.seed + 1)).rmse_hi != a.rmse_hi


def test_kosten_gleichstand_ist_zwei_hoch_p_durch_p_plus_zwei():
    rows = {r["p"]: r for r in ev.cost_rows(1000)}
    assert rows[17]["enum"] == 131072 and rows[17]["saltelli"] == 19000 and rows[17]["breakeven"] == pytest.approx(131072 / 19)
    assert rows[7]["enum"] < rows[7]["saltelli"] and rows[12]["enum"] < rows[12]["saltelli"] and rows[17]["enum"] > rows[17]["saltelli"]
    assert rows[7]["total"] == 8000 and rows[7]["owen"] == 16000


def test_anteil_der_laeufe_mit_varianz_gleicht_der_formel_bei_vielen_wiederholungen():
    """Direkt über 400 Wiederholungen bei N = 1000: Anteil mit mindestens einem Ausfall und nicht nur Ausfällen = 1 - (1 - P)^(2N) - P^(2N)."""
    prob = ev.make_problem(Settings(netz="3x4", p=0.99))
    tab = A.conn_table(12, named_netz("3x4").edges)
    P_fail = 1 - A.r_exact(A.conn_counts(tab, 17), 0.99)
    ok = 0
    for r in range(400):
        Am, Bm = A.draw("mc", 1000, 17, A.stream_seed(5, r))
        ok += float(np.concatenate([prob.f(Am), prob.f(Bm)]).var()) > 0
    assert ok / 400 == pytest.approx(1 - (1 - P_fail) ** 2000, abs=0.07)
