"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import sbi_algorithm as A
import sbi_constants as C
import sbi_evaluation as ev
from sbi_evaluation import Settings
from sbi_scenario import Netz, named_netz, parse_edges


def test_eine_eingabe_gibt_erst_gleich_gesamtindex_eins():
    """Eine einzige Eingabe: alle Varianz hängt an ihr, S = T = 1; die Schätzer laufen mit p = 1."""
    f = lambda u: u[:, 0] ** 2
    Am, Bm = A.draw("mc", 2000, 1, 3)
    est = A.estimators(*A.pick_freeze(f, Am, Bm))
    assert est["S_jansen"][0] == pytest.approx(1.0, abs=0.05) and est["T_jansen"][0] == pytest.approx(1.0, abs=0.05)


def test_kleinste_zeilenzahl():
    f = lambda u: u[:, 0] + u[:, 1]
    Am, Bm = A.draw("mc", 2, 2, 3)
    est = A.estimators(*A.pick_freeze(f, Am, Bm))
    assert all(v.shape == (2,) for v in est.values())
    assert A.draw("owen", 2, 2, 3)[0].shape == (2, 2) and A.draw("owen", 1, 2, 3)[0].shape == (1, 2)


def test_entwurf_fehler():
    with pytest.raises(ValueError):
        A.draw("halton", 4, 2, 1)
    with pytest.raises(ValueError, match="2p"):
        A.draw("owen", 4, 33, 1)
    assert A.draw("owen", 4, 32, 1)[0].shape == (4, 32)


def test_konstanter_integrand_ist_unbestimmt_statt_unsinn():
    """Ein Knoten mit einer Schleife: immer zusammenhängend, Wert 1, Varianz 0: die Schätzer sind NaN, das Verdikt benennt es, nichts bricht."""
    netz = Netz("Einzelknoten", 1, ((0, 0),), ((0.0, 0.0),))
    tab = A.conn_table(1, netz.edges)
    with pytest.raises(ValueError, match="Varianz null"):
        A.exact_netz("sprung", tab, 1, 0.9)                       # exakte Indizes brauchen Varianz > 0


def test_unmoegliche_einstellungen_werden_vorher_gemeldet():
    assert ev.available(Settings(integrand="glatt", netz="3x4")) and "bis 12 Kanten" in ev.available(Settings(integrand="glatt", netz="3x4"))
    assert ev.available(Settings(integrand="exp", d=32, design="owen")) is None and ev.available(Settings(integrand="sprung", netz="3x4", design="owen")) is None
    assert ev.available(Settings()) is None and ev.available(Settings(integrand="exp", d=8)) is None


def test_unbekannte_namen_werfen_fehler():
    with pytest.raises(ValueError):
        ev.make_problem(Settings(), key="unbekannt")
    with pytest.raises(ValueError):
        A.exp_weights("kubisch", 4)
    with pytest.raises(ValueError):
        A.exact_netz("unbekannt", np.zeros(4, dtype=bool), 2, 0.5)


def test_netz_mit_zu_vielen_kanten_wird_nicht_aufgezaehlt():
    edges = tuple((i, j) for i in range(7) for j in range(i + 1, 7))
    with pytest.raises(ValueError):
        ev.netz_table(Netz("K7", 7, edges, tuple((0.0, 0.0) for _ in range(7))))


def test_kleinste_und_groesste_n_aller_experimente():
    s = Settings(netz="hantel", n=C.N_MIN)
    assert len(ev.analyse(s).ns) == 1
    assert ev.verdict_curve(ev.analyse(s))[0] == "info" and "mindestens zwei" in ev.verdict_curve(ev.analyse(s))[1]
    assert ev.small_index_experiment(s)["rows"] and ev.rank_experiment(s)["rows"] and ev.interval_experiment(s)["rows"]


def test_seltene_ausgabe_liefert_unbestimmte_werte_ohne_absturz():
    s = Settings(netz="3x4", p=0.99, n=100)
    a = ev.analyse(s)
    assert a.defined[0] < 0.3
    kind, text = ev.verdict_main(a, s)
    assert kind == "warning" and "nur" in text
    assert ev.interval_experiment(s)["rows"] and ev.rank_experiment(s)["rows"]


def test_eingabepruefung_der_spielwiese():
    with pytest.raises(ValueError, match="Schleife"):
        parse_edges("0-0")
    with pytest.raises(ValueError, match="Keine Kanten"):
        parse_edges("\n# nur Kommentar\n")
    with pytest.raises(ValueError, match="Doppelte Kante"):
        parse_edges("0-1\n1-0")
    with pytest.raises(ValueError, match="erwartet zwei ganze Zahlen"):
        parse_edges("0-1-2")
    with pytest.raises(ValueError, match="höchstens 17"):
        parse_edges("\n".join(f"{i}-{j}" for i in range(7) for j in range(i + 1, 7)))
    with pytest.raises(ValueError, match="höchstens 14"):
        parse_edges("\n".join(f"{i}-{i + 1}" for i in range(15)))


def test_spielwiese_mit_eigenem_netz_und_beim_begleiter_der_sprung():
    netz = parse_edges("0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3")
    r = ev.custom_result(Settings(integrand="exp", n=100), netz)
    assert r["key"] == "sprung" and r["analysis"].p == 7
    assert ev.custom_result(Settings(integrand="glatt", n=100), netz)["key"] == "glatt"
    gross = parse_edges("\n".join(f"{i}-{j}" for i in range(5) for j in range(i + 1, 5)) + "\n0-5\n1-5\n2-5\n3-5")
    with pytest.raises(ValueError, match="bis 12 Kanten"):
        ev.custom_result(Settings(integrand="glatt", n=100), gross)


def test_benannte_netze_haben_die_erwartete_kantenzahl():
    assert [named_netz(k).m for k in C.NETZE] == [7, 12, 17]


def test_formate():
    assert ev.de(1234.567, 1) == "1.234,6" and ev.de(-0.5, 2) == "−0,50" and ev.de_e(0.000042) == "4,2e-05"
    assert ev.dn(float("nan")) == "nicht bestimmbar" and ev.dn(float("inf")) == "unendlich"
    assert ev.idx(0.0283) == "0,0283" and ev.idx(0.2445) == "0,244" and ev.idx(0.00002) == "2,0e-05" and ev.idx(float("nan")) == "nicht bestimmbar" and ev.idx(0.0) == "0,0000"
    assert ev.prozent(0.414) == "41 %" and ev.prozent(float("nan")) == "nicht bestimmbar" and ev.prozent(0.0123, 1) == "1,2 %"
