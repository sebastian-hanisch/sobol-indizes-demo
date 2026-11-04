"""Kern von Hand nachgerechnet (DEMO-PLAYBOOK §1b): jede Zahl hier steht auch auf Papier."""

import numpy as np
import pytest

import sbi_algorithm as A


def _f(u):
    """f(u) = u1 + 2 u2: additiv, Var = 5/12, S = T = (1/5, 4/5)."""
    return u[:, 0] + 2 * u[:, 1]


AM = np.array([[0.2, 0.4], [0.6, 0.8]])
BM = np.array([[0.5, 0.1], [0.9, 0.3]])


def test_zufallsstrom_gleicht_der_ganzzahl_referenz():
    seed = A.stream_seed(7, 3)
    ref = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert A.uniforms(seed, 5).tolist() == ref


def test_die_ersten_vier_sobol_punkte_in_zwei_dimensionen_von_hand():
    assert (A.sobol_ints(4, 2).astype(np.float64) / 2**32).tolist() == [[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]


def test_pick_freeze_laeufe_von_hand():
    """f_A = (1,0; 2,2), f_B = (0,7; 1,5); AB_1 (Spalte 1 aus B) = ((0,5; 0,4), (0,9; 0,8)) ⇒ (1,3; 2,5); AB_2 = ((0,2; 0,1), (0,6; 0,3)) ⇒ (0,4; 1,2)."""
    fA, fB, fAB = A.pick_freeze(_f, AM, BM)
    assert fA.tolist() == pytest.approx([1.0, 2.2]) and fB.tolist() == pytest.approx([0.7, 1.5])
    assert fAB[:, 0].tolist() == pytest.approx([1.3, 2.5]) and fAB[:, 1].tolist() == pytest.approx([0.4, 1.2])
    assert A.pick_freeze(_f, AM, BM, need_B=False)[1] is None


def test_schaetzer_von_hand():
    """V = Var(1,0; 2,2; 0,7; 1,5) = 0,3225 (ohne Freiheitsgradkorrektur); Mittel von f_A = 1,6."""
    fA, fB, fAB = A.pick_freeze(_f, AM, BM)
    est = A.estimators(fA, fB, fAB)
    V = 0.3225
    assert est["S_saltelli"].tolist() == pytest.approx([0.33 / V, -0.96 / V])                       # mean(f_B (f_AB - f_A)): (0,7·0,3 + 1,5·0,3)/2 und (0,7·(-0,6) + 1,5·(-1,0))/2
    assert est["S_jansen"].tolist() == pytest.approx([1 - 0.68 / (2 * V), 1 - 0.09 / (2 * V)])      # (f_B - f_AB)²: (0,36; 1,0) und (0,09; 0,09)
    assert est["T_jansen"].tolist() == pytest.approx([0.09 / (2 * V), 0.68 / (2 * V)])              # (f_A - f_AB)²: (0,09; 0,09) und (0,36; 1,0)
    assert est["S_homma"][0] == pytest.approx((2.33 - 1.6**2) / V)                                   # mean(f_B f_AB1) = (0,91 + 3,75)/2 = 2,33
    assert est["S_janon"][0] == pytest.approx((2.33 - 1.5**2) / (2.67 - 1.5**2))                     # Mittel von (f_B + f_AB1)/2 = 1,5; Mittel der Quadrate (1,09; 4,25) = 2,67
    assert est["T_homma"][0] == pytest.approx(1 - (np.mean([1.0 * 1.3, 2.2 * 2.5]) - 1.6**2) / V)


def test_owen_schaetzer_von_hand():
    """f(u) = u, N = 1: f(x) = 0,2, f(y) = 0,6, f(z_1 : x_-1) = f(0,9) = 0,9, f(x_1 : y_-1) = f(0,2) = 0,2; (0,2 - 0,9)(0,2 - 0,6)/V = 0,28/0,04 = 7."""
    out = A.owen_estimator(lambda u: u[:, 0], np.array([[0.2]]), np.array([[0.6]]), np.array([[0.9]]))
    assert out.tolist() == pytest.approx([7.0])


def test_varianz_null_macht_alle_schaetzer_unbestimmt():
    z = np.ones(5)
    est = A.estimators(z, z, np.ones((5, 3)))
    assert all(np.isnan(v).all() and len(v) == 3 for v in est.values())
    assert np.isnan(A.owen_estimator(lambda u: np.ones(len(u)), np.zeros((4, 2)), np.ones((4, 2)), np.full((4, 2), 0.5))).all()


def test_exakte_indizes_des_pfads_von_hand(pfad, tab_pfad):
    """p = ½, X = x1 x2: c = (¼, ¼, ¼, ¼), Var = 3/16, S_i = 1/3, T_i = 2/3, Σ T = 4/3 (mittlere Dimension)."""
    S, T, var = A.exact_netz("sprung", tab_pfad, 2, 0.5)
    assert var == pytest.approx(3 / 16) and S.tolist() == pytest.approx([1 / 3, 1 / 3]) and T.tolist() == pytest.approx([2 / 3, 2 / 3]) and T.sum() == pytest.approx(4 / 3)


def test_exakte_indizes_des_begleiters_in_einer_und_zwei_dimensionen():
    S, T, var = A.exact_exp(np.array([1.0]))
    assert S.tolist() == pytest.approx([1.0]) and T.tolist() == pytest.approx([1.0]) and var == pytest.approx((np.e**2 - 1) / 2 - (np.e - 1) ** 2)
    S, T, var = A.exact_exp(np.ones(2))
    mu, m2 = np.e - 1, (np.e**2 - 1) / 2
    v = m2 - mu**2
    assert var == pytest.approx(m2**2 - mu**4) and S.tolist() == pytest.approx([v * mu**2 / var] * 2)
    assert T.tolist() == pytest.approx([1 - (mu**2 * m2 - mu**4) / var] * 2)
    assert T[0] == pytest.approx(S[0] + v**2 / var)                      # Gesamt = Erst + Wechselwirkung σ²_{12}/Var, σ²_{12} = v²


def test_indizes_aus_den_varianzanteilen_von_hand():
    """σ² über die Mengen {1}, {2}, {1,2} = (0,2; 0,3; 0,5): S = (0,2; 0,3), T = (0,7; 0,8)."""
    S, T, var = A.indices_from_sigma2(np.array([9.9, 0.2, 0.3, 0.5]), 2)
    assert var == pytest.approx(1.0) and S.tolist() == pytest.approx([0.2, 0.3]) and T.tolist() == pytest.approx([0.7, 0.8])


def test_zusammenhangstabelle_und_zaehlung_am_dreieck(tab_dreieck):
    assert A.conn_counts(tab_dreieck, 3) == [0, 0, 3, 1]
    assert tab_dreieck.tolist() == [False, False, False, True, False, True, True, True]


def test_entwurf_formen_und_die_zwei_haelften_des_owen_punktes():
    a, b = A.draw("mc", 8, 3, 5)
    assert a.shape == b.shape == (8, 3) and not np.allclose(a, b)
    a, b = A.draw("owen", 16, 3, 5)
    assert a.shape == b.shape == (16, 3)
    for M in (a, b):
        for j in range(3):
            assert sorted(np.floor(M[:, j] * 16).astype(int).tolist()) == list(range(16))          # je Spalte ein Punkt je Intervall (Netz in Einzelkoordinaten)


def test_bootstrap_intervall_von_hand_bei_konstantem_unterschied():
    """f_A = 0 und f_B = f_AB = 1 in allen Zeilen (V = 1/4): Jansen T = 0,5 (f_A - f_AB)²/V/2 = 1/(2 · 0,25) = 2, S = 1; jede Neuziehung gibt dasselbe, das Intervall ist ein Punkt."""
    fA, fB, fAB = np.zeros(50), np.ones(50), np.ones((50, 1))
    out = A.bootstrap_interval(fA, fB, fAB, 0, np.random.default_rng(1), 20)
    assert out["T"] == pytest.approx((2.0, 2.0)) and out["S"] == pytest.approx((1.0, 1.0))
    nan = A.bootstrap_interval(np.ones(5), np.ones(5), np.ones((5, 1)), 0, np.random.default_rng(1), 10)
    assert np.isnan(nan["S"][0]) and np.isnan(nan["T"][1])
