"""Unabhängige Orakel (ORACLE-PLAYBOOK §5): scipy für die Sobol-Folge, Brute-Force-Aufzählungen und Quadratur für die exakten Indizes, exakte Erwartungen der Schätzer-Zähler über alle Zustände.

Hier steht, was der Kern nicht selbst beweisen darf: dass die exakten Indizes stimmen (Aufzählung aller bedingten Erwartungen, Quadratur), dass die Zähler der Schätzer im Mittel genau τ_i² bzw. E Var(f | u_-i) treffen
(Summen über alle Paare bzw. Tripel von Kantenzuständen), dass Σ T die mittlere Dimension aus Stück 9 ist, und dass die Schätzer bei großem N gegen die exakten Indizes laufen."""

import numpy as np
import pytest
from scipy.stats import qmc

import sbi_algorithm as A
from sbi_scenario import grid_netz


def test_eingebettete_richtungszahlen_geben_dieselbe_folge_wie_scipy():
    for d, N in ((64, 4096), (17, 1024), (2, 8)):
        assert (A.sobol_ints(N, d).astype(np.float64) / 2.0**32 == qmc.Sobol(d, scramble=False).random(N)).all(), (d, N)


# --- exakte Indizes ---------------------------------------------------------------------------------------------------------------------

def _indizes_brute_force(tab, m, p):
    """S_i = Var(E[X | x_i])/Var und T_i = 1 - Var(E[X | x_-i])/Var, alle Erwartungen als Summen über die 2^m Zustände (unabhängig von Schmetterlingsschritt und Möbius-Inversion)."""
    states = np.arange(1 << m)
    w = np.ones(1 << m)
    for e in range(m):
        w *= np.where((states >> e) & 1 == 1, p, 1 - p)
    X = tab.astype(float)
    mean = float((w * X).sum())
    var = float((w * X * X).sum() - mean**2)

    def var_cond(keep_mask):
        key = states & keep_mask
        g_sum = np.bincount(key, weights=w * X, minlength=1 << m)
        g_w = np.bincount(key, weights=w, minlength=1 << m)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(g_w > 0, g_sum / g_w, 0.0)
        return float((g_w * g * g).sum() - mean**2)

    full = (1 << m) - 1
    S = np.array([var_cond(1 << i) / var for i in range(m)])
    T = np.array([1 - var_cond(full & ~(1 << i)) / var for i in range(m)])
    return S, T, var


@pytest.mark.parametrize("p", [0.5, 0.7, 0.9])
def test_exakte_indizes_gegen_aufzaehlung_der_bedingten_erwartungen_auf_der_hantel(hantel, tab_hantel, p):
    S, T, var = A.exact_netz("sprung", tab_hantel, hantel.m, p)
    S0, T0, var0 = _indizes_brute_force(tab_hantel, hantel.m, p)
    assert var == pytest.approx(var0, rel=1e-12) and S.tolist() == pytest.approx(S0.tolist(), abs=1e-12) and T.tolist() == pytest.approx(T0.tolist(), abs=1e-12)


def test_exakte_indizes_gegen_aufzaehlung_auf_dem_gitter_3x3(gitter33, tab_gitter33):
    S, T, var = A.exact_netz("sprung", tab_gitter33, gitter33.m, 0.9)
    S0, T0, _ = _indizes_brute_force(tab_gitter33, gitter33.m, 0.9)
    assert S.tolist() == pytest.approx(S0.tolist(), abs=1e-12) and T.tolist() == pytest.approx(T0.tolist(), abs=1e-12)


@pytest.mark.parametrize("key,m", [("hantel", 7), ("3x3", 12), ("3x4", 17)])
def test_summe_der_gesamtindizes_ist_die_mittlere_dimension_aus_stueck_9(key, m):
    from sbi_scenario import named_netz
    net = named_netz(key)
    tab = A.conn_table(net.n, net.edges)
    for kind, p in (("sprung", 0.9), ("sprung", 0.7), ("glatt", 0.9)):
        S, T, var = A.exact_netz(kind, tab, m, p)
        sig = A.anova_coeffs(tab, m, p) ** 2 if kind == "sprung" else A.glatt_sigma2(tab, m)
        d = A.dim_summary(A.order_shares(sig, m))
        assert T.sum() == pytest.approx(d["mean_dim"], abs=1e-10) and S.sum() == pytest.approx(d["additive"], abs=1e-10), (key, kind, p)
        assert (S <= T + 1e-12).all() and T.sum() >= 1 - 1e-12 and S.sum() <= 1 + 1e-12


def test_exakte_indizes_des_begleiters_gegen_quadratur():
    """exp(u1 + 0,5 u2 + 2 u3): Var(E[f | u_i]) und Var(E[f | u_-i]) numerisch aus der Gauß–Legendre-Quadratur in drei Dimensionen."""
    a = np.array([1.0, 0.5, 2.0])
    x, w = np.polynomial.legendre.leggauss(28)
    x, w = (x + 1) / 2, w / 2
    X = np.meshgrid(x, x, x, indexing="ij")
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    f = np.exp(sum(a[i] * X[i] for i in range(3)))
    mean = (W * f).sum()
    var = (W * f * f).sum() - mean**2

    def var_cond(keep):
        axes = tuple(i for i in range(3) if i not in keep)
        g = (f * W).sum(axis=axes, keepdims=True) / W.sum(axis=axes, keepdims=True)
        return float(((g * g) * W).sum() - mean**2)

    S, T, v = A.exact_exp(a)
    assert v == pytest.approx(var, rel=1e-10)
    for i in range(3):
        assert S[i] == pytest.approx(var_cond((i,)) / var, abs=1e-10)
        assert T[i] == pytest.approx(1 - var_cond(tuple(j for j in range(3) if j != i)) / var, abs=1e-10)


# --- Erwartungstreue der Zähler: exakt über alle Zustandspaare und -tripel --------------------------------------------------------------------

def _erwartung_zaehler(tab, m, p):
    """Exakte Erwartungen der Zähler als Summen über alle Paare (a, b) von Kantenzuständen (2^(2m) Zustände): Saltelli, Jansen (beide Formen), Homma-Zähler."""
    states = np.arange(1 << m)
    w = np.ones(1 << m)
    for e in range(m):
        w *= np.where((states >> e) & 1 == 1, p, 1 - p)
    X = tab.astype(float)
    a, b = np.meshgrid(states, states, indexing="ij")
    Wab = w[:, None] * w[None, :]
    out = {"saltelli": [], "jansen_S": [], "jansen_T": [], "homma": []}
    for i in range(m):
        mix = (a & ~(1 << i)) | (b & (1 << i))                            # AB_i: Zustand a, aber Bit i aus b
        fA, fB, fAB = X[a], X[b], X[mix]
        out["saltelli"].append(float((Wab * fB * (fAB - fA)).sum()))
        out["jansen_S"].append(float((Wab * (fB - fAB) ** 2).sum()) / 2)
        out["jansen_T"].append(float((Wab * (fA - fAB) ** 2).sum()) / 2)
        out["homma"].append(float((Wab * fB * fAB).sum()))
    return out, w, X


def test_zaehler_der_schaetzer_sind_im_mittel_exakt_tau_und_gesamtvarianz(hantel, tab_hantel):
    """E[f_B (f_AB - f_A)] = τ_i² = Var S_i; E[(f_B - f_AB)²]/2 = Var - τ_i²; E[(f_A - f_AB)²]/2 = Var T_i; E[f_B f_AB] = μ² + τ_i²."""
    p = 0.7
    S, T, var = A.exact_netz("sprung", tab_hantel, hantel.m, p)
    e, w, X = _erwartung_zaehler(tab_hantel, hantel.m, p)
    mu = float((w * X).sum())
    assert e["saltelli"] == pytest.approx((S * var).tolist(), abs=1e-12)
    assert e["jansen_S"] == pytest.approx(((1 - S) * var).tolist(), abs=1e-12)
    assert e["jansen_T"] == pytest.approx((T * var).tolist(), abs=1e-12)
    assert e["homma"] == pytest.approx((mu**2 + S * var).tolist(), abs=1e-12)


def test_owen_zaehler_ist_im_mittel_exakt_tau_ueber_alle_zustandstripel():
    """Hantel (7 Kanten) hat 2^21 Tripel (x, y, z); Erwartung von (f(x) - f(z_i : x_-i)) (f(x_i : y_-i) - f(y)) über alle Tripel gleich τ_i²."""
    net = __import__("sbi_scenario").hantel_netz()
    tab = A.conn_table(net.n, net.edges)
    m, p = net.m, 0.7
    S, T, var = A.exact_netz("sprung", tab, m, p)
    states = np.arange(1 << m)
    w = np.ones(1 << m)
    for e in range(m):
        w *= np.where((states >> e) & 1 == 1, p, 1 - p)
    X = tab.astype(float)
    x, y, z = np.meshgrid(states, states, states, indexing="ij")
    W3 = w[:, None, None] * w[None, :, None] * w[None, None, :]
    for i in (0, 6):
        bit = 1 << i
        zx = (x & ~bit) | (z & bit)                                      # z_i : x_-i
        xy = (y & ~bit) | (x & bit)                                      # x_i : y_-i
        val = float((W3 * (X[x] - X[zx]) * (X[xy] - X[y])).sum())
        assert val == pytest.approx(S[i] * var, abs=1e-12), i


# --- Schätzer laufen gegen die exakten Indizes ---------------------------------------------------------------------------------------------

def test_schaetzer_bei_sehr_grossem_n_gegen_die_exakten_indizes(hantel, tab_hantel):
    S, T, _ = A.exact_netz("sprung", tab_hantel, hantel.m, 0.9)
    f = lambda u: A.f_sprung(u, 0.9, tab_hantel)
    Am, Bm = A.draw("mc", 150_000, hantel.m, 5)
    est = A.estimators(*A.pick_freeze(f, Am, Bm))
    for k in ("S_saltelli", "S_jansen", "S_janon", "S_homma"):
        assert est[k].tolist() == pytest.approx(S.tolist(), abs=0.02), k
    for k in ("T_jansen", "T_homma"):
        assert est[k].tolist() == pytest.approx(T.tolist(), abs=0.02), k
    u3 = A.uniforms(7, 3 * 60_000 * hantel.m).reshape(60_000, 3 * hantel.m)
    own = A.owen_estimator(f, u3[:, :hantel.m], u3[:, hantel.m:2 * hantel.m], u3[:, 2 * hantel.m:])
    assert own.tolist() == pytest.approx(S.tolist(), abs=0.03)


def test_glatter_integrand_schaetzer_gegen_exakte_indizes(hantel, tab_hantel):
    S, T, _ = A.exact_netz("glatt", tab_hantel, hantel.m, 0.9)
    f = lambda u: A.f_glatt(u, tab_hantel)
    Am, Bm = A.draw("mc", 40_000, hantel.m, 9)
    est = A.estimators(*A.pick_freeze(f, Am, Bm))
    assert est["S_jansen"].tolist() == pytest.approx(S.tolist(), abs=0.02) and est["T_jansen"].tolist() == pytest.approx(T.tolist(), abs=0.02)


def test_begleiter_schaetzer_gegen_geschlossene_indizes():
    a = A.exp_weights("harmonisch", 8)
    S, T, _ = A.exact_exp(a)
    Am, Bm = A.draw("mc", 120_000, 8, 3)
    est = A.estimators(*A.pick_freeze(lambda u: A.f_exp(u, a), Am, Bm))
    assert est["S_jansen"].tolist() == pytest.approx(S.tolist(), abs=0.02) and est["T_jansen"].tolist() == pytest.approx(T.tolist(), abs=0.01)


def test_owen_sobol_entwurf_hat_erwartungstreue_zaehler(tab_hantel):
    """A und B als Hälften eines Owen-Sobol-Punktes: der Saltelli-Zähler mittelt über viele Randomisierungen auf τ_i² (z-Band), obwohl A und B nicht unabhängig gezogen sind."""
    m = 7
    S, T, var = A.exact_netz("sprung", tab_hantel, m, 0.9)
    f = lambda u: A.f_sprung(u, 0.9, tab_hantel)
    nums = []
    for s in range(300):
        Am, Bm = A.draw("owen", 64, m, A.stream_seed(11, s))
        fA, fB, fAB = A.pick_freeze(f, Am, Bm)
        nums.append(np.mean(fB[:, None] * (fAB - fA[:, None]), axis=0))
    nums = np.array(nums)
    se = nums.std(axis=0, ddof=1) / np.sqrt(len(nums))
    assert (np.abs(nums.mean(axis=0) - S * var) < 4.5 * se).all()


def test_zusammenhangstabelle_gegen_networkx():
    import networkx as nx
    net = grid_netz(2, 3)
    tab = A.conn_table(net.n, net.edges)
    for mask in range(0, 1 << net.m, 5):
        g = nx.Graph()
        g.add_nodes_from(range(net.n))
        g.add_edges_from(e for i, e in enumerate(net.edges) if (mask >> i) & 1)
        assert bool(tab[mask]) == nx.is_connected(g), mask
