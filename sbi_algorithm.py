"""Kern der Demo (nur numpy).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom    stream_seed, uniforms                                       SplitMix64, zählerbasiert
  Sobol-Folge     direction_ints, sobol_ints, owen_scramble, points           Joe-Kuo-Richtungszahlen (64 Dimensionen), Owen-Scrambling bis Tiefe ceil(log2 N) (aus Stück 9)
  Integranden     f_sprung, f_glatt, f_exp, conn_table, conn_counts           Zusammenhang eines Netzes (Sprung), Zuverlässigkeit bei unsicheren p (glatt), exp(Σ aᵢuᵢ)
  Exakte Indizes  anova_coeffs, indices_from_sigma2, exact_exp, exact_netz    Erst- und Gesamtindizes aus der ANOVA bzw. geschlossen
  Entwurf         draw, pick_freeze                                           A, B (und C): unabhängig oder als Hälften eines Owen-Sobol-Punktes; Mischungen AB_i
  Schätzer        estimators, owen_estimator, sum_estimates                   Saltelli, Jansen, Janon/Monod, Homma–Saltelli, Owen 2013
  Fehlerbalken    bootstrap_interval                                          Zeilen-Bootstrap, der den ganzen Schätzer samt Varianz im Nenner neu rechnet
Zufall kommt nur über einen übergebenen Strom, nie aus einem globalen Generator.

Pick-Freeze (Saltelli): zwei unabhängige Matrizen A und B (N x p), AB_i = A mit Spalte i aus B. Läufe f(A), f(B), f(AB_1) ... f(AB_p): (p + 2) N. Der Gesamtindex allein braucht (p + 1) N.
Erstordnungsindex S_i = Var(E[f | u_i]) / Var f; Gesamtindex T_i = E[Var(f | u_-i)] / Var f.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import sbi_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
BITS = C.BITS


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


# --- Sobol-Folge und Owen-Scrambling (aus Stück 9, dort getestet) ------------------------------------------------------------------------------
# Richtungszahlen der Koordinaten 2 bis 64 nach Joe und Kuo (2008, Tabelle new-joe-kuo-6.21201): (Polynom als Bitmuster x^s + a_1 x^(s-1) + ... + 1, Anfangswerte m_1 ... m_s).
# Koordinate 1 ist die van-der-Corput-Folge. Die Tabelle ist ein Auszug aus der Tabelle in scipy und im Test ganzzahlig gegen scipy.stats.qmc.Sobol geprüft.

JOE_KUO = (
    (3, (1,)),
    (7, (1, 3)),
    (11, (1, 3, 1)),
    (13, (1, 1, 1)),
    (19, (1, 1, 3, 3)),
    (25, (1, 3, 5, 13)),
    (37, (1, 1, 5, 5, 17)),
    (41, (1, 1, 5, 5, 5)),
    (47, (1, 1, 7, 11, 19)),
    (55, (1, 1, 5, 1, 1)),
    (59, (1, 1, 1, 3, 11)),
    (61, (1, 3, 5, 5, 31)),
    (67, (1, 3, 3, 9, 7, 49)),
    (91, (1, 1, 1, 15, 21, 21)),
    (97, (1, 3, 1, 13, 27, 49)),
    (103, (1, 1, 1, 15, 7, 5)),
    (109, (1, 3, 1, 15, 13, 25)),
    (115, (1, 1, 5, 5, 19, 61)),
    (131, (1, 3, 7, 11, 23, 15, 103)),
    (137, (1, 3, 7, 13, 13, 15, 69)),
    (143, (1, 1, 3, 13, 7, 35, 63)),
    (145, (1, 3, 5, 9, 1, 25, 53)),
    (157, (1, 3, 1, 13, 9, 35, 107)),
    (167, (1, 3, 1, 5, 27, 61, 31)),
    (171, (1, 1, 5, 11, 19, 41, 61)),
    (185, (1, 3, 5, 3, 3, 13, 69)),
    (191, (1, 1, 7, 13, 1, 19, 1)),
    (193, (1, 3, 7, 5, 13, 19, 59)),
    (203, (1, 1, 3, 9, 25, 29, 41)),
    (211, (1, 3, 5, 13, 23, 1, 55)),
    (213, (1, 3, 7, 3, 13, 59, 17)),
    (229, (1, 3, 1, 3, 5, 53, 69)),
    (239, (1, 1, 5, 5, 23, 33, 13)),
    (241, (1, 1, 7, 7, 1, 61, 123)),
    (247, (1, 1, 7, 9, 13, 61, 49)),
    (253, (1, 3, 3, 5, 3, 55, 33)),
    (285, (1, 3, 1, 15, 31, 13, 49, 245)),
    (299, (1, 3, 5, 15, 31, 59, 63, 97)),
    (301, (1, 3, 1, 11, 11, 11, 77, 249)),
    (333, (1, 3, 1, 11, 27, 43, 71, 9)),
    (351, (1, 1, 7, 15, 21, 11, 81, 45)),
    (355, (1, 3, 7, 3, 25, 31, 65, 79)),
    (357, (1, 3, 1, 1, 19, 11, 3, 205)),
    (361, (1, 1, 5, 9, 19, 21, 29, 157)),
    (369, (1, 3, 7, 11, 1, 33, 89, 185)),
    (391, (1, 3, 3, 3, 15, 9, 79, 71)),
    (397, (1, 3, 7, 11, 15, 39, 119, 27)),
    (425, (1, 1, 3, 1, 11, 31, 97, 225)),
    (451, (1, 1, 1, 3, 23, 43, 57, 177)),
    (463, (1, 3, 7, 7, 17, 17, 37, 71)),
    (487, (1, 3, 1, 5, 27, 63, 123, 213)),
    (501, (1, 1, 3, 5, 11, 43, 53, 133)),
    (529, (1, 3, 5, 5, 29, 17, 47, 173, 479)),
    (539, (1, 3, 3, 11, 3, 1, 109, 9, 69)),
    (545, (1, 1, 1, 5, 17, 39, 23, 5, 343)),
    (557, (1, 3, 1, 5, 25, 15, 31, 103, 499)),
    (563, (1, 1, 1, 11, 11, 17, 63, 105, 183)),
    (601, (1, 1, 5, 11, 9, 29, 97, 231, 363)),
    (607, (1, 1, 5, 15, 19, 45, 41, 7, 383)),
    (617, (1, 3, 7, 7, 31, 19, 83, 137, 221)),
    (623, (1, 1, 1, 3, 23, 15, 111, 223, 83)),
    (631, (1, 1, 5, 13, 31, 15, 55, 25, 161)),
    (637, (1, 1, 3, 13, 25, 47, 39, 87, 257)),
)


@lru_cache(maxsize=None)
def direction_ints(d: int) -> np.ndarray:
    """(BITS, d) uint64: V[k, j] = Richtungszahl k+1 der Koordinate j, links ausgerichtet (m_k * 2^(BITS - k)). Zwischengespeichert, schreibgeschützt."""
    if not 1 <= d <= C.MAX_DIM:
        raise ValueError(f"Dimension {d}: eingebettet sind 1 bis {C.MAX_DIM}")
    V = np.zeros((BITS, d), dtype=np.uint64)
    for j in range(d):
        if j == 0:
            for k in range(BITS):
                V[k, 0] = 1 << (BITS - 1 - k)
            continue
        poly, init = JOE_KUO[j - 1]
        s = poly.bit_length() - 1
        a = [(poly >> (s - 1 - i)) & 1 for i in range(s - 1)]     # a_1 ... a_{s-1}
        m = list(init)
        for k in range(s, BITS):
            new = m[k - s] ^ (m[k - s] << s)
            for i in range(1, s):
                if a[i - 1]:
                    new ^= m[k - i] << i
            m.append(new)
        for k in range(BITS):
            V[k, j] = m[k] << (BITS - 1 - k)
    V.setflags(write=False)
    return V


def sobol_ints(N: int, d: int) -> np.ndarray:
    """(N, d) uint64 der ersten N Sobol-Punkte in Gray-Code-Reihenfolge, links ausgerichtet (Wert = x / 2^BITS)."""
    if N < 1:
        raise ValueError("N muss mindestens 1 sein")
    V = direction_ints(d)
    idx = np.arange(N, dtype=np.uint64)
    gray = idx ^ (idx >> np.uint64(1))
    out = np.zeros((N, d), dtype=np.uint64)
    for k in range(max(1, int(N - 1).bit_length())):
        bit = ((gray >> np.uint64(k)) & np.uint64(1)).astype(bool)
        out[bit] ^= V[k][None, :]
    return out


def scramble_depth(N: int) -> int:
    """Tiefe, bis zu der gescrambelt werden muss: ceil(log2 N), mindestens 1 (ab dort sind die Präfixe einer Koordinate paarweise verschieden)."""
    return max(1, int(N - 1).bit_length())


def owen_scramble(x: np.ndarray, seed: int, depth: int) -> np.ndarray:
    """Verschachteltes Owen-Scrambling der obersten `depth` Bits: Bit j (von oben) wird mit einem Bit gekippt, das aus (Koordinate, j, Präfix der Bits darüber) per Mischfunktion kommt.

    Rückgabe: die obersten `depth` Bits als Ganzzahl (Wert = Ergebnis / 2^depth)."""
    d = x.shape[1]
    out = np.zeros_like(x)
    col = (np.arange(d, dtype=np.uint64) + np.uint64(1))[None, :] * np.uint64(0xD1342543DE82EF95)
    base = np.uint64(seed)
    for j in range(depth):
        prefix = (x >> np.uint64(BITS - j)) if j > 0 else np.zeros_like(x)
        with np.errstate(over="ignore"):
            key = _mix_arr(base + col + np.uint64(j + 1) * np.uint64(GAMMA) + _mix_arr(prefix + np.uint64(1) + (np.uint64(j) << np.uint64(40))))
        flip = (key >> np.uint64(63)) & np.uint64(1)
        bit = (x >> np.uint64(BITS - 1 - j)) & np.uint64(1)
        out |= (bit ^ flip) << np.uint64(depth - 1 - j)
    return out


# --- Netze und Integranden (aus Stück 9) ------------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile (`present` ist (B, m) bool): Kleinste-Marke-Ausbreitung, zusammenhängend genau dann, wenn alle Marken 0 sind."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def conn_table(n: int, edges, chunk: int = 1 << 15) -> np.ndarray:
    """Zusammenhang für alle 2^m Kantenmengen in Maskenreihenfolge (Kante e = Bit e)."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    tab = np.empty(1 << m, dtype=bool)
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        tab[start:start + len(masks)] = connected(((masks[:, None] >> shifts) & 1).astype(bool), n, edges)
    return tab


def popcount(m: int) -> np.ndarray:
    """Zahl der gesetzten Bits für alle Masken 0 ... 2^m - 1."""
    k = np.zeros(1 << m, dtype=np.int64)
    masks = np.arange(1 << m, dtype=np.int64)
    for e in range(m):
        k += (masks >> e) & 1
    return k


def conn_counts(tab: np.ndarray, m: int) -> list[int]:
    """N_k = Zahl der zusammenhängenden Kantenmengen mit k Kanten (k = 0 ... m)."""
    return np.bincount(popcount(m)[tab], minlength=m + 1).astype(np.int64).tolist()


def r_exact(Nk, p: float) -> float:
    """Zuverlässigkeit R(p) = Σ N_k p^k (1-p)^(m-k) bei Haltewahrscheinlichkeit p je Kante."""
    m = len(Nk) - 1
    return float(sum(c * p**k * (1 - p) ** (m - k) for k, c in enumerate(Nk)))


def f_sprung(u: np.ndarray, p: float, tab: np.ndarray) -> np.ndarray:
    """1, wenn das Netz mit den Kanten {u_e < p} zusammenhängt, sonst 0."""
    m = u.shape[1]
    mask = ((u < p).astype(np.int64) << np.arange(m, dtype=np.int64)).sum(axis=1)
    return tab[mask].astype(np.float64)


def _kron_weights(p: np.ndarray) -> np.ndarray:
    """(B, 2^k): Gewicht jedes Zustands (Bit e = Kante e hält) bei Haltewahrscheinlichkeiten p[:, e]: Produkt aus p_e (hält) und 1 - p_e (fällt aus)."""
    W = np.ones((p.shape[0], 1))
    for e in range(p.shape[1]):
        W = np.concatenate([W * (1 - p[:, e:e + 1]), W * p[:, e:e + 1]], axis=1)       # niedrigstes Bit = erste Kante
    return W


def f_glatt(u: np.ndarray, tab: np.ndarray) -> np.ndarray:
    """Zuverlässigkeit bei unsicheren Haltewahrscheinlichkeiten p_e = P_SMOOTH + DELTA_SMOOTH (u_e - 1/2): Σ_x tab[x] Π_e (p_e oder 1 - p_e), ein Polynom in u.

    Die Kanten werden in zwei Hälften geteilt: Zustand x = x1 + 2^m1 x2; die Summe ist dann (W2 @ T)·W1 zeilenweise, T = tab als (2^m2, 2^m1)-Matrix."""
    N, m = u.shape
    p = C.P_SMOOTH + C.DELTA_SMOOTH * (u - 0.5)
    m1 = m // 2
    T = tab.astype(np.float64).reshape(1 << (m - m1), 1 << m1)
    W1 = _kron_weights(p[:, :m1])
    W2 = _kron_weights(p[:, m1:])
    return ((W2 @ T) * W1).sum(axis=1)


def f_exp(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.exp(u @ a)


def exp_weights(name: str, d: int) -> np.ndarray:
    i = np.arange(1, d + 1, dtype=np.float64)
    if name == "gleich":
        return np.ones(d)
    if name == "quadratisch":
        return 2.0 / i**2
    if name == "harmonisch":
        return 1.0 / i
    raise ValueError(f"unbekannte Gewichte {name}")


def exp_exact(a: np.ndarray) -> float:
    return float(np.prod(np.expm1(a) / a))


def anova_coeffs(tab: np.ndarray, m: int, p: float) -> np.ndarray:
    """Koeffizienten c_S der Orthonormalbasis (Bernoulli(p), phi = (x-p)/sqrt(pq)) für eine Funktion der Kantenzustände; sigma_S² = c_S², c_leer = Mittelwert.

    Schmetterlingsschritt je Koordinate: (f0, f1) -> (q f0 + p f1, sqrt(pq) (f1 - f0)). Eintrag zur Maske S = Koeffizient der Menge S (Bit e gesetzt = Kante e in S)."""
    a = tab.astype(np.float64).copy()
    q = 1 - p
    s = math.sqrt(p * q)
    for e in range(m):
        a = a.reshape(-1, 2, 1 << e)
        f0, f1 = a[:, 0, :].copy(), a[:, 1, :].copy()
        a[:, 0, :] = q * f0 + p * f1
        a[:, 1, :] = s * (f1 - f0)
        a = a.reshape(-1)
    return a


def order_shares(sigma2: np.ndarray, m: int) -> np.ndarray:
    """Anteil der Varianz je Ordnung |S| = 0 ... m (sigma2[S] für S nicht leer; sigma2[0] wird ignoriert)."""
    s = sigma2.copy()
    s[0] = 0.0
    tot = s.sum()
    return np.bincount(popcount(m), weights=s, minlength=m + 1) / tot if tot > 0 else np.full(m + 1, np.nan)


def dim_summary(shares: np.ndarray) -> dict:
    """Aus den Ordnungsanteilen: additiver Anteil (Ordnung 1), mittlere Dimension, kleinste Ordnung mit 90 % und 99 % der Varianz (Überlagerungsdimension)."""
    if not np.all(np.isfinite(shares)):
        return {"additive": float("nan"), "mean_dim": float("nan"), "sup90": None, "sup99": None}
    cum = np.cumsum(shares)
    orders = np.arange(len(shares))
    return {"additive": float(shares[1]) if len(shares) > 1 else 1.0, "mean_dim": float((orders * shares).sum()),
            "sup90": int(np.searchsorted(cum, 0.90 - 1e-12)), "sup99": int(np.searchsorted(cum, 0.99 - 1e-12))}


def glatt_sigma2(tab: np.ndarray, m: int) -> np.ndarray:
    """Varianzanteile sigma_S² des glatten Integranden: c_S² r^|S| mit r = (delta²/12)/(p q) (exakt: f ist multilinear in den zentrierten u_e)."""
    c = anova_coeffs(tab, m, C.P_SMOOTH)
    r = (C.DELTA_SMOOTH**2 / 12.0) / (C.P_SMOOTH * (1 - C.P_SMOOTH))
    return c**2 * r ** popcount(m)


# --- Exakte Indizes -----------------------------------------------------------------------------------------------------------------------

def indices_from_sigma2(sig: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Erst- und Gesamtindizes aus den Varianzanteilen sigma_S² der ANOVA (sig[0] wird ignoriert): S_i = sigma_{i}²/Var, T_i = Summe über alle S mit i in S / Var."""
    s = sig.copy()
    s[0] = 0.0
    var = float(s.sum())
    if not var > 0:
        raise ValueError("Varianz null: die Indizes sind undefiniert")
    masks = np.arange(1 << m)
    S = np.array([s[1 << e] for e in range(m)]) / var
    T = np.array([s[((masks >> e) & 1) == 1].sum() for e in range(m)]) / var
    return S, T, var


def exact_netz(kind: str, tab: np.ndarray, m: int, p: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Exakte (S, T, Var) für 'sprung' (Haltewahrscheinlichkeit p; Funktion der Kantenzustände, Bernoulli-Basis) oder 'glatt' (multilinear in den zentrierten u_e)."""
    if kind == "sprung":
        return indices_from_sigma2(anova_coeffs(tab, m, p) ** 2, m)
    if kind == "glatt":
        return indices_from_sigma2(glatt_sigma2(tab, m), m)
    raise ValueError(f"unbekannter Integrand {kind}")


def exact_exp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Exakte Indizes von exp(Σ a_i u_i): S_i = (v_i/mu_i²) Π mu_j² / Var, T_i = 1 - (mu_i² Π_{j≠i} m2_j - Π mu_j²)/Var mit mu = E e^{a u}, m2 = E e^{2 a u}, v = m2 - mu²."""
    mu = np.expm1(a) / a
    m2 = np.expm1(2 * a) / (2 * a)
    v = m2 - mu**2
    p_mu2 = float(np.prod(mu**2))
    var = float(np.prod(m2)) - p_mu2
    S = (v / mu**2) * p_mu2 / var
    T = np.array([1.0 - (mu[i] ** 2 * float(np.prod(np.delete(m2, i))) - p_mu2) / var for i in range(len(a))])
    return S, T, var


# --- Entwurf (A, B, AB_i) ------------------------------------------------------------------------------------------------------------------

def owen_points(N: int, d: int, seed: int) -> np.ndarray:
    """(N, d) Owen-gescramblete Sobol-Punkte (wie `points("sobol_owen")` in Stück 9)."""
    x = sobol_ints(N, d)
    fill = uniforms(stream_seed(seed, 1), N * d).reshape(N, d)
    depth = scramble_depth(N)
    return (owen_scramble(x, stream_seed(seed, 3), depth).astype(np.float64) + fill) / float(1 << depth)


def draw(design: str, N: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A und B (je N x p): 'mc' unabhängig; 'owen' = die zwei Hälften eines 2p-dimensionalen Owen-Sobol-Punktes (Quasi-Zufallszahlen im Saltelli-Entwurf)."""
    if design == "mc":
        u = uniforms(seed, 2 * N * p).reshape(N, 2 * p)
    elif design == "owen":
        if 2 * p > C.MAX_DIM:
            raise ValueError(f"Owen-Sobol-Entwurf braucht 2p ≤ {C.MAX_DIM} Dimensionen (p = {p})")
        u = owen_points(N, 2 * p, seed)
    else:
        raise ValueError(f"unbekannter Entwurf {design}")
    return u[:, :p], u[:, p:]


def pick_freeze(f, A_pts: np.ndarray, B_pts: np.ndarray, need_B: bool = True):
    """Modellläufe: f(A), f(B) (oder None) und f(AB_i) für alle i (N x p), AB_i = A mit Spalte i aus B."""
    N, p = A_pts.shape
    fA = f(A_pts)
    fB = f(B_pts) if need_B else None
    fAB = np.empty((N, p))
    for i in range(p):
        X = A_pts.copy()
        X[:, i] = B_pts[:, i]
        fAB[:, i] = f(X)
    return fA, fB, fAB


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------

ESTIMATORS = ("S_saltelli", "S_jansen", "S_janon", "S_homma", "T_jansen", "T_homma")


def estimators(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray) -> dict[str, np.ndarray]:
    """Alle Pick-Freeze-Schätzer aus den Läufen; Wörterbuch Name -> Vektor über die Eingaben. Ist die Varianz null (alle Läufe gleich), sind alle Schätzer NaN (Index undefiniert)."""
    p = fAB.shape[1]
    var = float(np.concatenate([fA, fB]).var())
    if not var > 0:
        return {k: np.full(p, np.nan) for k in ESTIMATORS}
    f0 = float(fA.mean())
    b, a = fB[:, None], fA[:, None]
    mean_s = (b + fAB) / 2
    ms = np.mean(mean_s, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):                  # Janon: Nenner null, wenn f_B und f_AB_i in allen Zeilen gleich sind
        janon = (np.mean(b * fAB, axis=0) - ms**2) / (np.mean((b**2 + fAB**2) / 2, axis=0) - ms**2)
    return {
        "S_saltelli": np.mean(b * (fAB - a), axis=0) / var,
        "S_jansen": 1.0 - np.mean((b - fAB) ** 2, axis=0) / (2 * var),
        "S_janon": janon,
        "S_homma": (np.mean(b * fAB, axis=0) - f0**2) / var,
        "T_jansen": np.mean((a - fAB) ** 2, axis=0) / (2 * var),
        "T_homma": 1.0 - (np.mean(a * fAB, axis=0) - f0**2) / var,
    }


def owen_estimator(f, A_pts: np.ndarray, B_pts: np.ndarray, C_pts: np.ndarray) -> np.ndarray:
    """Owens Schätzer „Correlation 2“ (Owen 2013) für S_i = tau_i²/Var mit drei unabhängigen Matrizen x = A, y = B, z = C:
    mean((f(x) - f(z_i : x_-i)) (f(x_i : y_-i) - f(y)))/Var; z_i : x_-i = A mit Spalte i aus C, x_i : y_-i = B mit Spalte i aus A. Läufe: (2p + 2) N."""
    N, p = A_pts.shape
    fA, fB = f(A_pts), f(B_pts)
    var = float(np.concatenate([fA, fB]).var())
    if not var > 0:
        return np.full(p, np.nan)
    out = np.empty(p)
    for i in range(p):
        Z = A_pts.copy()
        Z[:, i] = C_pts[:, i]
        X = B_pts.copy()
        X[:, i] = A_pts[:, i]
        out[i] = np.mean((fA - f(Z)) * (f(X) - fB)) / var
    return out


def bootstrap_interval(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray, i: int, rng: np.random.Generator, B: int, fixed_denominator: bool = False) -> dict[str, tuple[float, float]]:
    """95-%-Perzentil-Intervall für S_i (Jansen) und T_i (Jansen) aus einem Zeilen-Bootstrap: Zeilen (f_A, f_B, f_AB_i) mit Zurücklegen ziehen.

    fixed_denominator=False rechnet den ganzen Schätzer (Zähler und Varianz im Nenner) je Ziehung neu; True hält die Varianz der Originalstichprobe fest (die falsche Näherung)."""
    N = len(fA)
    var0 = float(np.concatenate([fA, fB]).var())
    s_vals, t_vals = np.empty(B), np.empty(B)
    for k in range(B):
        idx = rng.integers(0, N, size=N)
        a, b, ab = fA[idx], fB[idx], fAB[idx, i]
        var = var0 if fixed_denominator else float(np.concatenate([a, b]).var())
        if not var > 0:
            s_vals[k] = t_vals[k] = np.nan
            continue
        s_vals[k] = 1.0 - np.mean((b - ab) ** 2) / (2 * var)
        t_vals[k] = np.mean((a - ab) ** 2) / (2 * var)
    out = {}
    for name, v in (("S", s_vals), ("T", t_vals)):
        v = v[np.isfinite(v)]
        out[name] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"), float("nan"))
    return out
