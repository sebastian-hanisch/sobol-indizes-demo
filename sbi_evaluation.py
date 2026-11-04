"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Gemessen wird, wie genau Pick-Freeze-Schätzer die Erst- und Gesamtindizes treffen; Wahrheit ist exakt (ANOVA aus der Aufzählung aller Kantenmengen bzw. geschlossene Formeln).
RMSE und Verzerrung gelten je für den größten und den kleinsten exakten Index; „verschluckt“ heißt: der RMSE ist größer als der Index selbst.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np

import sbi_algorithm as A
import sbi_constants as C
from sbi_scenario import Netz, named_netz

SMOOTH_EDGE_LIMIT = 12          # der glatte Integrand wertet je Lauf 2^(m/2)-Matrizen aus; darüber wird die Messung zu langsam


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def idx(x: float) -> str:
    """Index oder RMSE: vier Nachkommastellen, bei sehr kleinen Werten wissenschaftlich."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if abs(x) < 1e-4 and x != 0:
        return de_e(x)
    return de(x, 4 if abs(x) < 0.1 else 3)


def prozent(x: float, nd: int = 0) -> str:
    return "nicht bestimmbar" if np.isnan(x) else f"{de(x * 100, nd)} %"


@dataclass(frozen=True)
class Settings:
    integrand: str = C.DEFAULT_INTEGRAND
    netz: str = C.DEFAULT_NETZ
    p: float = C.DEFAULT_P
    d: int = C.DEFAULT_D
    weight: str = C.DEFAULT_WEIGHT
    n: int = C.DEFAULT_N
    design: str = C.DEFAULT_DESIGN
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Problem:
    key: str
    label: str
    p: int                                # Zahl der Eingaben
    S: np.ndarray                         # exakte Erstordnungsindizes
    T: np.ndarray                         # exakte Gesamtindizes
    var: float
    f: Callable = dataclasses.field(default=None, compare=False)


@dataclass(frozen=True)
class Analysis:
    label: str
    p: int
    S: tuple
    T: tuple
    var: float
    i_s_hi: int
    i_s_lo: int
    i_t_hi: int
    i_t_lo: int
    ns: tuple                             # Zeilenzahlen N der Genauigkeitskurve
    runs: tuple                           # Modellläufe (p + 2) N je N
    defined: tuple                        # Anteil der Läufe mit Varianz > 0 je N
    rmse_hi: dict                         # Schätzer -> RMSE für den größten Index je N
    rmse_lo: dict                         # Schätzer -> RMSE für den kleinsten Index je N
    neg_lo: dict                          # Schätzer -> Anteil negativer Schätzungen des kleinsten Index je N (nur Erstordnung)
    sums: dict                            # {"T": (Summe exakt, [(Mittel, RMSE) je N]), "S": (...)}
    design: str
    reps: int


# --- Probleme ---------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz):
    """Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten): Zusammenhangstabelle und Zählung N_k."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    tab = A.conn_table(netz.n, netz.edges)
    return tab, A.conn_counts(tab, netz.m)


def available(settings: Settings, netz: Netz | None = None) -> str | None:
    """None, wenn die Einstellung gerechnet werden kann, sonst der Grund in einem Satz."""
    if settings.integrand == "exp":
        if settings.design == "owen" and 2 * settings.d > C.MAX_DIM:
            return f"Der Owen-Sobol-Entwurf braucht 2·d ≤ {C.MAX_DIM} Dimensionen."
        return None
    netz = netz if netz is not None else named_netz(settings.netz)
    if settings.integrand == "glatt" and netz.m > SMOOTH_EDGE_LIMIT:
        return f"Der glatte Integrand wird nur bis {SMOOTH_EDGE_LIMIT} Kanten gerechnet ({netz.name} hat {netz.m})."
    if settings.design == "owen" and 2 * netz.m > C.MAX_DIM:
        return f"Der Owen-Sobol-Entwurf braucht 2p ≤ {C.MAX_DIM} Dimensionen ({netz.name} hat p = {netz.m})."
    return None


def make_problem(settings: Settings, netz: Netz | None = None, table=None, key: str | None = None) -> Problem:
    """Der gewählte Integrand mit exakten Indizes."""
    key = key or settings.integrand
    if key == "exp":
        a = A.exp_weights(settings.weight, settings.d)
        S, T, var = A.exact_exp(a)
        return Problem("exp", f"Begleiter exp(Σ aᵢuᵢ), {C.WEIGHTS[settings.weight]}, d = {settings.d}", settings.d, S, T, var, lambda u, a=a: A.f_exp(u, a))
    netz = netz if netz is not None else named_netz(settings.netz)
    tab, Nk = table if table is not None else netz_table(netz)
    m = netz.m
    if key == "sprung":
        S, T, var = A.exact_netz("sprung", tab, m, settings.p)
        return Problem("sprung", f"Sprung: {netz.name} hängt zusammen (p = {de(settings.p, 2)})", m, S, T, var, lambda u, p=settings.p, tab=tab: A.f_sprung(u, p, tab))
    if key == "glatt":
        S, T, var = A.exact_netz("glatt", tab, m, settings.p)
        return Problem("glatt", f"glatt: Zuverlässigkeit {netz.name} bei unsicheren p", m, S, T, var, lambda u, tab=tab: A.f_glatt(u, tab))
    raise ValueError(f"unbekannter Integrand {key}")


# --- Messung ----------------------------------------------------------------------------------------------------------------------------

def _rmse(err: np.ndarray) -> float:
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")


def run_estimators(prob: Problem, design: str, N: int, reps: int, seed: int, tag: int) -> dict:
    """reps Pick-Freeze-Läufe mit festen Strömen: Schätzer -> Matrix (reps x p); NaN, wo die Stichprobe keine Varianz hat."""
    out = {k: np.empty((reps, prob.p)) for k in A.ESTIMATORS}
    for r in range(reps):
        Am, Bm = A.draw(design, N, prob.p, A.stream_seed(seed, tag, N, r))
        fA, fB, fAB = A.pick_freeze(prob.f, Am, Bm)
        for k, v in A.estimators(fA, fB, fAB).items():
            out[k][r] = v
    return out


def ranks_of(S: np.ndarray, T: np.ndarray):
    return int(np.argmax(S)), int(np.argmin(S)), int(np.argmax(T)), int(np.argmin(T))


def analyse(settings: Settings, netz: Netz | None = None, table=None, key: str | None = None) -> Analysis:
    """Genauigkeit gegen N (N = 100 bis settings.n): RMSE für den größten und kleinsten Index, Anteil negativer Schätzungen und die Summen der Indizes."""
    prob = make_problem(settings, netz, table, key)
    iSh, iSl, iTh, iTl = ranks_of(prob.S, prob.T)
    ns = tuple(N for N in C.N_CHOICES if N <= settings.n)
    rmse_hi = {k: [] for k in A.ESTIMATORS}
    rmse_lo = {k: [] for k in A.ESTIMATORS}
    neg_lo = {k: [] for k in A.ESTIMATORS if k.startswith("S")}
    sum_t, sum_s, defined = [], [], []
    for N in ns:
        est = run_estimators(prob, settings.design, N, C.REPS, settings.seed, 1)
        defined.append(float(np.mean(np.isfinite(est["T_jansen"][:, 0]))))
        for k, v in est.items():
            tr = prob.S if k.startswith("S") else prob.T
            hi, lo = (iSh, iSl) if k.startswith("S") else (iTh, iTl)
            rmse_hi[k].append(_rmse(v[:, hi] - tr[hi]))
            rmse_lo[k].append(_rmse(v[:, lo] - tr[lo]))
            if k in neg_lo:
                col = v[:, lo][np.isfinite(v[:, lo])]
                neg_lo[k].append(float(np.mean(col < 0)) if len(col) else float("nan"))
        st, ss = est["T_jansen"].sum(axis=1), est["S_janon"].sum(axis=1)
        sum_t.append((float(np.nanmean(st)) if np.isfinite(st).any() else float("nan"), _rmse(st - prob.T.sum())))
        sum_s.append((float(np.nanmean(ss)) if np.isfinite(ss).any() else float("nan"), _rmse(ss - prob.S.sum())))
    return Analysis(prob.label, prob.p, tuple(prob.S), tuple(prob.T), prob.var, iSh, iSl, iTh, iTl, ns, tuple((prob.p + 2) * N for N in ns), tuple(defined),
                    {k: tuple(v) for k, v in rmse_hi.items()}, {k: tuple(v) for k, v in rmse_lo.items()}, {k: tuple(v) for k, v in neg_lo.items()},
                    {"T": (float(prob.T.sum()), tuple(sum_t)), "S": (float(prob.S.sum()), tuple(sum_s))}, settings.design, C.REPS)


# --- Experiment 2: kleine Indizes und die Wahl des Schätzers --------------------------------------------------------------------------------

def small_index_experiment(settings: Settings, netz: Netz | None = None, table=None) -> dict:
    """Bei gleicher Zahl von Modellläufen: Saltelli, Jansen, Janon (je (p + 2) N) und Owen 2013 (3 Matrizen, (2p + 2) N Läufe, N_owen = N (p + 2)/(2p + 2)) für den größten und kleinsten Erstordnungsindex;
    dazu der Gesamtindex nach Jansen für den kleinsten Gesamtindex."""
    prob = make_problem(settings, netz, table)
    iSh, iSl, iTh, iTl = ranks_of(prob.S, prob.T)
    N, p = settings.n, prob.p
    No = max(2, int(round(N * (p + 2) / (2 * p + 2))))
    e = {k: [] for k in ("S_saltelli", "S_jansen", "S_janon", "S_owen", "T_jansen")}
    for r in range(C.GRID_REPS):
        Am, Bm = A.draw(settings.design, N, p, A.stream_seed(settings.seed, 2, N, r))
        fA, fB, fAB = A.pick_freeze(prob.f, Am, Bm)
        est = A.estimators(fA, fB, fAB)
        for k in ("S_saltelli", "S_jansen", "S_janon", "T_jansen"):
            e[k].append(est[k])
        u3 = A.uniforms(A.stream_seed(settings.seed, 22, N, r), 3 * No * p).reshape(No, 3 * p)
        e["S_owen"].append(A.owen_estimator(prob.f, u3[:, :p], u3[:, p:2 * p], u3[:, 2 * p:]))
    rows = []
    for k, v in e.items():
        v = np.array(v)
        tr = prob.S if k.startswith("S") else prob.T
        hi, lo = (iSh, iSl) if k.startswith("S") else (iTh, iTl)
        col = v[:, lo][np.isfinite(v[:, lo])]
        rows.append({"est": k, "rmse_hi": _rmse(v[:, hi] - tr[hi]), "rmse_lo": _rmse(v[:, lo] - tr[lo]), "neg_lo": float(np.mean(col < 0)) if len(col) else float("nan"),
                     "runs": (p + 2) * N if k != "S_owen" else (2 * p + 2) * No, "N": N if k != "S_owen" else No, "truth_lo": float(tr[lo]), "truth_hi": float(tr[hi])})
    return {"rows": rows, "S_lo": float(prob.S[iSl]), "S_hi": float(prob.S[iSh]), "T_lo": float(prob.T[iTl]), "N": N, "N_owen": No, "p": p}


# --- Experiment 3: Rangfolge ------------------------------------------------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_experiment(settings: Settings, netz: Netz | None = None, table=None) -> dict:
    """Rangfolge nach dem Jansen-Gesamtindex ((p + 1) N Läufe): Anteil der Läufe, in denen alle wirksamen Eingaben (exaktes T über RANK_FRACTION des größten) vor allen unwirksamen liegen; mittlere Rangkorrelation."""
    prob = make_problem(settings, netz, table)
    hi = prob.T > C.RANK_FRACTION * prob.T.max()
    unique_top = int(np.sum(np.isclose(prob.T, prob.T.max(), rtol=1e-9))) == 1          # bei symmetrischen Eingaben gibt es keine eindeutig wichtigste
    top = int(np.argmax(prob.T))
    rows = []
    for N in C.RANK_NS:
        sep, top1, rho, defd = 0, 0, [], 0
        for r in range(C.GRID_REPS):
            Am, Bm = A.draw("mc", N, prob.p, A.stream_seed(settings.seed, 3, N, r))
            fA, _, fAB = A.pick_freeze(prob.f, Am, Bm, need_B=False)
            var = float(fA.var())
            if not var > 0:
                continue
            defd += 1
            t = np.mean((fA[:, None] - fAB) ** 2, axis=0) / (2 * var)
            if hi.any() and (~hi).any():
                sep += bool(t[hi].min() > t[~hi].max())
            top1 += int(np.argmax(t)) == top
            rho.append(_spearman(t, prob.T))
        rows.append({"N": N, "runs": (prob.p + 1) * N, "p_sep": sep / C.GRID_REPS if hi.any() and (~hi).any() else float("nan"), "p_top1": top1 / C.GRID_REPS if unique_top else float("nan"),
                     "spearman": float(np.nanmean(rho)) if rho and np.isfinite(rho).any() else float("nan"), "defined": defd / C.GRID_REPS})
    return {"rows": rows, "n_hi": int(hi.sum()), "n_lo": int((~hi).sum()), "t_hi": float(prob.T[hi].min()) if hi.any() else float("nan"), "t_lo": float(prob.T[~hi].max()) if (~hi).any() else float("nan"),
            "p": prob.p, "label": prob.label}


# --- Experiment 5: Fehlerbalken ------------------------------------------------------------------------------------------------------------------

def interval_experiment(settings: Settings, netz: Netz | None = None, table=None) -> dict:
    """Überdeckung des 95-%-Bootstrap-Intervalls (Jansen) für den größten und kleinsten Index, mit dem Nenner neu gerechnet gegen mit festem Nenner; N = min(settings.n, CI_N_MAX)."""
    prob = make_problem(settings, netz, table)
    iSh, iSl, iTh, iTl = ranks_of(prob.S, prob.T)
    N = min(settings.n, C.CI_N_MAX)
    targets = (("S", "größter Erstordnungsindex", iSh, prob.S), ("S", "kleinster Erstordnungsindex", iSl, prob.S), ("T", "größter Gesamtindex", iTh, prob.T), ("T", "kleinster Gesamtindex", iTl, prob.T))
    cov = {(t[1], fx): [] for t in targets for fx in (False, True)}
    hw = {(t[1], fx): [] for t in targets for fx in (False, True)}
    for r in range(C.CI_REPS):
        Am, Bm = A.draw(settings.design, N, prob.p, A.stream_seed(settings.seed, 5, N, r))
        fA, fB, fAB = A.pick_freeze(prob.f, Am, Bm)
        for kind, name, i, tr in targets:
            for fx in (False, True):
                rng = np.random.default_rng(A.stream_seed(settings.seed, 55, r, i, int(fx)) % (2**63))
                lo, hi = A.bootstrap_interval(fA, fB, fAB, i, rng, C.BOOT, fx)[kind]
                if np.isfinite(lo):
                    cov[(name, fx)].append(lo <= tr[i] <= hi)
                    hw[(name, fx)].append((hi - lo) / 2)
    rows = []
    for kind, name, i, tr in targets:
        for fx in (False, True):
            c, h = cov[(name, fx)], hw[(name, fx)]
            rows.append({"target": name, "kind": kind, "fixed": fx, "truth": float(tr[i]), "coverage": float(np.mean(c)) if c else float("nan"), "halfwidth": float(np.mean(h)) if h else float("nan"), "valid": len(c) / C.CI_REPS})
    return {"rows": rows, "N": N, "reps": C.CI_REPS, "boot": C.BOOT, "label": prob.label}


# --- Kosten (exakt) --------------------------------------------------------------------------------------------------------------------------

def cost_rows(N: int) -> list[dict]:
    """Modellläufe: Pick-Freeze (p + 2) N, nur Gesamtindizes (p + 1) N, Owen 2013 (2p + 2) N gegen die Aufzählung aller 2^m Kantenmengen; Gleichstand bei N = 2^m/(p + 2)."""
    rows = []
    for key in C.NETZE:
        m = named_netz(key).m
        rows.append({"netz": named_netz(key).name, "p": m, "enum": 1 << m, "saltelli": (m + 2) * N, "total": (m + 1) * N, "owen": (2 * m + 2) * N, "breakeven": (1 << m) / (m + 2)})
    return rows


def custom_result(settings: Settings, netz: Netz) -> dict:
    """Spielwiese: das eigene Netz mit dem gewählten Netz-Integranden (beim Begleiter: der Sprung)."""
    table = netz_table(netz)
    key = "sprung" if settings.integrand == "exp" else settings.integrand
    reason = available(dataclasses.replace(settings, integrand=key), netz)
    if reason:
        raise ValueError(reason)
    return {"analysis": analyse(settings, netz, table, key), "key": key}


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Kernfrage: Wie genau und zu welchem Preis misst man, welche Eingabe die Varianz treibt?"""
    N, last = a.ns[-1], len(a.ns) - 1
    s_lo, s_hi, t_lo = a.S[a.i_s_lo], a.S[a.i_s_hi], a.T[a.i_t_lo]
    best = min(("S_saltelli", "S_jansen", "S_janon"), key=lambda k: a.rmse_lo[k][last] if np.isfinite(a.rmse_lo[k][last]) else 9.0)
    rl, rt = a.rmse_lo[best][last], a.rmse_lo["T_jansen"][last]
    worst = int(np.argmin(a.defined))
    if a.defined[worst] < 0.9:
        return "warning", (f"{a.label}: Bei N = {de(a.ns[worst], 0)} haben nur {prozent(a.defined[worst])} der Läufe überhaupt eine Varianz (die Ausgabe ist zu selten verschieden); dort sind die Indizes undefiniert. "
                           f"Bei N = {de(N, 0)} ist der RMSE des größten Gesamtindex {idx(a.rmse_hi['T_jansen'][last])} bei einem Index von {idx(a.T[a.i_t_hi])}.")
    head = (f"{a.label}, {a.p} Eingaben, N = {de(N, 0)} Zeilen ({de(a.runs[last], 0)} Modellläufe): größter Erstordnungsindex {idx(s_hi)} (RMSE {idx(a.rmse_hi['S_jansen'][last])}), "
            f"kleinster {idx(s_lo)} mit dem RMSE {idx(rl)} ({best.replace('S_', '').capitalize()}, der beste der drei); Anteil negativer Schätzungen {prozent(a.neg_lo['S_jansen'][last])}. "
            f"Der kleinste Gesamtindex {idx(t_lo)} hat den RMSE {idx(rt)} ({prozent(rt / t_lo)} des Index), nie negativ. ")
    if rl > s_lo:
        return "warning", head + "Der kleine Erstordnungsindex wird vom Rauschen verschluckt: der Fehler ist größer als der Index."
    if rl > 0.3 * s_lo:
        return "info", head + "Der kleine Erstordnungsindex ist nur grob bestimmt (Fehler über 30 % des Index)."
    return "success", head + "Auch der kleinste Erstordnungsindex ist bestimmt."


def verdict_curve(a: Analysis) -> tuple[str, str]:
    ns = a.ns
    if len(ns) < 2:
        return "info", "Für ein Verhältnis der Fehler braucht die Kurve mindestens zwei Punktzahlen."
    first, last = 0, len(ns) - 1
    ratio = a.rmse_hi["S_jansen"][first] / a.rmse_hi["S_jansen"][last] if a.rmse_hi["S_jansen"][last] > 0 else float("nan")
    expected = (ns[last] / ns[first]) ** 0.5
    text = (f"Von N = {de(ns[first], 0)} auf {de(ns[last], 0)} fällt der RMSE des größten Erstordnungsindex (Jansen) auf das {dn(1 / ratio, 3)}-Fache; bei 1/√N wäre es das {de(1 / expected, 3)}-Fache. "
            f"Homma–Saltelli (nicht zentriert) hat bei N = {de(ns[last], 0)} den RMSE {idx(a.rmse_hi['S_homma'][last])} gegen {idx(a.rmse_hi['S_jansen'][last])} bei Jansen; "
            f"der Gesamtindex nach Homma–Saltelli {idx(a.rmse_hi['T_homma'][last])} gegen {idx(a.rmse_hi['T_jansen'][last])}.")
    return "info", text


def verdict_small(ex: dict) -> tuple[str, str]:
    r = {x["est"]: x for x in ex["rows"]}
    best_lo = min(("S_saltelli", "S_jansen", "S_janon", "S_owen"), key=lambda k: r[k]["rmse_lo"] if np.isfinite(r[k]["rmse_lo"]) else 9.0)
    best_hi = min(("S_saltelli", "S_jansen", "S_janon", "S_owen"), key=lambda k: r[k]["rmse_hi"] if np.isfinite(r[k]["rmse_hi"]) else 9.0)
    names = {"S_saltelli": "Saltelli", "S_jansen": "Jansen", "S_janon": "Janon", "S_owen": "Owen"}
    text = (f"Bei {de(r['S_jansen']['runs'], 0)} Modellläufen (Owen: {de(r['S_owen']['runs'], 0)} mit N = {de(ex['N_owen'], 0)}): kleinster Erstordnungsindex {idx(ex['S_lo'])}: "
            f"RMSE Saltelli {idx(r['S_saltelli']['rmse_lo'])}, Jansen {idx(r['S_jansen']['rmse_lo'])}, Janon {idx(r['S_janon']['rmse_lo'])}, Owen {idx(r['S_owen']['rmse_lo'])}; "
            f"größter {idx(ex['S_hi'])}: Saltelli {idx(r['S_saltelli']['rmse_hi'])}, Jansen {idx(r['S_jansen']['rmse_hi'])}, Janon {idx(r['S_janon']['rmse_hi'])}, Owen {idx(r['S_owen']['rmse_hi'])}. "
            f"Am besten für den kleinen Index: {names[best_lo]}; für den großen: {names[best_hi]}. Der kleinste Gesamtindex {idx(ex['T_lo'])} hat mit Jansen den RMSE {idx(r['T_jansen']['rmse_lo'])}.")
    return "info", text


def verdict_rank(ex: dict) -> tuple[str, str]:
    rows = ex["rows"]
    first_ok = next((x for x in rows if x["p_sep"] >= 0.95), None)
    parts = "; ".join(f"N = {de(x['N'], 0)} ({de(x['runs'], 0)} Läufe): {dn(x['p_sep'], 2)}" for x in rows)
    if np.isnan(rows[0]["p_sep"]):
        return "info", f"{ex['label']}: Alle Eingaben sind gleich wirksam (oder keine); eine Trennung in wirksam und unwirksam gibt es hier nicht."
    text = (f"{ex['label']}: {ex['n_hi']} wirksame Eingaben (T ab {idx(ex['t_hi'])}) gegen {ex['n_lo']} unwirksamere (T bis {idx(ex['t_lo'])}). Anteil der Läufe, in denen alle wirksamen vor allen anderen stehen: {parts}. ")
    text += (f"Eine sichere Trennung (95 %) gelingt ab N = {de(first_ok['N'], 0)} ({de(first_ok['runs'], 0)} Läufe)." if first_ok else "Eine sichere Trennung (95 %) gelingt in diesem Bereich nicht.")
    return "info", text


def verdict_sums(a: Analysis) -> tuple[str, str]:
    last = len(a.ns) - 1
    st, ss = a.sums["T"], a.sums["S"]
    text = (f"Σ der Gesamtindizes (= mittlere Dimension aus Stück 9) exakt {de(st[0], 2)}; geschätzt bei N = {de(a.ns[last], 0)}: Mittel {dn(st[1][last][0], 2)}, RMSE {dn(st[1][last][1], 3)} ({prozent(st[1][last][1] / st[0])}). "
            f"Σ der Erstordnungsindizes (= additiver Anteil, Stück 5) exakt {de(ss[0], 3)}; geschätzt: Mittel {dn(ss[1][last][0], 3)}, RMSE {dn(ss[1][last][1], 3)} ({prozent(ss[1][last][1] / ss[0])}).")
    kind = "info" if ss[1][last][1] < ss[0] else "warning"
    return kind, text


def verdict_intervals(ex: dict) -> tuple[str, str]:
    rows = {(r["target"], r["fixed"]): r for r in ex["rows"]}
    big, small = rows[("größter Erstordnungsindex", False)], rows[("kleinster Erstordnungsindex", False)]
    bigf = rows[("größter Erstordnungsindex", True)]
    text = (f"{ex['label']}, N = {de(ex['N'], 0)}, {ex['reps']} Wiederholungen, {ex['boot']} Bootstrap-Ziehungen. Größter Erstordnungsindex: Überdeckung {de(big['coverage'], 3)} (Halbbreite {idx(big['halfwidth'])}) mit dem Nenner neu gerechnet, "
            f"{de(bigf['coverage'], 3)} (Halbbreite {idx(bigf['halfwidth'])}) mit festem Nenner. Kleinster Erstordnungsindex {idx(small['truth'])}: Überdeckung {de(small['coverage'], 3)}, Halbbreite {idx(small['halfwidth'])}. ")
    tl = rows[("kleinster Gesamtindex", False)]
    text += f"Kleinster Gesamtindex {idx(tl['truth'])}: Überdeckung {de(tl['coverage'], 3)}, Halbbreite {idx(tl['halfwidth'])}."
    return "info", text


def verdict_cost(rows: list[dict], N: int) -> tuple[str, str]:
    parts = "; ".join(f"{r['netz']} (p = {r['p']}): Aufzählung {de(r['enum'], 0)} Läufe, Pick-Freeze {de(r['saltelli'], 0)}, Gleichstand bei N = {de(r['breakeven'], 0)}" for r in rows)
    return "info", f"Bei N = {de(N, 0)}: {parts}. Bei wenigen diskreten Eingaben mit bekannter Verteilung ist die Aufzählung billiger; Pick-Freeze lohnt sich erst, wenn 2^p die Läufe weit übersteigt."
