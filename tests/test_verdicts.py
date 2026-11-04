"""Verdikte (reine Funktionen in sbi_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import sbi_evaluation as ev

S = ev.Settings()


def _a(**kw) -> ev.Analysis:
    ests = ("S_saltelli", "S_jansen", "S_janon", "S_homma", "T_jansen", "T_homma")
    hi = {k: (0.1, 0.05, 0.03) for k in ests}
    lo = {k: (0.05, 0.03, 0.01) for k in ests}
    hi["S_jansen"], hi["S_homma"], hi["T_jansen"], hi["T_homma"] = (0.10, 0.06, 0.03), (0.50, 0.30, 0.125), (0.1, 0.06, 0.03), (0.2, 0.1, 0.07)
    lo["S_jansen"], lo["S_janon"], lo["S_saltelli"], lo["T_jansen"] = (0.05, 0.03, 0.0125), (0.04, 0.02, 0.0099), (0.06, 0.04, 0.0162), (0.04, 0.02, 0.0085)
    basis = ev.Analysis(
        label="Sprung: Gitter 3×3 hängt zusammen (p = 0,90)", p=12, S=(0.0283, 0.0024), T=(0.2254, 0.0659), var=0.05, i_s_hi=0, i_s_lo=1, i_t_hi=0, i_t_lo=1, ns=(100, 1000, 10000),
        runs=(1400, 14000, 140000), defined=(1.0, 1.0, 1.0), rmse_hi=hi, rmse_lo=lo, neg_lo={"S_saltelli": (0.2, 0.3, 0.31), "S_jansen": (0.59, 0.5, 0.41), "S_janon": (0.7, 0.46, 0.39), "S_homma": (0.5, 0.5, 0.49)},
        sums={"T": (2.067, ((2.1, 0.63), (2.07, 0.18), (2.07, 0.058))), "S": (0.2365, ((0.3, 1.2), (0.25, 0.33), (0.24, 0.098)))}, design="mc", reps=100)
    return dataclasses.replace(basis, **kw)


def test_kleiner_index_verschluckt_mit_zahlen():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "warning"
    for frag in ("Gitter 3×3 hängt zusammen (p = 0,90), 12 Eingaben, N = 10.000 Zeilen (140.000 Modellläufe)", "größter Erstordnungsindex 0,0283", "kleinster 0,0024 mit dem RMSE 0,0099 (Janon, der beste der drei)",
                 "Anteil negativer Schätzungen 41 %", "kleinste Gesamtindex 0,0659 hat den RMSE 0,0085 (13 % des Index)", "vom Rauschen verschluckt"):
        assert frag in text, frag


def test_schwellen_verschluckt_grob_bestimmt():
    def lo(v):
        d = {k: (v, v, v) for k in ("S_saltelli", "S_jansen", "S_janon", "S_homma", "T_jansen", "T_homma")}
        return _a(rmse_lo=d)
    assert ev.verdict_main(lo(0.0025), S)[0] == "warning"                       # RMSE über dem Index 0,0024
    kind, text = ev.verdict_main(lo(0.0010), S)
    assert kind == "info" and "grob bestimmt" in text                           # über 30 % des Index
    kind, text = ev.verdict_main(lo(0.0005), S)
    assert kind == "success" and "bestimmt" in text


def test_seltene_ausgabe_meldet_den_anteil_der_laeufe_mit_varianz():
    kind, text = ev.verdict_main(_a(defined=(0.08, 0.5, 0.62)), S)
    assert kind == "warning" and "Bei N = 100 haben nur 8 % der Läufe überhaupt eine Varianz" in text and "Bei N = 10.000 ist der RMSE des größten Gesamtindex 0,0300 bei einem Index von 0,225" in text


def test_kurven_verdikt_nennt_das_verhaeltnis_und_die_nicht_zentrierten_schaetzer():
    kind, text = ev.verdict_curve(_a())
    assert kind == "info" and "auf das 0,300-Fache" in text and "bei 1/√N wäre es das 0,100-Fache" in text
    assert "Homma–Saltelli (nicht zentriert) hat bei N = 10.000 den RMSE 0,125 gegen 0,0300 bei Jansen" in text and "0,0700 gegen 0,0300" in text
    assert "mindestens zwei" in ev.verdict_curve(_a(ns=(100,), runs=(1400,)))[1]


def _small(**kw):
    rows = [{"est": "S_saltelli", "rmse_lo": 0.0162, "rmse_hi": 0.0278, "neg_lo": 0.31, "runs": 140000, "N": 10000}, {"est": "S_jansen", "rmse_lo": 0.0119, "rmse_hi": 0.0172, "neg_lo": 0.41, "runs": 140000, "N": 10000},
            {"est": "S_janon", "rmse_lo": 0.0099, "rmse_hi": 0.0120, "neg_lo": 0.39, "runs": 140000, "N": 10000}, {"est": "S_owen", "rmse_lo": 0.0031, "rmse_hi": 0.0103, "neg_lo": 0.2, "runs": 139995, "N": 5385},
            {"est": "T_jansen", "rmse_lo": 0.0085, "rmse_hi": 0.0149, "neg_lo": 0.0, "runs": 140000, "N": 10000}]
    return {"rows": rows, "S_lo": 0.0024, "S_hi": 0.0283, "T_lo": 0.0659, "N": 10000, "N_owen": 5385, "p": 12, **kw}


def test_kleine_index_verdikt():
    kind, text = ev.verdict_small(_small())
    assert kind == "info"
    for frag in ("Bei 140.000 Modellläufen (Owen: 139.995 mit N = 5.385)", "kleinster Erstordnungsindex 0,0024: RMSE Saltelli 0,0162, Jansen 0,0119, Janon 0,0099, Owen 0,0031", "größter 0,0283: Saltelli 0,0278, Jansen 0,0172, Janon 0,0120, Owen 0,0103",
                 "Am besten für den kleinen Index: Owen; für den großen: Owen", "Der kleinste Gesamtindex 0,0659 hat mit Jansen den RMSE 0,0085"):
        assert frag in text, frag
    ex = _small()
    ex["rows"][3]["rmse_hi"] = 0.09
    assert "für den großen: Janon" in ev.verdict_small(ex)[1]


def test_rang_verdikt():
    rows = [{"N": 30, "runs": 390, "p_sep": 0.0, "p_top1": float("nan"), "spearman": 0.22, "defined": 1.0}, {"N": 300, "runs": 3900, "p_sep": 0.43, "p_top1": float("nan"), "spearman": 0.63, "defined": 1.0},
            {"N": 1000, "runs": 13000, "p_sep": 0.97, "p_top1": float("nan"), "spearman": 0.67, "defined": 1.0}]
    kind, text = ev.verdict_rank({"rows": rows, "n_hi": 8, "n_lo": 4, "t_hi": 0.2254, "t_lo": 0.0659, "p": 12, "label": "Gitter 3×3"})
    assert kind == "info"
    for frag in ("8 wirksame Eingaben (T ab 0,225) gegen 4 unwirksamere (T bis 0,0659)", "N = 300 (3.900 Läufe): 0,43", "Eine sichere Trennung (95 %) gelingt ab N = 1.000 (13.000 Läufe)."):
        assert frag in text, frag
    rows[2]["p_sep"] = 0.5
    assert "gelingt in diesem Bereich nicht" in ev.verdict_rank({"rows": rows, "n_hi": 8, "n_lo": 4, "t_hi": 0.2, "t_lo": 0.1, "p": 12, "label": "x"})[1]
    rows[0]["p_sep"] = float("nan")
    assert "gleich wirksam" in ev.verdict_rank({"rows": rows, "n_hi": 12, "n_lo": 0, "t_hi": 0.2, "t_lo": float("nan"), "p": 12, "label": "x"})[1]


def test_summen_verdikt():
    kind, text = ev.verdict_sums(_a())
    assert kind == "info"
    for frag in ("exakt 2,07; geschätzt bei N = 10.000: Mittel 2,07, RMSE 0,058 (3 %)", "exakt 0,236; geschätzt: Mittel 0,240, RMSE 0,098 (41 %)"):
        assert frag in text, frag
    bad = _a(sums={"T": (2.067, ((2.1, 0.6),) * 3), "S": (0.2365, ((0.3, 1.2),) * 3)})
    assert ev.verdict_sums(bad)[0] == "warning"


def test_intervall_verdikt():
    rows = []
    for target, truth, cov, hw, covf, hwf in (("größter Erstordnungsindex", 0.0283, 0.953, 0.1004, 1.0, 0.182), ("kleinster Erstordnungsindex", 0.0024, 0.88, 0.0706, 1.0, 0.18),
                                               ("größter Gesamtindex", 0.2254, 0.913, 0.0924, 0.947, 0.092), ("kleinster Gesamtindex", 0.0659, 0.953, 0.0487, 0.947, 0.049)):
        rows.append({"target": target, "kind": target[0], "fixed": False, "truth": truth, "coverage": cov, "halfwidth": hw, "valid": 1.0})
        rows.append({"target": target, "kind": target[0], "fixed": True, "truth": truth, "coverage": covf, "halfwidth": hwf, "valid": 1.0})
    kind, text = ev.verdict_intervals({"rows": rows, "N": 1000, "reps": 100, "boot": 100, "label": "Sprung: Gitter 3×3"})
    assert kind == "info"
    for frag in ("N = 1.000, 100 Wiederholungen, 100 Bootstrap-Ziehungen", "Überdeckung 0,953 (Halbbreite 0,100) mit dem Nenner neu gerechnet, 1,000 (Halbbreite 0,182) mit festem Nenner",
                 "Kleinster Erstordnungsindex 0,0024: Überdeckung 0,880, Halbbreite 0,0706", "Kleinster Gesamtindex 0,0659: Überdeckung 0,953, Halbbreite 0,0487"):
        assert frag in text, frag


def test_kosten_verdikt():
    kind, text = ev.verdict_cost(ev.cost_rows(3000), 3000)
    assert kind == "info" and "Bei N = 3.000" in text and "Hantel (p = 7): Aufzählung 128 Läufe, Pick-Freeze 27.000, Gleichstand bei N = 14" in text and "Gitter 3×4 (p = 17): Aufzählung 131.072 Läufe, Pick-Freeze 57.000, Gleichstand bei N = 6.899" in text
