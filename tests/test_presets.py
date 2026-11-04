"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import sbi_constants as C
import sbi_evaluation as ev
import sbi_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["integrand"] in C.INTEGRANDS and preset["netz"] in C.NETZE and preset["weight"] in C.WEIGHTS and preset["design"] in C.DESIGNS, name
        assert preset["p"] in C.P_CHOICES and preset["d"] in C.D_CHOICES and preset["n"] in C.N_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard_und_ist_rechenbar():
    standard = C.PRESETS["Standardfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        if name != "Standardfall (Voreinstellung)":
            assert preset != standard, name
        assert ev.available(ev.Settings(**preset)) is None, name


def test_standardfall_zeigt_den_grossen_und_verschluckt_den_kleinen_index():
    s = ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"])
    a = ev.analyse(s)
    assert ev.verdict_main(a, s)[0] == "warning" and a.rmse_hi["S_jansen"][-1] < 0.05 and a.rmse_lo["S_jansen"][-1] > a.S[a.i_s_lo]


def test_preset_hantel_hat_eine_dominierende_bruecke():
    s = ev.Settings(**C.PRESETS["Hantel: eine Brücke"])
    prob = ev.make_problem(s)
    assert prob.S[6] > 0.6 and prob.S[:6].max() < 0.02 and ev.analyse(s).p == 7


def test_preset_kleine_indizes_am_begleiter_hat_indizes_von_0_63_bis_0_01():
    s = ev.Settings(**C.PRESETS["Kleine Indizes am glatten Begleiter"])
    prob = ev.make_problem(s)
    assert prob.S.max() == pytest.approx(0.63, abs=0.005) and prob.S.min() == pytest.approx(0.01, abs=0.002)


def test_preset_seltene_ausgabe_ist_zu_selten():
    s = ev.Settings(**C.PRESETS["Seltene Ausgabe"])
    a = ev.analyse(s)
    kind, text = ev.verdict_main(a, s)
    assert kind == "warning" and a.defined[-1] > 0.95 and a.rmse_hi["T_jansen"][-1] > 0.5 * a.T[a.i_t_hi]
    assert a.defined[0] < 0.3


def test_preset_quasi_zufallszahlen_ist_um_ein_vielfaches_genauer():
    s = ev.Settings(**C.PRESETS["Quasi-Zufallszahlen im Entwurf"])
    mc = ev.analyse(ev.Settings(**{**C.PRESETS["Quasi-Zufallszahlen im Entwurf"], "design": "mc"}))
    ow = ev.analyse(s)
    assert s.design == "owen" and ow.rmse_hi["S_saltelli"][-1] < mc.rmse_hi["S_saltelli"][-1] / 5
