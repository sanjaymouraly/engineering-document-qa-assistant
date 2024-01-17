"""Tests for the deterministic Engineering Value Verification Engine."""
from __future__ import annotations

import pytest
from decimal import Decimal

from engrag.verify.numbers import (compare_values, equal_within,
                                   extract_values)


def one(text: str, idx: int = 0):
    vs = extract_values(text)
    assert vs, f"no value extracted from {text!r}"
    return vs[idx]


# ------------------------------------------------------------------ parsing
def test_parses_core_units():
    for text, raw, canon in [
        ("Rated voltage 11 kV", "11 kV", "voltage"),
        ("current of 630 A flows", "630 A", "current"),
        ("apparent power 1600 kVA", "1600 kVA", "apparent_power"),
        ("loss is 12 kW", "12 kW", "active_power"),
        ("impedance 12.5 %", "12.5 %", "percent"),
        ("cross-section 300 mm²", "300 mm²", "area"),
        ("ambient at 40 °C", "40 °C", "temperature_c"),
        ("frequency 50 Hz", "50 Hz", "frequency"),
        ("short circuit 25 kA for 3 s", "25 kA", "current"),
        ("duration 3 s", "3 s", "time_s"),
        ("resistance 0.25 Ω", "0.25 Ω", "resistance"),
        ("vibration 7.1 mm/s RMS", "7.1 mm/s", "velocity"),
        ("length 2500 mm", "2500 mm", "length"),
        ("torque 45 Nm", "45 Nm", "torque"),
        ("speed 1485 rpm", "1485 rpm", "speed"),
    ]:
        got = one(text)
        assert got.raw == raw, f"{text}: got {got.raw}"
        assert got.canonical == canon, f"{text}: got {got.canonical}"


def test_fractional_and_thousand_separators():
    assert one("rated 1,600 kVA").value == 1600
    assert one("rated 11500 V").value == 11500


def test_bare_numbers_are_not_engineering_values():
    # a number only counts when it carries (or shares) a recognised unit —
    # otherwise every clause number would look like a measurement
    assert not any(v.canonical == "time_s" and v.value == 3
                   for v in extract_values("Clause 7.1 applies to item 3"))
    assert not any(v.canonical == "length" for v in extract_values("see 3.1.2"))


def test_slash_list_expansion():
    vs = extract_values("Rated voltages 66/33/11 kV respectively")
    kvs = sorted(float(v.value) for v in vs if v.canonical == "voltage")
    assert kvs == [11.0, 33.0, 66.0]


def test_unit_is_not_split_from_number_by_text():
    # "7.1 mm/s RMS" must parse as velocity, not "7.1 mm" length
    v = one("vibration limit 7.1 mm/s RMS measured")
    assert v.canonical == "velocity"
    assert v.raw == "7.1 mm/s"


# ---------------------------------------------------------------- qualifiers
def test_qualifier_before_number_wins():
    assert one("The rated current is 515 A").qualifier == "rated"
    assert one("Maximum system voltage 145.5 kV").qualifier == "maximum"
    assert one("Typical noise 72 dB(A), guaranteed 75 dB(A)").qualifier == "typical"


def test_table_row_qualifier_and_no_leak_across_rows():
    text = ("Maximum system voltage | Um | 145.5 kV | HV side\n"
            "Rated voltage | U1r | 33 kV | HV side")
    vs = extract_values(text)
    by = {float(v.value): v for v in vs if v.canonical == "voltage"}
    assert by[145.5].qualifier == "maximum"
    assert by[33.0].qualifier == "rated"


def test_qualifier_does_not_leak_from_neighbouring_row():
    text = ("Rated frequency | f | 50 Hz | —\n"
            "Impedance | Uk | 12.5 % | measured at factory")
    vs = extract_values(text)
    pct = [v for v in vs if v.canonical == "percent"]
    assert pct and pct[0].qualifier is None or pct[0].qualifier in ("measured", "guaranteed")


# ------------------------------------------------------------------ ranges
def test_range_capture():
    vs = extract_values("Ambient temperature from -10 °C to +50 °C")
    temps = [v for v in vs if v.canonical.startswith("temperature")]
    assert temps, "temperatures parsed"
    assert any(v.range_tuple == (Decimal(-10), Decimal(50)) for v in temps), \
        "range attached to the bounds"
    v2 = extract_values("impedance range 12 to 12.5 %")
    assert any(v.range_tuple for v in v2)


def test_tolerance():
    v2 = one("Rated current 630 A ±10 %")
    assert v2.tolerance_pct == pytest.approx(10.0)
    v3 = one("Duration 3 s ± 0.2 s")
    assert v3.tolerance_pct is None or v3.tolerance_pct > 0


# ---------------------------------------------------------------- comparison
def test_exact_match_and_unit_conversion():
    a = one("rated voltage is 11 kV")
    b = one("rated voltage of 11,000 V")
    c = compare_values(a, b, 1e-9)
    assert c.status == "unit_verified"
    d = one("rated voltage is 12 kV")
    assert compare_values(d, b, 1e-9).status == "mismatch"


def test_area_mm2_cm2_conversion():
    a = one("cross-section 302 mm²")
    b = one("cross-section 3.02 cm²")
    assert compare_values(a, b, 0.001).status in ("unit_verified",
                                                  "verified_within_tolerance")


def test_time_conversions():
    assert compare_values(one("delay 2 s"), one("delay 2 s"), 1e-9).status == "verified"
    assert compare_values(one("duration 60 s"), one("duration 1 min"),
                          1e-9).status == "unit_verified"
    # minutes must not parse as milli-inches
    assert one("cycle 45 min").canonical == "time_min"


def test_power_families_not_interconvertible():
    kw = one("output 1600 kW")
    kva = one("rating 1600 kVA")
    res = compare_values(kw, kva, 1e-9)
    assert res.status in ("not_comparable", "ambiguous", "mismatch")
    assert res.status != "verified"


def test_scope_conflict_equal_magnitude():
    rated = one("The rated system voltage is 145.5 kV")
    maximum = one("Maximum system voltage 145.5 kV")
    res = compare_values(rated, maximum, 1e-9)
    assert res.status == "mismatch"
    assert "scope" in res.explanation.lower() or "rated" in res.explanation


def test_tolerance_window_verifies():
    spec = one("impedance 12.5 % at rated current")
    measured = one("impedance 12.56 %")
    res = compare_values(measured, spec, 0.01)
    assert res.status == "verified_within_tolerance"
    assert compare_values(measured, spec, 1e-9).status == "mismatch"


def test_incomparable_dimensions():
    a = one("length 1000 mm")
    b = one("current 1000 A")
    assert compare_values(a, b, 1e-9).status in ("not_comparable", "mismatch")


def test_display_is_human_readable():
    v = one("rated 11000 V")
    disp = v.display() if hasattr(v, "display") else v.raw
    assert "11" in disp and "V" in disp


def test_equal_within_decimal_exactness():
    # 0.1 + 0.2 float artifacts must not break exact comparison
    ok, diff = equal_within(0.30000000000000004, 0.3, 0.0)
    assert not ok          # exact mode is strict by design
    ok, diff = equal_within(0.30000000000000004, 0.3, 1e-6)
    assert ok


def test_percent_is_dimensionless():
    p = one("efficiency 98 %")
    a = one("5 %")
    assert compare_values(p, a, 1e-9).status == "mismatch"
    assert p.canonical == "percent"


def test_variant_detection():
    v = one("DEMO-TR-01 rated voltage 33 kV")
    assert v.variant and "TR-01" in v.variant.upper()
