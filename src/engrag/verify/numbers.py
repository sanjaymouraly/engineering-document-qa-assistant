"""Engineering value extraction and unit normalization (Pint-backed).

Parsing rules are deliberately conservative:

* A value counts only when a *known engineering unit token* (optionally with
  an SI prefix) follows the number — so years, page/clause numbers and bare
  counts never masquerade as engineering quantities.
* At every position the longest candidate unit token wins (``rpm`` before
  ``m``, ``kPa`` before… nothing else matches).
* Dimensions come from a private Pint registry.  Categories that share the
  SI dimension of power but differ physically (W vs VA vs var) are kept in
  separate canonical categories and are never silently converted — the
  verifier reports ``ambiguous`` with an explanation instead.
* Qualifiers around the value (rated/max/typical/min, ±tolerance, test
  temperature, phase note, model variant, line label) are captured: equal
  magnitudes referring to different engineering scopes must not be declared
  "verified" without stating the caveat.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext

import pint

getcontext().prec = 30

_registry = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)
_registry.define("square_millimetre = 1e-6 * meter ** 2 = mm_2")
_registry.define("square_centimetre = 1e-4 * meter ** 2 = cm_2")
_registry.define("meter_per_second = meter / second = m_s")
_registry.define("decibel = 0.1 = dB")   # power-ratio unit (exact decimal scale)

_Q = _registry.Quantity


@dataclass(frozen=True)
class UnitDef:
    key: str            # canonical category id
    token: str          # base unit token (regex alternation allowed)
    cs: bool            # case-sensitive token match
    prefixes: str       # allowed single-char SI prefixes ('' = none)
    pint_expr: str      # pint expression for the bare token
    label: str          # display category, e.g. "voltage"


# Longer compound tokens first; every pattern consumes its own leading space,
# longest overall match at a position wins (see _match_unit).
_UNIT_DEFS: list[UnitDef] = [
    UnitDef("area", r"(?:mm|cm)(?:²|2)", True, "", "mm_2", "cross-sectional area"),
    UnitDef("temperature_c", r"(?:°C|degC)", True, "", "degC", "temperature"),
    UnitDef("speed", "rpm", False, "", "rpm", "rotational speed"),
    UnitDef("resistance", r"ohm|Ω", False, "kM", "ohm", "resistance"),
    UnitDef("frequency", "hz", False, "km", "Hz", "frequency"),
    UnitDef("torque", "Nm", True, "k", "N*m", "torque"),
    UnitDef("pressure", "Pa", False, "kM", "Pa", "pressure"),
    UnitDef("energy", "Wh", False, "kM", "Wh", "energy"),
    UnitDef("apparent_power", "VA", True, "kM", "VA", "apparent power"),
    UnitDef("reactive_power", "VAR", False, "kM", "var", "reactive power"),
    UnitDef("active_power", "W", True, "kmM", "W", "active power"),
    UnitDef("voltage", "V", True, "kmM", "V", "voltage"),
    UnitDef("current", "A", True, "km", "A", "current"),
    UnitDef("temperature_k", "K", True, "", "K", "temperature"),
    UnitDef("length", "m", True, "kcm", "m", "length"),
    UnitDef("mass", "g", False, "k", "g", "mass"),
    UnitDef("time_s", "s", True, "m", "s", "duration"),
    UnitDef("time_min", "min", True, "", "min", "duration"),
    UnitDef("time_h", "h", True, "", "h", "duration"),
    UnitDef("percent", r"%|percent", False, "", "percent", "ratio"),
    UnitDef("velocity", r"(?:mm|c?m)/s", True, "", "m_s", "velocity"),
    UnitDef("level", r"(?:dB\(A\)|dB)", False, "", "dB",
            "sound level (dB)"),
]
_UNITS_BY_KEY = {u.key: u for u in _UNIT_DEFS}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SNUM = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
_NUM_RE = re.compile(r"-?" + _NUM)
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_CLAUSE_SPLIT = re.compile(r"[,;]")
# "66/33/11 kV" — slash-separated value list sharing one trailing unit
_LISTY_RE = re.compile(r"(?:\s*/\s*\d[\d,.]*)+(?=\s*(?:[A-Za-z°²%]|deg))")
# fallback range scan: "22.5 – 24.3 kV", "20 to 40 °C", "15/16 kV"
_RANGE_SCAN_RE = re.compile(
    # "12 to 12.5 %", "-10 °C to +50 °C", "22.5 kV – 24.3 kV"
    r"([+-]?\d[\d,.]*)\s*"
    r"(?:°\s*C|deg\s*C|%|mm\s*[²2]|k?VA|k?VAr|k?var|k?V|k?A|k?W|kHz|Hz|rpm|min|s|dB)?"
    r"\s*(?:(to|and|[-–—])\s*)([+-]?\d[\d,.]*)\s*"
    r"(°\s*C|deg\s*C|%|mm\s*[²2]|k?VA|k?VAr|k?var|k?V|k?A|k?W|kHz|Hz|rpm|min|s|dB)"
    r"(?![A-Za-z0-9])", re.I)


_EXCLUDE_BEFORE = re.compile(
    r"(?:year|page|clause|figure|fig|table|revision|rev|issue|version|ver|ref|"
    r"standard|serial|doc|item|no)\s*[#:.]?\s*$", re.I)

_QUALIFIER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("rated", re.compile(r"\b(rated|nameplate)\b", re.I)),
    ("maximum", re.compile(r"\b(max(?:\.|imum)?|highest|upper limit)\b", re.I)),
    ("minimum", re.compile(r"\b(min(?:\.|imum)?|lowest)\b", re.I)),
    ("typical", re.compile(r"\b(typ(?:ical)?\.?|approx(?:\.|imately)?"
                           r"|nominal|usual(?:ly)?|normal(?:ly)?)\b", re.I)),
    ("guaranteed", re.compile(r"\bguaranteed\b", re.I)),
    ("measured", re.compile(r"\bmeasured\b", re.I)),
    ("design", re.compile(r"\bdesign\b", re.I)),
]
_TOL_RE = re.compile(r"[±]\s*(" + _SNUM + r")\s*(%|°C)?")
_TOL_AFTER_RE = re.compile(r"^\s*(?:of\s+)?[±]\s*(" + _SNUM + r")\s*(%|°C)?")
_TOL_BEFORE_RE = re.compile(r"[±]\s*(" + _SNUM + r")\s*(?:%|°C)?\s*(?:[x×]\s*"
                            + _NUM + r")?\s*$")
_VARIANT_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,6}[-/ ]?\d{1,5}[A-Z0-9/]*\b")
# In pipe-table rows a same-row current rating ("1250 A") identifies the row
# (section size) a value belongs to — used to catch cross-column misattribution.
_ROW_TAG_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*A\b")

_RANGE_ALT = (r"mm/s|°\s*C|deg\s*C|mm\s*[²2]|cm\s*[²2]|k?VAR|k?VA|kV|kA|kHz|MHz|Hz|dB(?:\(A\))?|"
              r"mm2|rpm|ohm|kΩ|%|min|k?W|V|A|s|h")
# "20 to 40 °C", "22.5 kV to 24.3 kV", "-10 % to +5 %"
_RANGE_RE = re.compile(
    r"((?:" + _SNUM + r")\s*(?:" + _RANGE_ALT + r")?)\s*(?:to|[-–—]|and)\s*"
    r"((?:" + _SNUM + r")\s*(?:" + _RANGE_ALT + r"))", re.I)

# Exact label → pint expression.  Necessary because pint's defaults differ
# from engineering conventions: bare 'min' is milli-inch, 'Nm'/'MΩ' are not
# aliases, '°C' needs the degC name.
_LABEL_FIX = {
    "°C": "degC", "degC": "degC", "%": "percent", "percent": "percent",
    "Ω": "ohm", "mΩ": "mohm", "kΩ": "kohm", "MΩ": "Mohm",
    "ohm": "ohm", "Ohm": "ohm", "ohms": "ohm", "kOhm": "kohm", "MOhm": "Mohm",
    "mm²": "mm_2", "cm²": "cm_2", "mm2": "mm_2", "cm2": "cm_2",
    "min": "minute", "h": "hour",
    "Nm": "N*m", "kNm": "kN*m",
    "var": "var", "VAR": "var", "kvar": "kvar", "kVAR": "kvar", "Mvar": "Mvar",
    "MVAR": "Mvar",
}


def _pint_expr_for_label(label: str) -> str:
    lab = re.sub(r"\s+", "", label.strip())
    if lab in _LABEL_FIX:
        return _LABEL_FIX[lab]
    return lab


def _unit_patterns() -> list[tuple[UnitDef, re.Pattern]]:
    out = []
    for u in _UNIT_DEFS:
        prefix = f"(?:[{re.escape(u.prefixes)}])?" if u.prefixes else ""
        flags = 0 if u.cs else re.I
        rx = re.compile(r"[ \t\u00a0]?" + prefix + f"(?:{u.token})"
                        r"(?![A-Za-z0-9²°³⁴⁵⁶⁰¹])", flags)
        out.append((u, rx))
    return out


_UNIT_PATTERNS = _unit_patterns()

_EXACT_PREFIX_SCALES = {
    "V": Decimal(1), "kV": Decimal(1000), "mV": Decimal("0.001"), "MV": Decimal(1_000_000),
    "A": Decimal(1), "kA": Decimal(1000), "mA": Decimal("0.001"),
    "W": Decimal(1), "kW": Decimal(1000), "MW": Decimal(1_000_000),
    "VA": Decimal(1), "kVA": Decimal(1000), "MVA": Decimal(1_000_000),
    "var": Decimal(1), "kvar": Decimal(1000), "VAR": Decimal(1), "kVAR": Decimal(1000),
    "ohm": Decimal(1), "Ohm": Decimal(1), "Ω": Decimal(1), "kΩ": Decimal(1000),
    "MOhm": Decimal(1_000_000), "MΩ": Decimal(1_000_000), "mΩ": Decimal("0.001"),
    "Hz": Decimal(1), "kHz": Decimal(1000), "MHz": Decimal(1_000_000),
    "mm": Decimal("0.001"), "cm": Decimal("0.01"), "m": Decimal(1), "km": Decimal(1000),
    "g": Decimal("0.001"), "kg": Decimal(1),
    "s": Decimal(1), "ms": Decimal("0.001"), "min": Decimal(60), "h": Decimal(3600),
    "%": Decimal("0.01"), "percent": Decimal("0.01"),
    "rpm": Decimal(1) / Decimal(60), "Nm": Decimal(1), "kNm": Decimal(1000),
    "mm/s": Decimal("0.001"), "m/s": Decimal(1), "cm/s": Decimal("0.01"),
    "dB": Decimal("0.1"), "dB(A)": Decimal("0.1"),
    "Wh": Decimal(3600), "kWh": Decimal(3_600_000), "MWh": Decimal(3_600_000_000),
    "Pa": Decimal(1), "kPa": Decimal(1000), "MPa": Decimal(1_000_000),
    "K": Decimal(1),
}
_SCALE_CACHE: dict[str, Decimal] = {}


def _scale_to_base(unit_label: str) -> Decimal:
    cached = _SCALE_CACHE.get(unit_label)
    if cached is not None:
        return cached
    lab = unit_label.strip()
    lab_ns = re.sub(r"\s+", "", lab)
    if lab_ns in ("°C", "degC"):
        # offset unit: base conversion is handled by compare_values via pint;
        # the raw magnitude in °C is kept for display purposes
        _SCALE_CACHE[unit_label] = Decimal(1)
        return Decimal(1)
    exact = _EXACT_PREFIX_SCALES.get(lab) or _EXACT_PREFIX_SCALES.get(lab_ns)
    if exact is None:
        try:
            q = _Q(1.0, _pint_expr_for_label(lab))
            exact = Decimal(str(q.to_base_units().magnitude))
        except Exception:
            exact = Decimal(1)
    _SCALE_CACHE[unit_label] = exact
    return exact


@dataclass
class EngineeringValue:
    raw: str
    value: Decimal
    unit_label: str          # display unit as written, e.g. "kV"
    canonical: str           # UnitDef key
    dimension: str           # pint dimensionality string
    magnitude_si: float      # base-SI magnitude (float, informational)
    qualifier: str | None = None
    tolerance_pct: float | None = None
    range_tuple: tuple[Decimal, Decimal] | None = None
    temperature_c: float | None = None
    phase_note: str | None = None
    variant: str | None = None
    sentence: str = ""
    char_start: int = 0
    char_end: int = 0
    row_tags: tuple[str, ...] = ()   # same-row identifiers (table rows)

    @property
    def unit_def(self) -> UnitDef:
        return _UNITS_BY_KEY[self.canonical]

    @property
    def category_label(self) -> str:
        return self.unit_def.label

    def display(self) -> str:
        v = self.value
        s = str(int(v)) if v == v.to_integral_value() else f"{v:.6g}"
        return f"{s} {self.unit_label}"

    def base_magnitude(self) -> Decimal:
        return self.value * _scale_to_base(re.sub(r"\s+", "", self.unit_label))

    def context_summary(self) -> dict:
        return {"qualifier": self.qualifier, "tolerance_pct": self.tolerance_pct,
                "range": (str(self.range_tuple[0]), str(self.range_tuple[1]))
                if self.range_tuple else None,
                "temperature_c": self.temperature_c, "phase_note": self.phase_note,
                "variant": self.variant}


def _sentence_of(text: str, pos: int) -> tuple[str, int]:
    starts = [0] + [m.start() + 1 for m in _SENT_SPLIT.finditer(text)]
    start = max(s for s in starts if s <= pos)
    nxt = [m.start() + 1 for m in _SENT_SPLIT.finditer(text) if m.start() >= pos]
    end = nxt[0] if nxt else len(text)
    return text[start:end], start


def _line_of(text: str, pos: int) -> tuple[str, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start:len(text) if end < 0 else end], start


def _match_unit(text: str, pos: int) -> tuple[UnitDef, str, int] | None:
    """Longest unit token starting at/after ``pos`` (single space allowed)."""
    candidates: list[tuple[int, int, UnitDef, int]] = []
    for u, rx in _UNIT_PATTERNS:
        m = rx.match(text, pos)
        if m:
            candidates.append((m.end() - pos, len(m.group(0).strip()), u, m.end()))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, u, end = candidates[0]
    token = re.sub(r"\s+", " ", text[pos:end]).strip()
    return u, token, end


def extract_values(text: str, *, max_values: int = 120) -> list[EngineeringValue]:
    """Find engineering values-with-units in arbitrary text.

    Pass 1 scans ``number [unit] unit-token`` patterns (including
    slash-lists like ``66/33/11 kV``).  Pass 2 detects ranges
    (``20 to 40 °C``, ``22.5 – 24.3 kV``) and merges them into pass-1
    results or appends them as range-only values.
    """
    out: list[EngineeringValue] = []
    consumed: list[tuple[int, int]] = []
    starts_seen: set[int] = set()

    def build(start: int, num_txt: str, udef: UnitDef, token: str, end: int,
              is_list_item: bool) -> EngineeringValue | None:
        """Construct one value.  ``start`` is the absolute position of the
        number, ``end`` the absolute end of the shared unit token."""
        clean = num_txt.replace(",", "").replace(" ", "")
        try:
            value = Decimal(clean)
        except InvalidOperation:
            return None
        try:
            q0 = _Q(float(clean), _pint_expr_for_label(token))
            dimension = str(q0.dimensionality)
        except Exception:
            return None
        sentence, _ = _sentence_of(text, start)
        # the qualifier zone ends *before* the unit token: otherwise "1 min"
        # would look like the qualifier word "minimum"
        unit_begin = end - len(token.strip())
        qualifier = _qualifier_in_context(text, start,
                                          unit_span=(unit_begin, end))
        tol = _tolerance_pct(text, start, end, clean.lstrip("-"))
        temp_c = None
        if udef.key not in ("temperature_c", "temperature_k"):
            tcm = re.search(r"(\d+(?:\.\d+)?)\s*°C", sentence)
            if tcm:
                temp_c = float(tcm.group(1))
        phase = None
        pm = re.search(r"(phase[- ]to[- ](?:neutral|earth|ground)|phase[- ]phase|"
                       r"line[- ]to[- ]line|single[- ]phase|three[- ]phase)", sentence, re.I)
        if pm:
            phase = pm.group(1).lower().replace("  ", " ")
        vm = _VARIANT_RE.search(sentence)
        try:
            mag_si = float(_Q(float(clean), _pint_expr_for_label(token))
                           .to_base_units().magnitude)
        except Exception:
            mag_si = float("nan")
        return EngineeringValue(
            raw=(num_txt + " " + token) if is_list_item else text[start:end].strip(),
            value=value, unit_label=token, canonical=udef.key, dimension=dimension,
            magnitude_si=mag_si, qualifier=qualifier, tolerance_pct=tol,
            temperature_c=temp_c, phase_note=phase,
            variant=vm.group(0) if vm else None,
            sentence=sentence.strip(), char_start=start, char_end=end)

    for m in _NUM_RE.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        if m.start() > 0 and text[m.start() - 1] in "0123456789./":
            # '1' in "12.5" or "m/s" continuation; allow decimals and commas
            prev = text[m.start() - 1]
            ok_ctx = (prev in ".," and m.start() >= 2
                      and (text[m.start() - 2].isdigit() if prev == "." else True))
            if not ok_ctx:
                continue
            if prev == ",":
                continue        # thousands part
        start = m.start()
        raw_num = m.group(0)
        nums = [(start, raw_num)]
        unit_pos = m.end()
        list_m = _LISTY_RE.match(text, m.end())
        if list_m:
            nums += [(list_m.start() + lm.start(), lm.group(0))
                     for lm in re.finditer(_NUM, list_m.group(0))]
            unit_pos = list_m.end()
        um = _match_unit(text, unit_pos)
        if um is None:
            continue
        udef, token, end = um
        consumed.append((start, end))
        if _EXCLUDE_BEFORE.search(text[max(0, start - 28):start]):
            continue
        starts_seen.add(start)
        for i, (n_pos, n_txt) in enumerate(nums):
            v = build(n_pos, n_txt, udef, token, end, i > 0)
            if v is not None:
                if i > 0:
                    starts_seen.add(n_pos)
                out.append(v)
        if len(out) >= max_values:
            break

    # pass 2: ranges — attach a range tuple to values already extracted at
    # either bound, or append a range-only value when neither bound carried a
    # unit of its own.
    for rm in _RANGE_SCAN_RE.finditer(text):
        try:
            a = Decimal(rm.group(1).replace(",", "").lstrip("+"))
            b = Decimal(rm.group(3).replace(",", "").lstrip("+"))
        except InvalidOperation:
            continue
        if a == b or "±" in text[max(0, rm.start() - 25):min(len(text), rm.end() + 25)]:
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        um = _match_unit(text, rm.start(4))
        if um is None:
            continue
        udef, token, end = um
        wanted = {a, b, lo, hi, abs(a), abs(b)}
        related = [v for v in out if v.canonical == udef.key
                   and abs(v.value) in wanted]
        if related:
            for v in related:
                if v.range_tuple is None:
                    v.range_tuple = (lo, hi)
            continue
        for pos, num_txt in ((rm.start(3), rm.group(3)), (rm.start(1), rm.group(1))):
            if pos in starts_seen:
                continue
            v = build(pos, num_txt, udef, token, end, False)
            if v is not None:
                v.range_tuple = (lo, hi)
                out.append(v)
                starts_seen.add(pos)

    # drop the tolerance number itself when it appears as a bare "± X %"
    def _is_bare_tol(v: EngineeringValue) -> bool:
        prev = text[max(0, v.char_start - 3):v.char_start]
        if "±" in prev:
            return True
        before = text[max(0, v.char_start - 10):v.char_start]
        return bool(re.search(r"[±]\s*[\d.,]*\s*(?:of\s+|x\s*\d+\s*[x×]\s*)?$",
                              before))
    out = [v for v in out if not _is_bare_tol(v)]
    # Table rows ("a | b | c"): a current-rating cell in the same row (e.g.
    # "1250 A") identifies which section a value belongs to; claims that
    # quote a value from another row of the same table are then detectable.
    lines = text.splitlines() if text else []
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    for v in out:
        li = next((i for i, off in enumerate(offsets)
                   if off <= v.char_start < off + len(lines[i]) + 1), None)
        cur_line = lines[li] if li is not None else ""
        rowline = pdf_row_line(text, v.char_start) or v.sentence or ""
        if "|" not in cur_line and v.sentence:
            pm = re.search(r"\b(\d+(?:\.\d+)?)\s*A\s+"
                           r"(?:bus\s*|busbar\s*|cable\s*|switchgear\s*)?"
                           r"(?:section|rating|feeder|panel|compartment)\b",
                           v.sentence, re.I)
            if pm and pm.group(1) + " A" != str(v.value) + " A":
                v.row_tags = (pm.group(1) + " A",)
        # The row *label* is the FIRST cell of the row (e.g. "1250 A | … |
        # 31.5 kA for 3 s").  Other A-cells in the row are data columns, never
        # identifiers — treating them as labels produced false cross-row
        # conflicts on tables with several current columns.
        label_cell = ""
        if "|" in cur_line:
            label_cell = cur_line.split("|")[0]
        elif rowline != cur_line and "\n" in rowline:
            first = rowline.splitlines()[0].strip()
            label_cell = first.split("|")[0] if "|" in first else first
        if label_cell:
            mine = str(v.value)
            m = _ROW_TAG_RE.search(label_cell)
            if m and m.group(1) != mine:
                v.row_tags = (m.group(1) + " A",)
    out.sort(key=lambda v: v.char_start)
    return out


_ROW_LABEL_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*A\b")


def pdf_row_line(text: str, pos: int) -> str | None:
    """Rebuild a table row from a *column-wise* extraction.

    Some PDF layouts (and most text fallbacks) emit one cell per line, so a
    value's line carries no row label.  Walk upward over consecutive bare cell
    lines until the header row (a '|'-row) or a prose line, and return the
    reconstructed context — enough to attribute a value to its row's first
    cells.  Returns the plain line when it is already a pipe-row."""
    lines = text.splitlines()
    acc = 0
    li = None
    for i, ln in enumerate(lines):
        nxt = acc + len(ln) + 1
        if acc <= pos < nxt:
            li = i
            break
        acc = nxt
    if li is None:
        return None
    cur = lines[li]
    if "|" in cur:
        return cur
    up: list[str] = []
    j = li - 1
    while j >= 0 and len(up) < 8:
        prev = lines[j].strip()
        if "|" in prev:
            if up:
                up.append(prev)
            break
        if not prev or re.match(r"^[A-Z][a-z]+\b.*\.$", prev) \
                or len(prev.split()) > 6:
            break                      # prose line: row context ends here
        up.append(prev)
        j -= 1
    return "\n".join(list(reversed(up)) + [cur]) if up else cur


def _qualifier_in_context(text: str, abs_start: int,
                          unit_span: tuple[int, int] | None = None) -> str | None:
    """Qualifying word (rated/max/min/typical/…) governing this number.

    Search zone: the *clause* containing the number (clause = text between
    , ; . boundaries).  A qualifier before the number wins (nearest); one
    after the number counts only for table-row style short clauses.
    """
    # table rows ("| … |") are their own clauses: a qualifier from the
    # neighbouring row must not leak across
    # Clause boundaries, skipping separators that are part of a number
    # ("145.5", "1.200" thousands, "145,5" decimal comma): a '.' inside the
    # numeric token must not cut the clause.
    def _boundary(pos: int, ch: str) -> bool:
        if ch == "." and 0 <= pos < len(text) - 1 and text[pos + 1].isdigit():
            return False
        if ch == "," and 0 <= pos < len(text) - 1 and text[pos + 1].isdigit() \
                and pos > 0 and text[pos - 1].isdigit():
            return False
        return True
    left = -1
    for c in ",;\n.":
        i = abs_start
        while True:
            i = text.rfind(c, 0, i)
            if i < 0:
                break
            if _boundary(i, c):
                left = max(left, i)
                break
            i -= 1
    sent_bounds = [m.start() for m in _SENT_SPLIT.finditer(text) if m.start() < abs_start]
    left = max([left] + sent_bounds) + 1
    right_m = [m.start() for m in _SENT_SPLIT.finditer(text) if m.start() >= abs_start]
    for c in ",;\n.":
        cm = abs_start - 1
        while True:
            cm = text.find(c, cm + 1)
            if cm < 0:
                break
            if _boundary(cm, c):
                right_m.append(cm)
                break
    # NOTE: `right_m` can legitimately be empty (value at end of a line with
    # no trailing punctuation); min() of [] must NOT fall back to -1 — that
    # silently produced an empty clause and dropped qualifiers.
    right = min([r for r in right_m if r > abs_start], default=len(text))
    if right <= abs_start:
        right = len(text)
    clause = text[left:right]
    rel = abs_start - left
    def in_unit(rel_pos: int) -> bool:
        """True when a clause-relative match position lands inside the unit
        token (e.g. 'min' in '1 min' must not read as the qualifier
        'minimum')."""
        if unit_span is None:
            return False
        return left + unit_span[0] <= left + rel_pos <= left + unit_span[1]

    best: tuple[int, str] | None = None
    table_row = "|" in clause
    for name, pat in _QUALIFIER_PATTERNS:
        for cand in pat.finditer(clause):
            if cand.start() < rel and not in_unit(cand.start()) \
                    and (best is None or cand.start() > best[0]):
                best = (cand.start(), name)
    if best is None and (rel <= 12 or table_row):
        for name, pat in _QUALIFIER_PATTERNS:
            for cand in pat.finditer(clause):
                if cand.start() >= rel and not in_unit(cand.start()):
                    if best is None or cand.start() < best[0]:
                        best = (cand.start(), name)
                    break
    return best[1] if best else None


def _tolerance_pct(text: str, start: int, end: int, raw_num: str) -> float | None:
    """±tolerance is only taken from a position directly attached to the
    value (after its unit, or immediately before its number) — never from an
    unrelated part of the sentence."""
    after = text[end:end + 24]
    tm = _TOL_AFTER_RE.match(after)
    zone_before = text[max(0, start - 30):start]
    tb = None
    for m in _TOL_BEFORE_RE.finditer(zone_before):
        tb = m
    if tb and not tm and not after.lstrip().startswith("±"):
        tm = tb
    if not tm:
        return None
    try:
        tval = Decimal(tm.group(1).replace(",", ""))
        if tm.group(2) == "%":
            return float(abs(tval))
        base = Decimal(raw_num.replace(",", ""))
        if base == 0:
            return None
        return float(abs(tval) / base * 100)
    except (InvalidOperation, ZeroDivisionError):
        return None


def _range_tuple(text: str, start: int, end: int) -> tuple[Decimal, Decimal] | None:
    zone = text[max(0, start - 2):end + 42]
    rm = _RANGE_RE.search(zone)
    if not rm:
        return None
    try:
        a = Decimal(re.sub(r"[,^\s]", "", rm.group(1)).strip("-") if rm.group(1)[0] != "-"
                    else rm.group(1)[0] + re.sub(r"[,\s]", "", rm.group(1)[1:]))
        b = Decimal(re.sub(r"[,^\s]", "", rm.group(2)))
    except InvalidOperation:
        try:
            a = Decimal(re.sub(r"[^\d.\-]", "", rm.group(1)))
            b = Decimal(re.sub(r"[^\d.\-]", "", rm.group(2)))
        except InvalidOperation:
            return None
    if a == 0 or b == 0:
        return None
    return (a, b) if a <= b else (b, a)


# ------------------------------------------------------------------ compare
@dataclass
class NumericComparison:
    status: str            # verified | unit_verified | mismatch | insufficient | ambiguous
    explanation: str
    claim: EngineeringValue | None
    source: EngineeringValue | None
    tolerance_pct: float
    relative_diff_pct: float = 0.0
    how: str = ""


_POWER_CATS = {"active_power", "apparent_power", "reactive_power", "energy"}


def compare_values(claim: EngineeringValue, source: EngineeringValue,
                   rel_tol: float) -> NumericComparison:
    """Compare one claim value with one candidate source value.

    Conversion rules (enforced here, tested heavily):
    * same category & unit      → direct Decimal comparison ("same-unit");
    * same category, prefixed   → exact Decimal prefix scaling ("converted");
    * °C ↔ K                    → offset-aware pint conversion;
    * W ↔ VA ↔ var ↔ Wh         → **never** converted → ``ambiguous`` (needs PF,
      demand factor or time);
    * different dimensions      → ``mismatch``.
    """
    rel_tol_pct = float(Decimal(str(rel_tol)) * 100)
    same_category = claim.canonical == source.canonical
    same_dimension = claim.dimension == source.dimension
    offset_pair = {claim.canonical, source.canonical} == {"temperature_c", "temperature_k"}
    if not same_category and not offset_pair:
        if claim.canonical in _POWER_CATS or source.canonical in _POWER_CATS:
            return NumericComparison(
                "ambiguous",
                f"The answer states {claim.category_label} ({claim.raw}) while the "
                f"source states {source.category_label} ({source.raw}); these are not "
                "interconvertible without extra parameters (power factor / load / time).",
                claim, source, rel_tol_pct)
        if not same_dimension:
            return NumericComparison(
                "mismatch",
                f"Different physical dimensions: answer says {claim.category_label} "
                f"({claim.raw}), source says {source.category_label} ({source.raw}).",
                claim, source, rel_tol_pct)
    if offset_pair:
        try:
            t1 = _Q(float(claim.value), _pint_expr_for_label(claim.unit_label)).to(
                _pint_expr_for_label(source.unit_label))
            claim_base = Decimal(str(t1.magnitude))
        except Exception:
            return NumericComparison("ambiguous", "Temperature conversion failed",
                                     claim, source, rel_tol_pct)
        source_base = source.value
        how = "temp-converted"
    else:
        claim_base, source_base = claim.base_magnitude(), source.base_magnitude()
        # exact conversion: base magnitudes agree under exact Decimal scaling
        exact_conv = claim_base == source_base
        if claim.canonical == source.canonical and claim.value == source.value and \
                _scale_to_base(re.sub(r"\s+", "", claim.unit_label)) == \
                _scale_to_base(re.sub(r"\s+", "", source.unit_label)):
            how = "same-unit"
        elif exact_conv:
            how = "converted"
        else:
            how = "tolerance"
    scope_conflict = _scope_conflict(claim, source)
    if (claim.row_tags and source.row_tags
            and not (set(claim.row_tags) & set(source.row_tags))
            and not (set(claim.row_tags) & {str(source.value)})):
        return NumericComparison(
            "mismatch",
            f"The source lists {source.raw} in the row identified by "
            f"{', '.join(source.row_tags)}, while the answer attributes it to "
            f"{', '.join(claim.row_tags)} — same table, different section/row; "
            "verify which row was meant.",
            claim, source, 0.0, 0.0, "row-misattribution")
    equal, reldiff = equal_within(claim_base, source_base, rel_tol)
    reldiff_pct = float(reldiff * 100)
    if equal and scope_conflict:
        return NumericComparison(
            "mismatch",
            (f"Numerically {claim.raw} matches source value {source.raw}, but the "
             f"answer calls it '{claim.qualifier}' while the source says "
             f"'{source.qualifier}' — different engineering scopes; do not treat "
             "as the same quantity."),
            claim, source, rel_tol_pct, reldiff_pct, "scope-conflict")
    if equal:
        status = {"same-unit": "verified", "converted": "unit_verified",
                  "temp-converted": "unit_verified",
                  "tolerance": "verified_within_tolerance"}[how]
        expl = "Value matches the cited source"
        if how == "tolerance":
            expl = (f"Answer value {claim.raw} is not exactly the source value "
                    f"{source.raw} but agrees within the configured tolerance "
                    f"(diff {reldiff_pct:.2f}% ≤ {rel_tol_pct:.2f}%)")
        if how in ("converted", "temp-converted"):
            expl += f" after converting {claim.raw} to {source.value} {source.unit_label}" \
                f" (= {source.base_magnitude():g} base units)"
        if source.range_tuple:
            lo, hi = source.range_tuple
            if lo <= source_base / (source_base / source.value if source_base else 1) <= hi \
                    or source.value in (lo, hi):
                expl += f"; the source expresses it as a range {lo}–{hi}"
        if source.tolerance_pct is not None:
            expl += f"; source tolerance ±{source.tolerance_pct:g}%"
        if claim.qualifier and source.qualifier and claim.qualifier != source.qualifier:
            expl += (f"; NOTE: the answer calls it '{claim.qualifier}' but the source "
                     f"labels it '{source.qualifier}' — scope caveat applies")
            status = "ambiguous"
        return NumericComparison(status, expl + ".", claim, source, rel_tol_pct,
                                 reldiff_pct, how)
    expl = (f"Answer value {claim.raw} vs source value {source.raw} "
            f"(difference {reldiff_pct:.1f}% exceeds tolerance {rel_tol_pct:.1f}%)")
    if claim.qualifier and source.qualifier and claim.qualifier != source.qualifier:
        expl += (f"; the answer says '{claim.qualifier}' but the source says "
                 f"'{source.qualifier}' — different engineering scope")
    elif bool(claim.qualifier) != bool(source.qualifier):
        expl += "; qualifiers do not align (one side is unqualified)"
    return NumericComparison("mismatch", expl + ".", claim, source, rel_tol_pct,
                             reldiff_pct, how)


# Qualifier scope model: 'rated/nameplate' is treated as the unqualified
# baseline (nameplate tables list rated values without saying so).  A claim
# with an explicit modifier that the source does not carry (typical/measured/
# design/guaranteed) or an outright conflicting scope is not a match.
_SCOPE_EQUIV = {None: "rated", "rated": "rated"}


def _scope_conflict(claim: EngineeringValue, source: EngineeringValue) -> bool:
    a = _SCOPE_EQUIV.get(claim.qualifier, claim.qualifier)
    b = _SCOPE_EQUIV.get(source.qualifier, source.qualifier)
    if a == b:
        return False
    # missing scope info on one side (None after mapping impossible here since
    # None maps to rated) — only genuine label differences conflict
    return True


def equal_within(a: Decimal, b: Decimal, rel_tol: float) -> tuple[bool, Decimal]:
    if a == b:
        return True, Decimal(0)
    denom = max(abs(a), abs(b))
    if denom == 0:
        return a == b, Decimal(1)
    diff = abs((a - b) / denom)
    return diff <= Decimal(str(rel_tol)), diff


def _fmt(x: Decimal | float) -> str:
    if isinstance(x, float):
        return f"{x:g}"
    if x == x.to_integral_value():
        return str(int(x))
    return f"{x:.6g}"


def values_comparable(a: EngineeringValue, b: EngineeringValue) -> bool:
    if {a.canonical, b.canonical} == {"temperature_c", "temperature_k"}:
        return True
    if a.canonical == b.canonical:
        return True
    return a.dimension == b.dimension and not ({a.canonical, b.canonical} & _POWER_CATS)


def power_categories() -> set[str]:
    return set(_POWER_CATS)
