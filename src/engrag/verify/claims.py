"""Claim-to-evidence verification and citation integrity.

Two distinct checks, deliberately kept apart:

1. **Citation integrity** — does the quoted passage actually exist in the
   cited document/page, and does the chunk hash still match the stored text
   (drift detection after reindex)?  Mechanical, exact.

2. **Claim support** — is the *factual content* of the sentence consistent
   with the evidence?  Engineering values in a sentence must appear in the
   quoted excerpt verbatim or as an exact unit conversion of a source value;
   otherwise the sentence is reported weak/insufficient/contradicted.
   Overlap alone never marks a claim "supported".

Additionally, ``find_conflicts`` detects *conflicting values across
evidence* (two documents stating different values for the same qualified
quantity), which the pipeline reports honestly instead of choosing one.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..db import Database
from .numbers import EngineeringValue, compare_values, extract_values
from .textmatch import phrase_in_text

_ASSET_RE = re.compile(
    r"\b(?:TR|TC|CB|VCB|ACB|MCC|MCP|VSD|VFD|DOL|PCC|TX|DEMO-[A-Z]+)-\d{1,5}\b",
    re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9*[])")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class NumericCheck:
    claim: str
    status: str            # verified|unit_verified|verified_within_tolerance|
                           # mismatch|insufficient|ambiguous
    answer_value: EngineeringValue | None
    source_value: EngineeringValue | None
    page_number: int
    document_id: int
    document_title: str
    explanation: str
    context_label: str
    tolerance_pct: float
    sentence_index: int = -1

    def to_dict(self) -> dict:
        return {
            "claim": self.claim, "status": self.status,
            "answer_value": self.answer_value.raw if self.answer_value else "",
            "source_value": self.source_value.raw if self.source_value else "",
            "answer_unit": self.answer_value.unit_label if self.answer_value else "",
            "source_unit": self.source_value.unit_label if self.source_value else "",
            "answer_category": self.answer_value.category_label if self.answer_value else "",
            "source_category": self.source_value.category_label if self.source_value else "",
            "document_id": self.document_id, "document_title": self.document_title,
            "page_number": self.page_number,
            "explanation": self.explanation, "context": self.context_label,
            "tolerance_pct": round(self.tolerance_pct, 3),
            "sentence_index": self.sentence_index,
            "answer_context": self.answer_value.context_summary() if self.answer_value else {},
            "source_context": self.source_value.context_summary() if self.source_value else {},
        }


@dataclass
class CitationVerdict:
    marker: str
    chunk_id: int | None
    document_id: int
    document_title: str
    page_number: int
    printed_label: str
    quote: str
    integrity: str          # ok | stale_chunk | missing_quote | invalid_ref
    support: str            # supported | weak | contradicted | insufficient
    quote_in_source: bool
    source_hash_now: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"marker": self.marker, "chunk_id": self.chunk_id,
                "document_id": self.document_id, "document_title": self.document_title,
                "page_number": self.page_number, "printed_label": self.printed_label,
                "quote": self.quote, "integrity": self.integrity, "support": self.support,
                "quote_in_source": self.quote_in_source,
                "source_hash_now": self.source_hash_now, "note": self.note}


@dataclass
class SentenceVerdict:
    index: int
    text: str
    support: str            # supported | weak | contradicted | insufficient | advisory
    citations: list[CitationVerdict] = field(default_factory=list)
    numeric_checks: list[NumericCheck] = field(default_factory=list)
    note: str = ""
    suggested_replacement: str | None = None

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "support": self.support,
                "note": self.note, "citations": [c.to_dict() for c in self.citations],
                "numeric_checks": [n.to_dict() for n in self.numeric_checks],
                "suggested_replacement": self.suggested_replacement}


@dataclass
class VerificationReport:
    sentences: list[SentenceVerdict] = field(default_factory=list)
    numeric_checks: list[NumericCheck] = field(default_factory=list)
    citations: list[CitationVerdict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    overall_support: str = "unknown"       # grounded | partially_grounded | unverified
    corrected_answer: str | None = None
    abstain: bool = False
    abstain_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_support": self.overall_support,
            "abstain": self.abstain, "abstain_reason": self.abstain_reason,
            "corrected_answer": self.corrected_answer,
            "conflicts": self.conflicts,
            "sentences": [s.to_dict() for s in self.sentences],
            "numeric_checks": [n.to_dict() for n in self.numeric_checks],
            "citations": [c.to_dict() for c in self.citations],
        }


class AnswerVerifier:
    """Verifies an answer's citations, quotes and numeric claims."""

    def __init__(self, db: Database, *, rel_tolerance: float = 1e-9):
        self.db = db
        self.rel_tolerance = rel_tolerance

    # ------------------------------------------------------------- parsing
    @staticmethod
    def split_sentences(answer: str) -> list[str]:
        """Split at . ! ; : but never inside an open “quoted block”, and
        never immediately after a dangling citation marker."""
        text = answer.strip()
        parts: list[str] = []
        buf: list[str] = []
        i = 0
        n = len(text)
        quotes_open = 0
        while i < n:
            ch = text[i]
            buf.append(ch)
            if ch in "“":
                quotes_open += 1
            elif ch in "”":
                quotes_open = max(0, quotes_open - 1)
            elif ch == "\n" and quotes_open == 0:
                ahead = text[i + 1:]
                stripped = ahead.lstrip("\n ")
                # a numbered evidence block or a list bullet always starts a
                # fresh sentence: each carries its own quotation and its own
                # citation marker, and merging them would attribute a quote to
                # the wrong passage
                if (("".join(buf).strip())
                        and (re.match(r"^\d{1,2}\.\s", stripped)
                             or re.match(r"^[\u2022\-\*]\s", stripped))):
                    parts.append("".join(buf).strip())
                    buf = []
            elif ch in ".!?;:" and quotes_open == 0:
                j = i + 1
                while j < n and text[j] in "*_` ":
                    j += 1
                if j < n and text[j] in "\n":
                    nxt = text[j:].lstrip("\n")
                    if not nxt or not (nxt[0].isupper() or nxt[0] in "0123456789*[•"):
                        i += 1
                        continue
                    if re.search(r"\[[^\]]{0,30}$", "".join(buf)):
                        i += 1
                        continue          # keep dangling marker with next part
                    parts.append("".join(buf).strip())
                    buf = []
            i += 1
        if buf:
            parts.append("".join(buf).strip())
        return [p2 for p2 in parts if p2]

    CITE_FULL_RE = re.compile(r"\[D(\d+)\s*p(\d+)(?::q(\d+))?\]")

    @staticmethod
    def parse_citations(sentence: str):
        """Return [(marker, doc_id, page, chunk_id)] for [D3 p12:q9]-style and
        [n] positional markers."""
        found = []
        for m in AnswerVerifier.CITE_FULL_RE.finditer(sentence):
            found.append((m.group(0), int(m.group(1)), int(m.group(2)),
                          int(m.group(3)) if m.group(3) else None))
        for m in re.finditer(r"(?<!\d)\[(\d{1,2})\](?!\d)", sentence):
            found.append((m.group(0), None, None, None))
        return found

    # --------------------------------------------------------------- verify
    def verify(self, answer: str, evidence: list[dict],
               question: str = "") -> VerificationReport:
        """evidence items: {ref_index, chunk_id, document_id, page_start,
        page_end, text, excerpt?, text_sha1, title, matched_page?}"""
        report = VerificationReport()
        ev_by_ref = {e["ref_index"]: e for e in evidence}
        ev_by_doc_page: dict[tuple[int, int], list[dict]] = {}
        for e in evidence:
            page = e.get("matched_page") or e["page_start"]
            ev_by_doc_page.setdefault((e["document_id"], page), []).append(e)
        value_cache: dict[int, list[EngineeringValue]] = {}

        def values_of(e: dict, *, excerpt: bool = False) -> list[EngineeringValue]:
            key = e.get("chunk_id") if e.get("chunk_id") is not None else id(e)
            key = f"{'x' if excerpt else 't'}{key}"
            if key not in value_cache:
                text = e.get("excerpt") if excerpt and e.get("excerpt") else e["text"]
                value_cache[key] = extract_values(text)
            return value_cache[key]

        all_source_values = [(v, e) for e in evidence for v in values_of(e)]

        # The extractive digest ends with a navigation aid that re-lists
        # already-cited passages side by side.  Those lines are quotations, not
        # independent claims: checking their numbers against *other* passages
        # would report the corpus's own disagreements twice (the conflict
        # report owns that job).  They are still anti-fabrication checked — a
        # navigation excerpt that cannot be found verbatim is flagged.
        cut = answer.find(NAV_SECTION_HEADING)
        main_part, nav_part = (answer, "") if cut < 0 else (answer[:cut], answer[cut:])

        sentences = self.split_sentences(main_part)
        for i, sent in enumerate(sentences):
            verdict = self._verify_sentence(i, sent, ev_by_ref, ev_by_doc_page,
                                            all_source_values, values_of)
            report.sentences.append(verdict)
            report.citations.extend(verdict.citations)
            report.numeric_checks.extend(verdict.numeric_checks)

        for sent in self.split_sentences(nav_part):
            if NAV_SECTION_HEADING in sent:
                sent = sent.split(NAV_SECTION_HEADING, 1)[1].strip()
                if not sent:
                    continue
            report.sentences.append(
                self._verify_navigation_excerpt(len(report.sentences), sent, evidence))

        cited_ev = [e for e in evidence]
        report.conflicts = find_conflicts(cited_ev, rel_tol=self.rel_tolerance)

        statuses = [s.support for s in report.sentences]
        if statuses and all(s in ("supported", "advisory") for s in statuses):
            report.overall_support = "grounded"
        elif any(s == "supported" for s in statuses):
            report.overall_support = "partially_grounded"
        else:
            report.overall_support = "unverified"
        mism = [n for n in report.numeric_checks if n.status == "mismatch"]
        if mism:
            report.abstain = True
            report.abstain_reason = (
                f"{len(mism)} numerical value(s) conflict with the cited "
                "evidence; the answer was withheld to avoid propagating an "
                "unsafe figure.")
        bad_cites = [c for c in report.citations
                     if c.integrity in ("stale_chunk", "missing_quote", "invalid_ref")]
        if bad_cites and not report.abstain:
            report.abstain = True
            report.abstain_reason = (
                "One or more citations failed integrity verification (quoted "
                "text not found at the cited page, or index drift detected); "
                "the answer was withheld.")
        return report

    # ---------------------------------------------------------- per sentence
    def _verify_navigation_excerpt(self, idx: int, text: str,
                                   evidence: list[dict]) -> SentenceVerdict:
        """Quote-existence check for a digest navigation line (no value claims)."""
        verdict = SentenceVerdict(index=idx, text=text, support="advisory")
        quotes = [q for _, _, q in quoted_spans(re.sub(r"\[[^\]]{0,40}\]", " ", text))]
        if not quotes:
            verdict.note = "navigation excerpt without a quotation; not a claim"
            return verdict
        missing = [q for q in quotes if _quote_in_evidence(q, evidence) is None]
        if missing:
            verdict.support = "insufficient"
            verdict.note = ("navigation excerpt not found verbatim in the retrieved "
                            "evidence — the quotation may be fabricated")
        else:
            verdict.note = ("quotation verified verbatim against the cited evidence "
                            "(navigation aid, not an independent claim)")
        return verdict

    def _verify_sentence(self, idx, sentence, ev_by_ref, ev_by_doc_page,
                         all_source_values, values_of,
                         all_evidence: list[dict] | None = None) -> SentenceVerdict:
        body = re.sub(r"\[[^\]]{0,40}\]", " ", sentence)   # strip markers
        verdict = SentenceVerdict(index=idx, text=sentence, support="insufficient")

        markers = self.parse_citations(sentence)
        cited_evidence: list[dict] = []
        for marker, doc_id, page, chunk_id in markers:
            ev = None
            if doc_id is not None and page is not None:
                cands = ev_by_doc_page.get((doc_id, page), [])
                if chunk_id is not None:
                    cands = [c for c in cands if c.get("chunk_id") == chunk_id]
                ev = cands[0] if cands else None
                if ev is None and chunk_id is not None:
                    ev = ev_by_doc_page.get((doc_id, page), [None])[0]
                if ev is None:
                    verdict.citations.append(CitationVerdict(
                        marker=marker, chunk_id=None, document_id=doc_id,
                        document_title=self._title_of(doc_id), page_number=page,
                        printed_label="", quote="", integrity="invalid_ref",
                        support="insufficient", quote_in_source=False,
                        source_hash_now="",
                        note="cited page is not covered by retrieved evidence"))
                    continue
            else:
                mpos = re.fullmatch(r"\[(\d{1,2})\]", marker)
                if mpos:
                    ev = ev_by_ref.get(int(mpos.group(1)))
            if ev is None:
                verdict.citations.append(CitationVerdict(
                    marker=marker, chunk_id=None, document_id=doc_id or -1,
                    document_title=self._title_of(doc_id) if doc_id else "",
                    page_number=page or 0, printed_label="", quote="",
                    integrity="invalid_ref", support="insufficient",
                    quote_in_source=False, source_hash_now="",
                    note="citation does not reference retrieved evidence"))
                continue
            cited_evidence.append(ev)
            verdict.citations.append(self._check_citation(marker, ev, body))

        claim_values = [v for v in extract_values(body)
                        if v.char_start >= _heading_span(body)]
        has_values = bool(claim_values)
        if not cited_evidence:
            if not has_values:
                verdict.support = "advisory"
                verdict.note = ("non-claim text (labels, headings or guidance; "
                                "no engineering values asserted)")
                return verdict
            cited_evidence = list(ev_by_ref.values())
            verdict.note = ("claim carries no citation marker; checked against "
                            "all retrieved evidence")

        # ---- numeric claims — verified against the QUOTED excerpt first.
        # Verbatim-quoted values ("the answer writes exactly what the source
        # states") are integrity-verified by the quote match itself; residual
        # internal inconsistencies of quoted tables are reported separately by
        # find_conflicts rather than turning the whole digest into "mismatch".
        claim_assets = {a.upper() for a in _ASSET_RE.findall(body)}
        quote_spans = quoted_spans(body)
        # resolve each distinct quotation once (several values usually sit in
        # the same quote) — keeps verification linear instead of quadratic
        quote_hits: dict[str, tuple[dict, str] | None] = {}

        def enclosing_quote(value: EngineeringValue):
            for a, b, q in quote_spans:
                if a <= value.char_start < b:
                    return q
            return None

        for v in claim_values:
            quoted = enclosing_quote(v)
            if quoted is not None:
                # A quotation counts as genuine when the exact wording exists
                # in ANY retrieved passage (the answer may group several
                # excerpts per line); the citation marker only says where the
                # *answer* points, the quote check says whether it exists.
                pool = (all_evidence or []) + [e for e in cited_evidence
                                               if e not in (all_evidence or [])]
                if quoted not in quote_hits:
                    quote_hits[quoted] = _quote_in_evidence(quoted, pool) or (
                        _quote_in_evidence(quoted, cited_evidence)
                        if not pool else None)
                found = quote_hits[quoted]
                if found is not None:
                    hit, how = found
                    verdict.numeric_checks.append(NumericCheck(
                        claim=v.raw, status="verified", answer_value=v,
                        source_value=v, document_id=hit["document_id"],
                        document_title=hit.get("title", ""),
                        page_number=hit.get("matched_page") or hit["page_start"],
                        explanation=(
                            "Quoted " + ("verbatim" if how == "verbatim"
                                         else "from the source (opening words "
                                              "matched; the quotation was "
                                              "clipped in the answer)")
                            + f" — the wording was found on page "
                            f"{hit.get('matched_page') or hit['page_start']} of "
                            f"{hit.get('title', 'the document')}, so the "
                            "transcription is correct. (A quotation is not an "
                            "endorsement: if the document disagrees with "
                            "another source, see the conflict report.)"),
                        context_label=v.qualifier or "",
                        tolerance_pct=self.rel_tolerance * 100, sentence_index=idx))
                    continue
                ev0 = (cited_evidence or all_evidence or [{}])[0]
                verdict.numeric_checks.append(NumericCheck(
                    claim=v.raw, status="insufficient", answer_value=v,
                    source_value=None,
                    document_id=ev0.get("document_id", -1),
                    document_title=ev0.get("title", ""),
                    page_number=ev0.get("matched_page") or ev0.get("page_start", 0),
                    explanation=("The answer presents this number inside a "
                                 "quotation, but the quoted wording was not "
                                 "found in the cited passage — the quotation "
                                 "may be fabricated or mis-cited."),
                    context_label=v.qualifier or "",
                    tolerance_pct=self.rel_tolerance * 100, sentence_index=idx))
                continue
            ev_hit = self._verbatim_source(v, cited_evidence, body, values_of)
            if ev_hit is not None and ev_hit[0] == "row-conflict":
                ev, sv = ev_hit[1]
                check = self._verify_value(v, idx, [ev],
                                           [], values_of, claim_assets)
                check.status = "mismatch"
                check.source_value = sv
                check.explanation = (
                    f"The cited table attributes {sv.raw} to the row "
                    f"{', '.join(sorted(set(v.row_tags or ()) & set(sv.row_tags)))}, not "
                    f"{v.raw}; the value quoted in the answer belongs to "
                    "another row of that table.")
                verdict.numeric_checks.append(check)
                continue
            if ev_hit is not None and ev_hit[0] == "ambiguous-row":
                ev, sv, cand = ev_hit[1]
                verdict.numeric_checks.append(NumericCheck(
                    claim=v.raw, status="ambiguous", answer_value=v,
                    source_value=sv, document_id=ev["document_id"],
                    document_title=ev.get("title", ""),
                    page_number=ev.get("matched_page") or ev["page_start"],
                    explanation=(f"{v.raw} appears in a table whose row labels "
                                 f"were extracted as separate lines; nearby row "
                                 f"labels: {', '.join(sorted(cand))}. The exact "
                                 "row/value alignment could not be established "
                                 "mechanically — open the source page to "
                                 "confirm which row applies."),
                    context_label=sv.qualifier or "",
                    tolerance_pct=self.rel_tolerance * 100, sentence_index=idx))
                continue
            if ev_hit is not None and ev_hit[0] == "scope":
                ev, sv = ev_hit[1]
                verdict.numeric_checks.append(NumericCheck(
                    claim=v.raw, status="ambiguous", answer_value=v, source_value=sv,
                    document_id=ev["document_id"], document_title=ev.get("title", ""),
                    page_number=ev.get("matched_page") or ev["page_start"],
                    explanation=(f"{v.raw} does appear in the cited passage, but the "
                                 f"source line presents it as "
                                 f"'{sv.qualifier or 'unqualified'}', not "
                                 f"'{v.qualifier}' — verify the engineering scope "
                                 "before relying on it."),
                    context_label=sv.qualifier or "",
                    tolerance_pct=self.rel_tolerance * 100, sentence_index=idx))
                continue
            if ev_hit is not None:
                ev, sv = ev_hit
                verdict.numeric_checks.append(NumericCheck(
                    claim=v.raw, status="verified",
                    answer_value=v, source_value=sv,
                    document_id=ev["document_id"],
                    document_title=ev.get("title", ""),
                    page_number=ev.get("matched_page") or ev["page_start"],
                    explanation=("The value is quoted verbatim in the cited "
                                 "excerpt, in a matching context (exact textual "
                                 "match)."),
                    context_label=(sv.qualifier if sv else "") or "",
                    tolerance_pct=self.rel_tolerance * 100, sentence_index=idx))
                continue
            check = self._verify_value(v, idx, cited_evidence, all_source_values,
                                       values_of, claim_assets)
            verdict.numeric_checks.append(check)

        # ---- safety-domain advisory tag
        if re.search(r"safe|protect|trip|isolat|earth|ground|maintain|risk|danger|"
                     r"withstand|arc[- ]flash", body, re.I):
            verdict.note = (verdict.note + " " if verdict.note else "") + \
                "safety-relevant statement — qualified human review required"

        # ---- lexical support + numeric outcome -> status
        overlap = self._overlap(body, cited_evidence)
        ok_statuses = ("verified", "unit_verified", "verified_within_tolerance")
        checks = verdict.numeric_checks
        if any(c.status == "mismatch" for c in checks):
            verdict.support = "contradicted"
        elif has_values and all(c.status in ok_statuses for c in checks) \
                and overlap >= 0.3:
            verdict.support = "supported"
        elif overlap >= 0.55 and not has_values:
            verdict.support = "supported"          # verbatim quoted text
        elif has_values and all(c.status == "ambiguous" for c in checks if c.status != "insufficient") \
                and any(c.status == "ambiguous" for c in checks):
            verdict.support = "weak"
        elif overlap >= 0.3:
            verdict.support = "weak"
        else:
            verdict.support = "insufficient"
        for c in checks:
            if c.status == "mismatch" and c.source_value is not None:
                verdict.suggested_replacement = (
                    f"Replace {c.answer_value.raw!r} with {c.source_value.raw!r} "
                    f"as stated on page {c.page_number} of {c.document_title}.")
                break
        return verdict

    # ------------------------------------------------------------- helpers
    def _title_of(self, doc_id: int | None) -> str:
        if not doc_id or doc_id < 0:
            return ""
        row = self.db.get_document(doc_id)
        return row["title"] if row else "(deleted document)"

    def _check_citation(self, marker: str, ev: dict, body: str) -> CitationVerdict:
        doc_id = ev["document_id"]
        page = ev.get("matched_page") or ev["page_start"]
        chunk_row = self.db.get_chunk(ev["chunk_id"]) if ev.get("chunk_id") else None
        integrity = "ok"
        note = ""
        if chunk_row is None:
            integrity = "stale_chunk"
            note = "chunk no longer present (document deleted or reindexed)"
        elif chunk_row["text_sha1"] != ev.get("text_sha1"):
            integrity = "stale_chunk"
            note = "chunk text hash changed since retrieval"
        page_row = self.db.get_page(doc_id, page)
        full_page_text = ""
        if page_row is not None:
            # tables live in their own column but belong to the page: a quote
            # taken from a table must be findable on that page
            tabs = (page_row["tables_text"] if "tables_text" in page_row.keys()
                    else "")
            full_page_text = (page_row["text"] + "\n\n" + tabs).strip()
        frag = self._longest_verbatim_fragment(body, ev.get("excerpt") or ev["text"])
        quote_in_chunk = frag is not None
        quote_in_page = False
        if quote_in_chunk and full_page_text:
            quote_in_page = phrase_in_text(frag.matched, full_page_text) is not None
        if not quote_in_chunk and integrity == "ok":
            integrity = "missing_quote" if len(body) > 40 else "ok"
            if integrity == "missing_quote":
                note = "no verbatim fragment of the sentence found in the cited evidence"
        support = "supported" if (quote_in_chunk and quote_in_page) else \
            ("weak" if quote_in_chunk else "insufficient")
        printed = page_row["printed_label"] if page_row else ""
        return CitationVerdict(
            marker=marker, chunk_id=ev.get("chunk_id"), document_id=doc_id,
            document_title=ev.get("title", ""), page_number=page,
            printed_label=printed or "", quote=frag.matched if frag else "",
            integrity=integrity, support=support,
            quote_in_source=bool(quote_in_page or quote_in_chunk),
            source_hash_now=chunk_row["text_sha1"] if chunk_row else "",
            note=note)

    @staticmethod
    def _longest_verbatim_fragment(sentence: str, chunk_text: str,
                                   min_tokens: int = 6):
        """Longest run of sentence words appearing verbatim in the chunk."""
        words = [w for w in sentence.split() if not re.fullmatch(r"\[[^\]]*\]", w)]
        if not words:
            return None
        if len(words) < min_tokens:
            probe = " ".join(words)
            span = phrase_in_text(probe, chunk_text)
            return span if (span and len(probe) >= 12) else None
        for length in range(min(len(words), 40), min_tokens - 1, -1):
            for i in range(0, len(words) - length + 1):
                frag = " ".join(words[i:i + length])
                span = phrase_in_text(frag, chunk_text)
                if span:
                    return span
        return None

    def _verify_value(self, v: EngineeringValue, sidx: int, cited_evidence,
                      all_source_values, values_of,
                      claim_assets: set[str]) -> NumericCheck:
        # 1) values in the quoted excerpt of the cited evidence
        def assets_in(text: str) -> set[str]:
            return {a.upper() for a in _ASSET_RE.findall(text)}
        excerpt_pool: list[tuple[EngineeringValue, dict]] = []
        for ev in cited_evidence:
            if claim_assets:
                ev_assets = assets_in(ev.get("excerpt") or ev["text"]) | \
                    assets_in(ev.get("title", ""))
                if ev_assets and not (claim_assets & ev_assets):
                    continue   # evidence belongs to a different asset
            for sv in values_of(ev, excerpt=True):
                excerpt_pool.append((sv, ev))
        best = self._best_check(v, excerpt_pool, sidx, scope="quoted excerpt") \
            if excerpt_pool else None
        if best is None:
            # unit-family confusion detector (kW vs kVA vs kVAr…): a same-
            # magnitude value in a sibling unit class is NOT a verification —
            # report the ambiguity explicitly instead of a blanket mismatch.
            family = _POWER_FAMILY.get(v.canonical)
            if family:
                sib = [(sv, ev) for sv, ev in excerpt_pool
                       if sv.canonical in family and sv.canonical != v.canonical
                       and sv.value == v.value]
                if sib:
                    sv, ev = sib[0]
                    return NumericCheck(
                        claim=v.raw, status="ambiguous", answer_value=v,
                        source_value=sv, document_id=ev["document_id"],
                        document_title=ev.get("title", ""),
                        page_number=ev.get("matched_page") or ev["page_start"],
                        explanation=(f"{v.raw} cannot be derived from "
                                     f"{sv.raw}: {v.unit_label} is "
                                     f"{'real' if v.canonical == 'active_power' else 'reactive' if v.canonical == 'reactive_power' else 'apparent'} "
                                     "power and the source states a different "
                                     "power family — converting between them "
                                     "requires a power factor, which this "
                                     "claim does not provide."),
                        context_label=sv.qualifier or "",
                        tolerance_pct=self.rel_tolerance * 100,
                        sentence_index=sidx)
        # 2) broader evidence (only to soften 'insufficient', never to verify
        #    over a contradicting excerpt)
        full_pool: list[tuple[EngineeringValue, dict]] = []
        for ev in cited_evidence:
            if claim_assets:
                ev_assets = assets_in(ev.get("excerpt") or ev["text"]) | \
                    assets_in(ev.get("title", ""))
                if ev_assets and not (claim_assets & ev_assets):
                    continue
            for sv in values_of(ev, excerpt=False):
                full_pool.append((sv, ev))
        excerpt_related_categories = any(
            values_comparable_safe(v, sv) for sv, _ in excerpt_pool)
        b2 = None
        b2_src = None
        if (best is None or best.status in ("insufficient", "mismatch")) and full_pool:
            b2, b2_src = self._best_check(v, full_pool, sidx,
                                          scope="document context",
                                          with_source=True)
        for cand, csrc in ((best, best.source_value if best else None),
                           (b2, b2_src)):
            if (cand is not None and cand.status == "mismatch"
                    and csrc is not None and csrc.value == v.value
                    and v.canonical in _POWER_FAMILY
                    and csrc.canonical in _POWER_FAMILY
                    and csrc.canonical != v.canonical):
                cand.status = "ambiguous"
                cand.source_value = csrc
                cand.explanation = (
                    f"{v.raw} numerically equals {csrc.raw} stated in the "
                    f"source, but the units are different power families "
                    f"({v.unit_label} vs {csrc.unit_label}); they are not "
                    "inter-convertible without a power factor — this is not "
                    "verified and not a simple typo.")
                break
        if best is not None and best.status == "mismatch" and v.canonical in _POWER_FAMILY:
            # the closest source value isn't even same-magnitude: still call
            # out the kW/kVA family confusion the claim exhibits
            sib = next((sv for sv, _ in (excerpt_pool + full_pool)
                        if sv.canonical in _POWER_FAMILY
                        and sv.canonical != v.canonical
                        and sv.value == v.value), None)
            if sib is not None:
                best.status = "ambiguous"
                best.source_value = sib
                best.explanation = (
                    f"{v.raw} is not stated in the source; the similarly "
                    f"named quantity there is {sib.raw} — different power "
                    "families (kW/kVA/kVAr) are not inter-convertible without "
                    "a power factor.")
        if best is not None and best.status != "insufficient":
            return best

        if excerpt_related_categories and (best is None or best.status == "insufficient"):
            # excerpt contains comparable values but not this one
            ev0 = cited_evidence[0]
            return NumericCheck(
                claim=v.raw, status="mismatch" if b2 is None else b2.status,
                answer_value=v, source_value=b2.source_value if b2 else None,
                document_id=ev0["document_id"], document_title=ev0.get("title", ""),
                page_number=ev0.get("matched_page") or ev0["page_start"],
                explanation=("The quoted excerpt lists values of this kind, but "
                             f"not {v.raw}; it is not directly supported by the "
                             "cited passage.")
                if b2 is None or b2.status != "verified" else
                ("Value appears elsewhere in the retrieved passage but not in "
                 "the quoted excerpt."),
                context_label=v.qualifier or "",
                tolerance_pct=self.rel_tolerance * 100, sentence_index=sidx)
        if b2 is not None and b2.status != "insufficient":
            return b2
        ev0 = cited_evidence[0] if cited_evidence else None
        return NumericCheck(
            claim=v.raw, status="insufficient", answer_value=v, source_value=None,
            document_id=ev0["document_id"] if ev0 else -1,
            document_title=ev0.get("title", "") if ev0 else "",
            page_number=(ev0.get("matched_page") or ev0["page_start"]) if ev0 else 0,
            explanation="No engineering value with a recognized unit was found "
                        "in the cited passages for this number.",
            context_label=v.qualifier or "",
            tolerance_pct=self.rel_tolerance * 100, sentence_index=sidx)

    def _best_check(self, claim: EngineeringValue, pool, sidx: int, *,
                    scope: str, with_source: bool = False):
        rank = {"verified": 0, "unit_verified": 1, "verified_within_tolerance": 2,
                "ambiguous": 3, "mismatch": 4}
        # ambiguity that stems from unit-family confusion (kW quoted against a
        # kVA rating) is more informative than a blanket mismatch
        if claim.canonical in ("active_power", "apparent_power", "reactive_power"):
            rank["ambiguous"] = 0
        scored: list[tuple[int, int, NumericCheck]] = []
        for sv, ev in pool:
            if not values_comparable_safe(claim, sv):
                continue
            comp = compare_values(claim, sv, self.rel_tolerance)
            ctx_bonus = 1 if (sv.qualifier == claim.qualifier
                              and sv.variant == claim.variant) else 0
            if sv.variant != claim.variant:
                ctx_bonus -= 1          # number sits in another row/variant
            if _context_aligned(claim.sentence or "", sv.sentence or ""):
                ctx_bonus += 1
            if ev.get("weak"):
                ctx_bonus -= 1
            expl = comp.explanation
            if scope == "document context":
                expl += " (value found in the retrieved passage, not in the " \
                        "quoted excerpt — treat with extra care)"
            scored.append((rank.get(comp.status, 5), -ctx_bonus, NumericCheck(
                claim=claim.raw, status=comp.status, answer_value=claim,
                source_value=sv, document_id=ev["document_id"],
                document_title=ev.get("title", ""),
                page_number=ev.get("matched_page") or ev["page_start"],
                explanation=expl, context_label=sv.qualifier or "",
                tolerance_pct=self.rel_tolerance * 100, sentence_index=sidx)))
        if not scored:
            return (None, None) if with_source else None
        scored.sort(key=lambda t: (t[0], t[1]))
        chk = scored[0][2]
        if with_source:
            return chk, chk.source_value
        return chk

    @staticmethod
    def _verbatim_source(v: EngineeringValue, evidence: list[dict],
                         claim_sentence: str, values_of=None):
        """Look for the claim value quoted verbatim in a cited excerpt.

        Returns ``None`` (nothing quoted), ``(evidence, value)`` for a clean
        verbatim match, ``("scope", ...)`` when the value is quoted but the
        source presents it under a different engineering scope, or
        ``("row-conflict", ...)`` when the value demonstrably belongs to
        another row of the same table.  A clean match anywhere wins over a
        conflict elsewhere — the digest can quote several passages at once and
        a value verified in its own context must not be vetoed by an
        unrelated table further down.
        """
        num = str(v.value) if v.value % 1 == 0 else str(v.value)
        claim_tags = set(v.row_tags or ())
        conflict = None

        def vals_for(e: dict) -> list[EngineeringValue]:
            # cached by the caller: re-extracting every passage for every
            # candidate value made verification quadratic
            if values_of is None:
                return extract_values(e.get("excerpt") or e["text"])
            return (values_of(e, excerpt=True) if e.get("excerpt")
                    else values_of(e))
        for e in evidence:
            hay = " ".join((e.get("excerpt") or e["text"]).split())
            if f"{num} {v.unit_label}" not in hay and f"{num}{v.unit_label}" not in hay:
                continue
            hay_raw = e.get("excerpt") or e["text"]
            for sv in vals_for(e):
                if str(sv.value) != num or sv.unit_label != v.unit_label \
                        or sv.canonical != v.canonical:
                    continue
                tags = set(sv.row_tags or ())
                if claim_tags and tags and not (claim_tags & tags):
                    # value sits in a differently-labelled row of the same
                    # table: remember as a conflict unless another passage
                    # verifies the claim cleanly
                    other = next((x for x in extract_values(hay_raw)
                                  if x.canonical == v.canonical and tags
                                  and claim_tags & set(x.row_tags or ())), None)
                    if other is not None and conflict is None:
                        conflict = ("row-conflict", (e, other))
                    continue
                if sv.sentence and not _context_aligned(claim_sentence, sv.sentence,
                                                        min_overlap=0.34):
                    continue
                # An answer that explicitly names a scope ("rated", "maximum",
                # "typical"…) is only confirmed by a source line naming the same
                # scope.  An *unqualified* value inside a quotation is not a
                # scope assertion, so the source's own label cannot contradict
                # it (the heading of a table chunk says "Rated voltage | 11 kV";
                # quoting that row is not a claim about which row it is).
                strict_scope = bool(v.qualifier) and v.qualifier != sv.qualifier
                if _scope_conflict(sv, v) or strict_scope:
                    # the answer states an engineering scope ("rated", "max",
                    # "typical"…): only a source line using the same scope word
                    # may confirm it verbatim.  An unqualified occurrence of the
                    # same number is NOT permission to adopt the label.
                    if conflict is None:
                        conflict = ("scope", (e, sv))
                    continue
                return e, sv
        return conflict


    @staticmethod
    def _overlap(sentence: str, evidence: list[dict]) -> float:
        stop = {"the", "and", "for", "with", "shall", "must", "should", "per",
                "this", "that", "from", "they", "their", "which", "when"}
        s_tok = {t.lower() for t in re.findall(r"[A-Za-z]{3,}|\d+(?:\.\d+)?", sentence)
                 if t.lower() not in stop}
        if not s_tok:
            return 1.0
        ev_tok: set[str] = set()
        for e in evidence:
            ev_tok |= {t.lower() for t in
                       re.findall(r"[A-Za-z]{3,}|\d+(?:\.\d+)?",
                                  e.get("excerpt") or e["text"])
                       if t.lower() not in stop}
        return len(s_tok & ev_tok) / len(s_tok)


_STOPWORDS = set("""the a an of to in is are for and or with on at as by from this
that it its shall must should can be been what which how many does do per by""".split())


_QUOTE_SPANS = re.compile(r"\u201c([^\u201d]{6,})\u201d")


NAV_SECTION_HEADING = "**Most directly matching passages**"

def _heading_span(text: str) -> int:
    """Length of the generated digest heading at the start of ``text`` (0 if none).

    The extractive answer labels each block as ``N. <document title>, page P —
    …`` before the quotation.  Numbers inside that label belong to a document
    *name* ("DEMO 11 kV switchgear handbook"), they are not claims the
    assistant is making, so they must not be checked as values.
    """
    if not re.match(r"\s*\d+[.)]\s", text):
        return 0
    q = text.find("\u201c")
    if q <= 0:
        return 0
    return q if "\u2014" in text[:q] else 0


def quoted_spans(text: str) -> list[tuple[int, int, str]]:
    """Character spans of “verbatim quote” regions in an answer sentence."""
    return [(m.start(), m.end(), m.group(1)) for m in _QUOTE_SPANS.finditer(text)]


def _quote_segments(quote: str) -> tuple[list[str], bool]:
    """Split a displayed quotation into the pieces that must exist verbatim.

    The answer marks omitted text with … (an *ellipsis is a display marker,
    never source content*), so each side of it is checked separately.  Returns
    the segments and whether the quotation was clipped at either end.
    """
    q = quote.strip()
    clipped = q.startswith("…") or q.endswith("…") or q.startswith("...")
    q = q.strip("…").strip().strip('"').strip()
    parts = [x.strip() for x in re.split(r"…|\[\u2026\]|\.{3}", q)]
    # collapse runs of whitespace: PDF text wraps mid-sentence and the answer
    # renders newlines as spaces, so "word\nword" and "word word" must match
    segs = [" ".join(x.split()) for x in parts if len(" ".join(x.split())) >= 12]
    if not segs and q:
        segs = [" ".join(q.split())]
    return segs, clipped


def _quote_in_evidence(quote: str, evidence: list[dict]) -> tuple[dict, str] | None:
    """Is this quoted string really present in one of the passages?

    Returns ``(evidence, how)`` with ``how`` in ``"verbatim"`` (every segment
    found) or ``"prefix"`` (only the visible opening words could be matched —
    the quotation was clipped in the answer).  Matching is whitespace
    insensitive because PDF extraction wraps lines arbitrarily, but the
    characters must be identical: lexical similarity is deliberately NOT
    accepted as proof that a quote exists.
    """
    from .textmatch import phrase_in_text
    segs, clipped = _quote_segments(quote)
    if not segs:
        return None
    longest = max(segs, key=len)[:260]      # already whitespace-normalised
    for ev in evidence:
        flats = []
        for hay in (ev.get("text"), ev.get("excerpt")):
            if hay:
                flats.append(" ".join(hay.split()))
        if not flats:
            continue
        if any(all(seg in flat for seg in segs) for flat in flats):
            return ev, "verbatim"
        if any(phrase_in_text(segs[0], flat) is not None for flat in flats):
            return ev, "verbatim"
        if clipped and any(longest in flat for flat in flats):
            return ev, "prefix"
    return None


def _context_aligned(claim_sentence: str, source_sentence: str,
                     min_overlap: float = 0.5) -> bool:
    """True when the claim's descriptive wording is covered by the source
    sentence it is being matched against.

    A quoted number only validates a claim when the claim talks about the
    same quantity: the distinctive words of the claim must appear in the
    source line (e.g. claim "rated system voltage … 145.5 kV" must NOT be
    validated by a different row that merely happens to contain 145.5 kV).
    """
    ct = {t.lower() for t in re.findall(r"[A-Za-z0-9°²%/\.\-]{3,}", claim_sentence)
          if t.lower() not in _STOPWORDS}
    st_ = {t.lower() for t in re.findall(r"[A-Za-z0-9°²%/\.\-]{3,}", source_sentence)
           if t.lower() not in _STOPWORDS}
    if not ct:
        return True
    ct -= {"is", "are"}
    inter = len(ct & st_)
    coverage = inter / max(1, len(ct))
    if len(st_) >= 5 and st_ <= ct:
        return True            # source row fully quoted inside the claim
    return coverage >= max(min_overlap, 0.4)


_POWER_FAMILY = {
    "active_power": {"active_power", "apparent_power", "reactive_power"},
    "apparent_power": {"active_power", "apparent_power", "reactive_power"},
    "reactive_power": {"active_power", "apparent_power", "reactive_power"},
}


def _scope_conflict(a: EngineeringValue, b: EngineeringValue) -> bool:
    equiv = {None: "rated", "rated": "rated"}
    return equiv.get(a.qualifier, a.qualifier) != equiv.get(b.qualifier, b.qualifier)


def values_comparable_safe(a: EngineeringValue, b: EngineeringValue) -> bool:
    """Same category (or temperature pair) — used to keep numeric checks from
    crossing physical meanings. Power categories (W/VA/var) stay separate."""
    if a.canonical == b.canonical:
        return True
    if {a.canonical, b.canonical} == {"temperature_c", "temperature_k"}:
        return True
    return False


def _subject_assets(v: EngineeringValue, e: dict) -> set[str]:
    """Equipment identifiers a value is explicitly attached to.

    A value counts as attached to an asset when the asset token appears in the
    value's own row/sentence, or in the *title of the document* the value comes
    from ("DEMO-TR-01 transformer maintenance manual" → every value in it is
    about DEMO-TR-01).  Nothing here is inferred: an absent token means "unknown
    subject", never "the same subject as the other passage".
    """
    context = [v.sentence or "", " ".join(v.row_tags or ()), e.get("title", "")]
    return {a.upper() for t in context for a in _ASSET_RE.findall(t)}


def find_conflicts(evidence: list[dict], *, rel_tol: float = 1e-9) -> list[dict]:
    """Detect two evidence passages stating different values for the same
    qualified quantity (planted contradictions, revision mismatches).

    Deliberately *very* conservative — a wrong "the sources disagree" report is
    as harmful as a missed one, because it invites a human to distrust good
    data.  A pair is reported only when all of the following hold:

    * different documents (cross-document disagreement);
    * identical canonical category (``kW``/``kVA``/``kvar`` never mix) **and**
      the same explicit qualifier (``rated``/``maximum``/``typical``/…);
    * the values genuinely differ, and their model/variant tags do not;
    * the two values are positively attributed to the *same* equipment: both
      sit under an asset token and those token sets intersect.

    The last rule is what stops a competent-looking but bogus report: a TR-01
    nameplate and an unrelated datasheet appendix both contain "rated apparent
    power", yet they are different machines and must never be called a
    contradiction.  When the subject cannot be established for both values, the
    pair is dropped silently rather than guessed at.
    """
    items: list[tuple[EngineeringValue, dict, set[str]]] = []
    for e in evidence:
        for v in extract_values(e["text"]):
            items.append((v, e, _subject_assets(v, e)))
    conflicts: list[dict] = []
    seen: set[tuple] = set()
    for i, (a, ea, subj_a) in enumerate(items):
        for b, eb, subj_b in items[i + 1:]:
            if a.canonical != b.canonical:
                continue
            if ea["document_id"] == eb["document_id"]:
                continue
            if not a.qualifier or a.qualifier != b.qualifier:
                continue
            if a.variant and b.variant and a.variant != b.variant:
                continue
            if not (subj_a and subj_b and (subj_a & subj_b)):
                continue
            comp = compare_values(a, b, rel_tol)
            if comp.status == "mismatch":
                key = tuple(sorted([(a.raw, ea["document_id"], ea["page_start"]),
                                    (b.raw, eb["document_id"], eb["page_start"])]))
                if key in seen:
                    continue
                seen.add(key)
                conflicts.append({
                    "category": a.category_label,
                    "qualifier": a.qualifier,
                    "values": [
                        {"raw": a.raw, "document_id": ea["document_id"],
                         "document_title": ea.get("title", ""),
                         "page": ea.get("matched_page") or ea["page_start"]},
                        {"raw": b.raw, "document_id": eb["document_id"],
                         "document_title": eb.get("title", ""),
                         "page": eb.get("matched_page") or eb["page_start"]}],
                    "explanation": ("Sources disagree: "
                                    f"'{a.raw}' ({ea.get('title','')}) vs "
                                    f"'{b.raw}' ({eb.get('title','')}). "
                                    "Both readings are quoted; a human should "
                                    "check the revision status of each document."),
                })
    return conflicts


def strip_citation_markers(text: str) -> str:
    return re.sub(r"\[[^\]]{1,40}\]", "", text).replace("  ", " ").strip()
