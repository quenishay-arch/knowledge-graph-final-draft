"""Priority-ranked grade classification with dynamic confidence scoring.

Priority (highest first):
  0  explicit grade markers in document text (Primary 5, Form 1 — not unit numbers, not series ranges)
  1  filename metadata (P4.pdf, p3b.pdf — strict patterns only)
  2  curriculum vocabulary / grammar topic difficulty
  3  lexical complexity
  4  contextual metadata (publisher notes — never series ranges like Primary 1-6)

No hardcoded grade defaults. Unknown when evidence is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from curriculum_os.engine.canonical import parse_canonical_document

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}

GRADE_ORDER = [f"P{i}" for i in range(1, 7)] + [f"S{i}" for i in range(1, 7)]

# Topic hints by curriculum band — specific grammar/vocabulary, not generic skills like "reading".
GRAMMAR_TOPIC_BANDS: dict[str, tuple[str, ...]] = {
    "P1": ("cvc words", "letter sounds", "five senses", "my family topic"),
    "P2": (
        "consonant blend",
        "short vowel",
        "toys",
        "colours",
        "colors",
        "clothes",
        "values education",
        "riddles",
    ),
    "P3": (
        "long vowel",
        "digraph",
        "simple past tense",
        "prepositions of place",
        "personal letter",
    ),
    "P4": ("comparative adjective", "superlative", "book report", "healthy living"),
    "P5": (
        "passive voice",
        "present perfect",
        "conditional sentence",
        "persuasive writing",
        "for and since",
        "ever and never",
    ),
    "P6": ("reported speech", "relative clause", "fact and opinion"),
    "S1": (
        "demonstrative pronoun",
        "connectives",
        "prepositions of time",
        "subject-verb agreement",
        "information report",
        "social media post",
    ),
    "S2": ("past continuous", "inferencing", "feature article"),
    "S3": ("indefinite pronoun", "bare infinitive", "noun phrase", "film review"),
    "S4": ("hkdse", "rhetorical device", "register analysis"),
    "S5": ("writer's intention", "letter to the editor", "social issues"),
    "S6": ("mock examination", "critical study", "language arts"),
}

# Advanced grammar that contradicts lower-primary classification.
ADVANCED_GRADE_MARKERS: dict[str, tuple[str, ...]] = {
    "P4": ("comparative", "superlative", "book report"),
    "P5": ("passive voice", "present perfect", "conditional"),
    "P6": ("reported speech", "relative clause"),
    "S1": ("demonstrative", "connectives", "prepositions of time"),
}

UNIT_CONTEXT = re.compile(
    r"\b(?:unit|chapter|module|u)\s*\d",
    re.IGNORECASE,
)
GRADE_RANGE = re.compile(
    r"\b(?:primary|secondary|form)\s*(\d)\s*[–\-]\s*(\d)\b",
    re.IGNORECASE,
)
SERIES_RANGE = re.compile(
    r"\bprimary\s*(\d)\s*[–\-]\s*(\d)\b|\bprimary\s*(\d)\s*to\s*(\d)\b",
    re.IGNORECASE,
)

EXPLICIT_CONTENT_PATTERNS = [
    (r"\bprimary\s+(?:one|two|three|four|five|six)\b", "primary_word"),
    (r"\bsecondary\s+(?:one|two|three|four|five|six)\b", "secondary_word"),
    (r"\bform\s+(?:one|two|three|four|five|six)\b", "form_word"),
    (r"\bprimary\s*([1-6])\s*([ab])\b", "primary_form"),
    (r"\bsecondary\s*([1-6])\s*([ab])\b", "secondary_form"),
    (r"\bform\s*([1-6])\b", "form_num"),
    (r"\bprimary\s*([1-6])\b(?!\s*[–\-]\s*\d)", "primary_num"),
    (r"\bsecondary\s*([1-6])\b(?!\s*[–\-]\s*\d)", "secondary_num"),
]

MIN_CONFIDENCE_TO_PREDICT = 0.38


@dataclass
class GradeSignal:
    source: str
    grade: str
    confidence: float
    priority: int
    evidence: list[str]

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class GradeDecision:
    grade: str
    confidence: float
    source: str
    signals: list[GradeSignal]
    warnings: list[str]
    needs_human_review: bool = False
    confidence_factors: dict = None

    def model_dump(self) -> dict:
        return {
            "grade": self.grade,
            "confidence": self.confidence,
            "source": self.source,
            "signals": [s.model_dump() for s in self.signals],
            "warnings": self.warnings,
            "needs_human_review": self.needs_human_review,
            "confidence_factors": self.confidence_factors or {},
        }


def normalize_grade(value: str, *, default_school_level: str = "primary") -> str:
    text = (value or "").strip()
    if not text:
        return "Unknown"

    if _is_series_range_text(text):
        return "Unknown"

    lower = text.lower()
    compact = re.sub(r"\s+", "", text).upper()
    if re.fullmatch(r"[PS][1-6]", compact):
        return compact

    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\bprimary\s+{word}\b", lower):
            return f"P{number}"
        if re.search(rf"\bsecondary\s+{word}\b", lower):
            return f"S{number}"
        if re.search(rf"\bform\s+{word}\b", lower):
            return f"S{number}"

    m = re.search(r"\bprimary\s*([1-6])\s*([ab])\b", lower)
    if m and not _digit_in_range_context(text, m.start()):
        return f"P{m.group(1)}"
    m = re.search(r"\bsecondary\s*([1-6])\s*([ab])\b", lower)
    if m and not _digit_in_range_context(text, m.start()):
        return f"S{m.group(1)}"
    m = re.search(r"\bform\s*([1-6])\b", lower)
    if m and not _digit_in_range_context(text, m.start()):
        return f"S{m.group(1)}"
    m = re.search(r"\bprimary\s*([1-6])\b", lower)
    if m and not _digit_in_range_context(text, m.start()):
        return f"P{m.group(1)}"
    m = re.search(r"\bsecondary\s*([1-6])\b", lower)
    if m and not _digit_in_range_context(text, m.start()):
        return f"S{m.group(1)}"

    m = re.search(r"\bbook\s*([1-6])\s*([ab])\b", lower)
    if m and ("primary" in lower or "secondary" in lower or default_school_level != "unknown"):
        prefix = "S" if "secondary" in lower or default_school_level == "secondary" else "P"
        return f"{prefix}{m.group(1)}"

    return "Unknown"


def _is_series_range_text(text: str) -> bool:
    return bool(SERIES_RANGE.search(text or ""))


def _digit_in_range_context(text: str, match_start: int) -> bool:
    window = text[max(0, match_start - 10) : match_start + 30]
    return bool(GRADE_RANGE.search(window) or SERIES_RANGE.search(window))


def _is_unit_context(text: str, match_start: int, match_end: int) -> bool:
    window = text[max(0, match_start - 20) : match_end + 20]
    return bool(UNIT_CONTEXT.search(window))


def _is_valid_explicit_match(text: str, match: re.Match) -> bool:
    snippet = match.group(0)
    if _is_series_range_text(text[max(0, match.start() - 15) : match.end() + 15]):
        return False
    if _is_unit_context(text, match.start(), match.end()):
        return False
    if re.search(r"\bunits?\s+\d", text[max(0, match.start() - 5) : match.end() + 10], re.I):
        return False
    return True


def explicit_grade_signals(text: str, *, source: str, priority: int) -> list[GradeSignal]:
    """Strict explicit grade anchors — excludes unit numbers and series ranges."""
    if not text:
        return []
    signals: list[GradeSignal] = []
    for pattern, _kind in EXPLICIT_CONTENT_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            if not _is_valid_explicit_match(text, match):
                continue
            grade = normalize_grade(match.group(0))
            if grade == "Unknown":
                continue
            signals.append(
                GradeSignal(
                    source=source,
                    grade=grade,
                    confidence=0.94,
                    priority=priority,
                    evidence=[match.group(0).strip()],
                )
            )
    return _dedupe_signals(signals)


def filename_grade_signal(pdf_path: str | Path) -> list[GradeSignal]:
    """Filename metadata — only strict P/S/NTP stem patterns, not ambiguous tokens."""
    parsed = parse_canonical_document(pdf_path)
    signals: list[GradeSignal] = []

    if parsed.publisher == "NTP" and parsed.inferred_grade:
        return [
            GradeSignal(
                source="filename_ntp",
                grade=parsed.inferred_grade,
                confidence=0.93,
                priority=1,
                evidence=[parsed.raw_filename],
            )
        ]

    if parsed.publisher in ("P", "S") and parsed.inferred_grade:
        return [
            GradeSignal(
                source="filename",
                grade=parsed.inferred_grade,
                confidence=0.88,
                priority=1,
                evidence=[parsed.raw_filename],
            )
        ]

    stem = Path(pdf_path).stem
    m = re.search(r"\b(?:primary|secondary|form)\s*[1-6]\b", stem.replace("_", " "), re.I)
    if m and not _is_series_range_text(m.group(0)):
        grade = normalize_grade(m.group(0))
        if grade != "Unknown":
            signals.append(
                GradeSignal(
                    source="filename",
                    grade=grade,
                    confidence=0.82,
                    priority=1,
                    evidence=[stem],
                )
            )
    return signals


def curriculum_topic_signals(extractions: list[dict], *, raw_text: str = "") -> list[GradeSignal]:
    """Semantic classification from extracted grammar/vocabulary only — not boilerplate text."""
    text_parts: list[str] = []
    for item in extractions:
        text_parts.extend(str(v) for v in item.get("grammar_points", []))
        text_parts.extend(str(v) for v in item.get("vocabulary_themes", []))
        text_parts.append(str(item.get("unit_title", "")))
    text = " ".join(text_parts).lower()
    if not text.strip():
        return []

    scores: dict[str, float] = {}
    evidence_map: dict[str, list[str]] = {}
    for grade, hints in GRAMMAR_TOPIC_BANDS.items():
        hits = [h for h in hints if h in text]
        if hits:
            scores[grade] = sum(1.0 + (len(h) / 40.0) for h in hits)
            evidence_map[grade] = hits[:5]

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_grade, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    if margin < 0.5 and len(ranked) > 1:
        return []

    confidence = min(0.82, 0.42 + 0.08 * best_score + 0.06 * margin)
    return [
        GradeSignal(
            source="curriculum_topics",
            grade=best_grade,
            confidence=confidence,
            priority=2,
            evidence=evidence_map[best_grade],
        )
    ]


def lexical_complexity_signal(text: str) -> GradeSignal | None:
    tokens = re.findall(r"[A-Za-z']+", text or "")
    if len(tokens) < 120:
        return None
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if len(re.findall(r"[A-Za-z']+", s)) >= 3]
    avg_sentence = len(tokens) / max(1, len(sentences))
    long_word_ratio = sum(1 for t in tokens if len(t) >= 8) / len(tokens)
    score = avg_sentence * 0.055 + long_word_ratio * 3.8
    if score < 0.95:
        grade = "P2"
    elif score < 1.25:
        grade = "P4"
    elif score < 1.55:
        grade = "P6"
    elif score < 1.9:
        grade = "S2"
    else:
        grade = "S4"
    return GradeSignal(
        source="lexical_complexity",
        grade=grade,
        confidence=0.38,
        priority=3,
        evidence=[
            f"avg_sentence_words={avg_sentence:.1f}",
            f"long_word_ratio={long_word_ratio:.2f}",
        ],
    )


def metadata_grade_signal(
    explicit_label: str,
    *,
    school_level: str = "unknown",
    evidence: list[str] | None = None,
) -> GradeSignal | None:
    label = (explicit_label or "").strip()
    if not label or _is_series_range_text(label):
        return None
    grade = normalize_grade(label, default_school_level=school_level)
    if grade == "Unknown":
        return None
    return GradeSignal(
        source="metadata",
        grade=grade,
        confidence=0.72,
        priority=4,
        evidence=(evidence or [label])[:5],
    )


def _grade_index(grade: str) -> int:
    try:
        return GRADE_ORDER.index(grade)
    except ValueError:
        return -1


def _grade_distance(a: str, b: str) -> int:
    ia, ib = _grade_index(a), _grade_index(b)
    if ia < 0 or ib < 0:
        return 0
    return abs(ia - ib)


def _signals_for_grade(signals: list[GradeSignal], grade: str) -> list[GradeSignal]:
    return [s for s in signals if s.grade == grade]


def _validator_grade(signals: list[GradeSignal]) -> str | None:
    validators = [s for s in signals if s.priority >= 2]
    if not validators:
        return None
    scores: dict[str, float] = {}
    for s in validators:
        scores[s.grade] = scores.get(s.grade, 0.0) + s.confidence
    return max(scores, key=scores.get) if scores else None


def _advanced_topics_detected(signals: list[GradeSignal], min_grade: str = "P4") -> bool:
    text = " ".join(" ".join(s.evidence) for s in signals if s.priority >= 2).lower()
    min_idx = _grade_index(min_grade)
    for grade, hints in ADVANCED_GRADE_MARKERS.items():
        if _grade_index(grade) < min_idx:
            continue
        if any(h in text for h in hints):
            return True
    return False


def _resolve_priority_zero_conflict(tier: list[GradeSignal], all_signals: list[GradeSignal]) -> GradeSignal:
    """When multiple explicit markers disagree, prefer the one corroborated by content."""
    validator = _validator_grade(all_signals)
    if validator:
        corroborated = [s for s in tier if s.grade == validator]
        if corroborated:
            return max(corroborated, key=lambda s: s.confidence)
    return max(tier, key=lambda s: (s.confidence, -_grade_index(s.grade)))


def _score_confidence(
    winner: GradeSignal,
    all_signals: list[GradeSignal],
    priority: int,
) -> tuple[float, list[str], dict, bool]:
    warnings: list[str] = []
    factors: dict = {"priority": priority, "source": winner.source}

    validator = _validator_grade(all_signals)
    filename = [s for s in all_signals if s.priority == 1 and s.source in ("filename", "filename_ntp")]
    explicit = [s for s in all_signals if s.priority == 0]

    base = winner.confidence
    if priority == 0:
        base = 0.94
        factors["basis"] = "explicit_marker"
    elif priority == 1:
        base = 0.86
        factors["basis"] = "filename_metadata"

    if validator and validator == winner.grade:
        base = min(0.98, base + 0.06)
        factors["content_agreement"] = True
    elif validator and validator != winner.grade:
        dist = _grade_distance(winner.grade, validator)
        factors["content_agreement"] = False
        factors["validator_grade"] = validator
        if dist >= 2:
            base -= 0.22
            warnings.append(
                f"Primary signal {winner.grade} ({winner.source}) strongly disagrees "
                f"with content difficulty ({validator}, distance={dist})."
            )
        else:
            base -= 0.12
            warnings.append(
                f"Primary signal {winner.grade} ({winner.source}) disagrees with "
                f"semantic content ({validator})."
            )

    if filename and priority == 1 and validator == winner.grade:
        base = min(0.96, base + 0.08)
        factors["filename_content_match"] = True
    elif filename and priority == 1 and validator and validator != winner.grade:
        factors["filename_content_match"] = False

    if priority >= 2 and not explicit and not filename:
        base = min(base, 0.68)
        factors["content_only"] = True
        warnings.append("Grade inferred from content only; no explicit marker or clear filename.")

    if winner.grade in ("P1", "P2") and _advanced_topics_detected(all_signals, min_grade="P4"):
        base -= 0.25
        warnings.append(
            f"Grade {winner.grade} conflicts with advanced grammar topics in content "
            f"(e.g. passive voice, present perfect)."
        )

    if len({s.grade for s in explicit}) > 1:
        warnings.append(
            f"Multiple explicit grade markers found: {', '.join(sorted({s.grade for s in explicit}))}."
        )

    confidence = round(max(0.0, min(0.98, base)), 3)
    needs_review = confidence < 0.72 or bool(warnings)
    factors["final_confidence"] = confidence
    return confidence, warnings, factors, needs_review


def choose_grade(signals: list[GradeSignal]) -> GradeDecision:
    """Priority cascade with dynamic confidence from cross-validation."""
    signals = _dedupe_signals(signals)
    if not signals:
        return GradeDecision(
            "UNKNOWN",
            0.0,
            "none",
            [],
            ["No grade signals found."],
            True,
            {"basis": "none"},
        )

    by_priority: dict[int, list[GradeSignal]] = {}
    for signal in signals:
        by_priority.setdefault(signal.priority, []).append(signal)

    for priority in sorted(by_priority):
        tier = by_priority[priority]
        if priority == 0 and len({s.grade for s in tier}) > 1:
            winner = _resolve_priority_zero_conflict(tier, signals)
        else:
            winner = max(tier, key=lambda s: s.confidence)

        if priority <= 1:
            confidence, warnings, factors, needs_review = _score_confidence(winner, signals, priority)
            if confidence < MIN_CONFIDENCE_TO_PREDICT:
                break
            return GradeDecision(
                winner.grade,
                confidence,
                winner.source,
                signals,
                warnings,
                needs_review,
                factors,
            )

    validators = [s for s in signals if s.priority >= 2]
    if validators:
        winner = max(validators, key=lambda s: s.confidence)
        confidence, warnings, factors, needs_review = _score_confidence(winner, signals, winner.priority)
        if confidence >= MIN_CONFIDENCE_TO_PREDICT:
            return GradeDecision(
                winner.grade,
                confidence,
                winner.source,
                signals,
                warnings,
                True,
                factors,
            )

    return GradeDecision(
        "UNKNOWN",
        0.0,
        "none",
        signals,
        ["Insufficient corroborated grade evidence; returning Unknown instead of guessing."],
        True,
        {"basis": "insufficient_evidence"},
    )


def _dedupe_signals(signals: list[GradeSignal]) -> list[GradeSignal]:
    best: dict[tuple[str, str, str], GradeSignal] = {}
    for signal in signals:
        key = (signal.source, signal.grade, "|".join(signal.evidence))
        if key not in best or signal.confidence > best[key].confidence:
            best[key] = signal
    return list(best.values())
