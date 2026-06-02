from typing import List, Literal

from pydantic import BaseModel, Field


class CurriculumExtraction(BaseModel):
    """Structured unit extraction from a textbook chunk."""

    unit_title: str = ""
    grammar_points: List[str] = Field(default_factory=list)
    vocabulary_themes: List[str] = Field(default_factory=list)
    # Optional for now — keep empty unless the chunk explicitly lists words.
    vocabulary_words: List[str] = Field(default_factory=list)
    language_skills: List[str] = Field(default_factory=list)


class DocumentTypeClassification(BaseModel):
    document_type: Literal[
        "full_textbook",
        "workbook",
        "single_unit",
        "worksheet",
        "assessment_paper",
        "toc_only",
        "unknown",
    ] = "unknown"
    evidence: List[str] = Field(default_factory=list)


class GlobalMetadata(BaseModel):
    explicit_grade_label: str = ""
    publisher: str = ""
    textbook_series: str = ""
    source_code: str = ""
    semester_or_book: str = ""
    unit_number: str = ""
    filename_grade_hint: str = ""
    filename_unit_hint: str = ""
    school_level: Literal["primary", "secondary", "unknown"] = "unknown"
    units_explicitly_listed: bool = False
    evidence: List[str] = Field(default_factory=list)


class UnitBoundary(BaseModel):
    unit_title: str
    start_page: int = 1
    end_page: int = 1
    evidence: List[str] = Field(default_factory=list)


class UnitBoundaryPlan(BaseModel):
    units: List[UnitBoundary] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)


class LevelEvidence(BaseModel):
    evidence: str
    suggested_level: str
    confidence: float = Field(ge=0.0, le=1.0)


class GradeInference(BaseModel):
    predicted_grade: str = "Unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class DocumentAnalysis(BaseModel):
    document_type: DocumentTypeClassification
    metadata: GlobalMetadata
    claimed_level: str = "Unknown"
    inferred_level: str = "Unknown"
    level_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    level_evidence: List[LevelEvidence] = Field(default_factory=list)
    consistency_issues: List[str] = Field(default_factory=list)
    level_decision_trace: List[str] = Field(default_factory=list)
    total_units_detected: int = 0
    document_unit_number: str = ""
