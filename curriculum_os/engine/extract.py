import os
import re
from pathlib import Path

import instructor
from openai import OpenAI

from curriculum_os.engine.config import DEPLOYMENT
from curriculum_os.engine.models import (
    CurriculumExtraction,
    DocumentTypeClassification,
    GradeInference,
    GlobalMetadata,
    UnitBoundaryPlan,
)
from curriculum_os.engine.prompts import (
    CURRICULUM_EXTRACTION_PROMPT,
    DOCUMENT_TYPE_PROMPT,
    GRADE_INFERENCE_PROMPT,
    GLOBAL_METADATA_PROMPT,
    VISUAL_UNIT_BOUNDARY_PROMPT,
)
from curriculum_os.engine.validate import validate_schema


def _azure_openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    if "/openai/deployments/" in endpoint:
        endpoint = endpoint.split("/openai/deployments/")[0].rstrip("/")
    if endpoint.endswith("/openai/v1"):
        return endpoint + "/"
    if endpoint.endswith("/openai"):
        return endpoint + "/v1/"
    return endpoint + "/openai/v1/"


def get_llm_client() -> instructor.Instructor:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        raise ValueError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY"
        )
    return instructor.from_openai(
        OpenAI(api_key=key, base_url=_azure_openai_base_url(endpoint))
    )


def extract_entities(
    client: instructor.Instructor,
    chunk: dict,
    *,
    model: str | None = None,
) -> CurriculumExtraction:
    """Run LLM extraction on one chunk and return validated entities."""
    deployment = model or DEPLOYMENT
    user_content = chunk.get("content", "")
    if chunk.get("unit_title"):
        user_content = f"Unit: {chunk['unit_title']}\n\n{user_content}"

    response = client.chat.completions.create(
        model=deployment,
        response_model=CurriculumExtraction,
        messages=[
            {"role": "system", "content": CURRICULUM_EXTRACTION_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    data = response.model_dump()
    if not data.get("unit_title") and chunk.get("unit_title"):
        data["unit_title"] = chunk["unit_title"]
    return validate_schema(data)


def classify_document_type(
    client: instructor.Instructor,
    text: str,
    *,
    model: str | None = None,
) -> DocumentTypeClassification:
    deployment = model or DEPLOYMENT
    return client.chat.completions.create(
        model=deployment,
        response_model=DocumentTypeClassification,
        messages=[
            {"role": "system", "content": DOCUMENT_TYPE_PROMPT},
            {"role": "user", "content": text[:12000]},
        ],
    )


def extract_global_metadata(
    client: instructor.Instructor,
    text: str,
    *,
    model: str | None = None,
) -> GlobalMetadata:
    deployment = model or DEPLOYMENT
    response = client.chat.completions.create(
        model=deployment,
        response_model=GlobalMetadata,
        messages=[
            {"role": "system", "content": GLOBAL_METADATA_PROMPT},
            {"role": "user", "content": text[:14000]},
        ],
    )
    return response


def infer_grade_from_content(
    client: instructor.Instructor,
    text: str,
    *,
    model: str | None = None,
) -> GradeInference:
    deployment = model or DEPLOYMENT
    return client.chat.completions.create(
        model=deployment,
        response_model=GradeInference,
        messages=[
            {"role": "system", "content": GRADE_INFERENCE_PROMPT},
            {"role": "user", "content": text[:18000]},
        ],
    )


def infer_grade_from_images(
    client: instructor.Instructor,
    image_data_urls: list[str],
    *,
    text_hint: str = "",
    model: str | None = None,
) -> GradeInference:
    deployment = model or DEPLOYMENT
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Infer the school grade from these rendered PDF pages. "
                "Use visible cover labels, headers, contents pages, and task complexity. "
                "Do not use filename. Text extracted from the PDF, if any:\n"
                f"{text_hint[:3000]}"
            ),
        }
    ]
    for url in image_data_urls[:6]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    return client.chat.completions.create(
        model=deployment,
        response_model=GradeInference,
        messages=[
            {"role": "system", "content": GRADE_INFERENCE_PROMPT},
            {"role": "user", "content": content},
        ],
    )


def detect_unit_boundaries_from_images(
    client: instructor.Instructor,
    image_data_urls: list[str],
    *,
    text_hint: str = "",
    model: str | None = None,
) -> UnitBoundaryPlan:
    deployment = model or DEPLOYMENT
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Identify visible unit boundaries in these rendered PDF pages. "
                "Use uploaded image order as physical page numbers. "
                "Do not use filename. Extracted text hint, if any:\n"
                f"{text_hint[:3000]}"
            ),
        }
    ]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    return client.chat.completions.create(
        model=deployment,
        response_model=UnitBoundaryPlan,
        messages=[
            {"role": "system", "content": VISUAL_UNIT_BOUNDARY_PROMPT},
            {"role": "user", "content": content},
        ],
    )


def extract_entities_from_images(
    client: instructor.Instructor,
    chunk: dict,
    image_data_urls: list[str],
    *,
    text_hint: str = "",
    model: str | None = None,
) -> CurriculumExtraction:
    deployment = model or DEPLOYMENT
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Extract curriculum structure from these rendered pages for one unit/document chunk. "
                f"Chunk title hint: {chunk.get('unit_title', '')}\n"
                f"Extracted text hint:\n{text_hint[:4000]}"
            ),
        }
    ]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model=deployment,
        response_model=CurriculumExtraction,
        messages=[
            {"role": "system", "content": CURRICULUM_EXTRACTION_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    data = response.model_dump()
    if not data.get("unit_title") and chunk.get("unit_title"):
        data["unit_title"] = chunk["unit_title"]
    return validate_schema(data)


from curriculum_os.engine.canonical import parse_canonical_document

GRADE_LABEL_PATTERN = re.compile(
    r"\b(?:Primary|Secondary|Form)\s*([1-6])\s*([A-B])?\b",
    re.IGNORECASE,
)
UNIT_LABEL_PATTERN = re.compile(r"\b(?:Unit|U)\s*([0-9]{1,2})\b", re.IGNORECASE)


def is_ntp_source(pdf_path: str | Path) -> bool:
    parsed = parse_canonical_document(pdf_path)
    return parsed.publisher == "NTP"


def primary_filename_grade(pdf_path: str | Path) -> tuple[str, list[str]]:
    parsed = parse_canonical_document(pdf_path)
    if parsed.publisher in ("P", "PRIMARY", "OPEN_TEXTBOOK", "PRIMARY_FORM_TOKEN") and parsed.inferred_grade:
        return parsed.inferred_grade, [
            f"canonical parser (primary filename): {parsed.raw_filename} -> {parsed.inferred_grade}"
        ]
    return "Unknown", []


def ntp_form_grade(pdf_path: str | Path) -> tuple[str, list[str]]:
    parsed = parse_canonical_document(pdf_path)
    if parsed.publisher == "NTP" and parsed.inferred_grade:
        return parsed.inferred_grade, [
            f"canonical parser (NTP form={parsed.form_number}{parsed.track or ''}): "
            f"series={parsed.series} -> {parsed.inferred_grade}"
        ]
    return "Unknown", []


def ntp_deterministic_grade(pdf_path: str | Path) -> tuple[str, list[str]]:
    return ntp_form_grade(pdf_path)


def parse_filename_hints(pdf_path: str | Path) -> dict:
    name = Path(pdf_path).stem
    parsed = parse_canonical_document(pdf_path)
    hints = {
        "source_code": "",
        "grade_token": "",
        "ntp_grade": "",
        "form_grade": "",
        "unit_number": "",
        "evidence": [],
        "canonical": parsed.model_dump(),
    }

    if parsed.inferred_grade:
        if parsed.publisher == "NTP":
            hints["ntp_grade"] = parsed.inferred_grade
            hints["form_grade"] = parsed.inferred_grade
        hints["evidence"].append(
            f"canonical parser: {parsed.publisher} form={parsed.form_number} -> {parsed.inferred_grade}"
        )

    if parsed.unit:
        hints["unit_number"] = parsed.unit
        hints["evidence"].append(f"canonical unit: U{parsed.unit}")

    code_match = re.match(r"^([A-Za-z]+[0-9][A-Za-z0-9]*)", name)
    if code_match:
        hints["source_code"] = code_match.group(1)
        hints["evidence"].append(f"filename source code: {hints['source_code']}")

    for token in re.split(r"[_\-.]+", name):
        m = re.search(r"\b([PS]?\d{1,2}[A-B]?)U(\d{1,2})\b", token, re.IGNORECASE)
        if m:
            hints["grade_token"] = m.group(1).upper()
            hints["unit_number"] = m.group(2)
            hints["evidence"].append(f"filename token: {m.group(0)}")
            break

    if not hints["unit_number"]:
        unit = re.search(r"\bU(\d{1,2})\b", name, re.IGNORECASE)
        if unit:
            hints["unit_number"] = unit.group(1)
            hints["evidence"].append(f"filename unit: U{hints['unit_number']}")
    return hints


def extract_unit_number(text: str) -> str:
    match = UNIT_LABEL_PATTERN.search(text or "")
    return match.group(1) if match else ""


def normalize_grade_label(label: str, school_level: str = "unknown") -> str:
    from curriculum_os.engine.grade_signals import normalize_grade

    return normalize_grade(label, default_school_level=school_level)


def resolve_grade(
    explicit_grade_label: str,
    school_level: str,
    filename_grade_token: str,
    *,
    filename_ntp_grade: str = "",
) -> tuple[str, list[str]]:
    evidence: list[str] = []
    ntp = (filename_ntp_grade or "").strip().upper()
    if ntp.startswith("S") and len(ntp) >= 2:
        evidence.append(f"deterministic NTP grade (pre-resolved): {ntp}")
        return ntp[:2], evidence

    explicit = normalize_grade_label(explicit_grade_label, school_level=school_level)
    if explicit != "Unknown":
        evidence.append(f"explicit grade label: {explicit_grade_label}")
        return explicit, evidence

    token = (filename_grade_token or "").strip().upper()
    if not token:
        return "Unknown", evidence

    if token.startswith("P") or token.startswith("S"):
        resolved = token[:2] if len(token) >= 2 else "Unknown"
        evidence.append(f"filename explicit grade token: {token}")
        return resolved, evidence

    evidence.append(f"filename token ignored as ambiguous: {token}")
    return "Unknown", evidence
