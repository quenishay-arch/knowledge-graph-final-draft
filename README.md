# HK English Curriculum OS

Unified pipeline for English curriculum PDFs:

**PDFs in one folder → content-first grade inference → unit segmentation → LLM extraction → knowledge graph export**

## Layout

```
curriculum_os/          # Current pipeline, router, canonical parser, export
curriculum_os/engine/   # PDF ingest, segmentation, extraction, graph builder
orchestrator/           # CLI entry
data/input_pdfs/        # Put every input PDF here
data/outputs/           # by_document/, knowledge_graph.json, knowledge_graph.xlsx
tests/                  # Unit and segmentation tests
benchmark/              # Optional scoring tools and manifest labels
```

## Quick Start

```bash
pip install -r requirements.txt
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_KEY=...

python -m orchestrator.cli route
python -m orchestrator.cli run
```

You can also pass another folder:

```bash
python -m orchestrator.cli run --pdf-dir /path/to/pdfs
```

## Grade Resolution

Ranked NLP-style pipeline — signals are processed in priority order, then cross-validated:

1. **Explicit text markers** — `Primary 5`, `Form 1`, `PRIMARY FIVE` in cover/front matter or body text.
2. **Filename shorthand** — `P4.pdf`, `p3b.pdf` (strong signal; content validates, flags mismatch).
3. **Curriculum vocabulary** — grammar topic difficulty (passive voice → P5, demonstratives → S1, toys/colours → P2).
4. **Lexical complexity** — sentence length and word-length heuristics when topics are ambiguous.
5. **Metadata** — publisher notes, series labels (tie-breaker only).

Only **NTP filenames** hard-lock grade (`NTP3E_1AU1.pdf` → `S1` from form token `1A`). Ambiguous names like `3B-book1_...pdf` or `1A_U5U6.pdf` are **not** locked — grade comes from explicit text + semantic classification.

| Pattern | Grade signal |
| --- | --- |
| `NTP3E_1AU1.pdf` | Hard lock `S1` from `1A` |
| `p3b.pdf` / `P4.pdf` | Filename signal `P3` / `P4` |
| `files.pdf` with `PRIMARY FIVE` on cover | Explicit text → `P5` |
| `yes.pdf` with connectives / demonstratives | Semantic → `S1` |
| `ok.pdf` with toys, colours, clothes | Semantic → `P2` |

**Confidence scoring** (dynamic — never defaults to P1):
- Explicit marker + content agree → 0.90+
- Filename + content agree → 0.88+
- Content only → ≤0.68, flagged for human review
- Conflicting signals → warnings + review
- No corroborated evidence → `UNKNOWN`

Rejected as grade markers: series ranges (`Primary 1–6`), unit labels (`Unit 1`), bare digits.

## Unit Segmentation

Units are detected and chunked through a parsing pipeline (independent of grade):

1. **Structural markers** — `Unit 1`, `Chapter 3`, `Module` labels with titles.
2. **TOC parsing** — Contents table rows (e.g. *International food fair*, *More places to see in Hong Kong*).
3. **Numeric headers** — bare numbers like `5 My toys` validated by unit-level content (Vocabulary, Reading, Language focus) and sequential consistency (5 → 6).
4. **Header grouping** — repeated section headers at page tops.
5. **Cross-check** — regex vs TOC vs headers vs segmenter; filename unit hints (`U5U6`, `c5_c6`, NTP `U1`) validate but never drive segmentation alone.

Each document gets a **unit confidence score** (0–1) and **warnings** when signals disagree or filename hints mismatch content. `needs_human_review` is set when confidence is low or cross-check fails.

Module-style documents (one module, N texts) count as **1 module** with sub-unit titles listed in evidence.

For image-only PDFs with little extractable text, the engine tries visual unit-boundary detection. If your Azure deployment does not support vision, the output includes a text-quality warning so you know OCR/vision is needed rather than getting silent hallucinated chunks.

## Tests

```bash
python -m pytest
python benchmark/smoke_grade.py
```
