DOCUMENT_TYPE_PROMPT = """
You classify educational PDF snippets into a single document type.

Choose exactly one:
- full_textbook
- workbook
- single_unit
- worksheet
- assessment_paper
- toc_only
- unknown

Rules:
- Use only explicit evidence in the snippet.
- Include concise evidence strings copied or paraphrased from text.
- Do not infer grade or curriculum in this step.
"""

GLOBAL_METADATA_PROMPT = """
You extract global textbook metadata from cover pages, headers and table of contents.

Return ONLY explicit metadata:
- explicit_grade_label: labels like "Primary 3", "P3", "Book 3A"
- publisher
- textbook_series
- source_code: series code if shown (e.g., NTP3E)
- semester_or_book
- unit_number: if explicitly shown on cover/header (e.g., Unit 1, U1)
- school_level: primary | secondary | unknown
- units_explicitly_listed: true only if unit/module list is visible
- evidence: short snippets supporting each field

Important:
- Do NOT infer or guess missing values.
- If unknown, return empty string (or unknown for school_level).
"""

CURRICULUM_EXTRACTION_PROMPT = """
You extract curriculum structure from one unit of a Hong Kong English textbook.

Extract ONLY what appears in the text:
- unit_title: module/unit heading if present
- grammar_points: language structures, grammar focus (list of strings)
- vocabulary_themes: vocabulary topics or word groups (list of strings)
- vocabulary_words: actual vocabulary words when explicitly listed (list of strings)
- language_skills: Reading, Writing, Speaking, Listening, Phonics, etc.

Rules:
- No invention.
- Keep lists concise and deduplicated.
- If a field is missing, return an empty list (or empty string for unit_title).
"""

GRADE_INFERENCE_PROMPT = """
You infer the most likely school grade for Hong Kong English learning material from content evidence.

Return:
- predicted_grade: one of P1, P2, P3, P4, P5, P6, S1, S2, S3, S4, S5, S6, Unknown
- confidence: 0.0 to 1.0
- evidence: concise snippets or signals from the material
- reasoning_summary: one short explanation

Decision rules:
- Prefer explicit labels in the material, e.g. Primary 3, P3, Secondary 2, Book 1A.
- If there is no explicit grade, infer from language complexity, text types, grammar points, task type, vocabulary, and curriculum level.
- Do not use the filename.
- Return Unknown with low confidence if the text is mostly copyright, blank, OCR noise, or insufficient for grade inference.
- Do not guess a grade just because the material mentions a unit number.
"""

VISUAL_UNIT_BOUNDARY_PROMPT = """
You identify visible unit boundaries in rendered textbook PDF pages.

Return:
- units: list of units with unit_title, start_page, end_page, evidence
- confidence: 0.0 to 1.0
- evidence: short global evidence

Rules:
- Page numbers are the uploaded image order: first image is page 1, second image is page 2, etc.
- Create a unit only when a visible cover, contents page, unit heading, module heading, or repeated header supports it.
- Do not invent regular page clusters.
- If the document shows one unit/module, return one unit.
- If the document visibly contains two units, return two units.
- If boundaries are unclear, return an empty units list or one broad document-level unit with low confidence.
"""
