# Segmentation fixtures

JSON files with a `pages` array (synthetic PDF page text) and optional `source_file` for filename hints.

Paired golden files live in `tests/golden/segmentation/` with the same basename.

To add a case:

1. Add `my_case.json` here with `id`, `pages`, `source_file`.
2. Run segmentation once and tune `tests/golden/segmentation/my_case.json` bounds:
   - `chunk_count_min` / `chunk_count_max`
   - `titles_contain_any` / `titles_must_not`
   - `boundary_sources_allowed`
