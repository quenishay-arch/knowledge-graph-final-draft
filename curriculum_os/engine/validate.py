from pydantic import ValidationError

from curriculum_os.engine.models import CurriculumExtraction


def validate_schema(data: dict | CurriculumExtraction) -> CurriculumExtraction:
    """Validate and normalize extraction output against the schema."""
    if isinstance(data, CurriculumExtraction):
        return data
    try:
        return CurriculumExtraction.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}") from e
