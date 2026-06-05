"""Helpers for extracting content from LLM responses.

LLMs frequently wrap their output in markdown code fences (```json ... ``` or
```python ... ```) even when the prompt asks for raw output. These helpers
strip that wrapper so the body can be parsed or executed.
"""


def _strip_code_fence(text: str) -> str:
    """Remove a leading/trailing markdown code fence if present.

    Handles ```json, ```python, plain ``` fences, and surrounding whitespace.
    If no fence is present, returns the trimmed text unchanged.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening fence (``` or ```json / ```python)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        cleaned = "\n".join(lines).strip()

    return cleaned


def extract_json(text: str) -> str:
    """Return the JSON body from an LLM response, stripping markdown fences."""
    return _strip_code_fence(text)


def extract_code(text: str) -> str:
    """Return the Python code body from an LLM response, stripping markdown fences."""
    return _strip_code_fence(text)
