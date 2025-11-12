"""Batch processing and XML prompt generation."""
import json
import logging
import re
import xml.sax.saxutils as saxutils
from typing import Any

from claude_lint.types import FileResult

logger = logging.getLogger(__name__)


def build_xml_prompt(claude_md_content: str, files: dict[str, str]) -> str:
    """Build XML prompt for Claude API.

    Args:
        claude_md_content: Content of CLAUDE.md
        files: Dict mapping file paths to their content

    Returns:
        XML formatted prompt
    """
    # Build files XML using list and join for efficiency
    files_xml_parts = []
    for file_path, content in files.items():
        escaped_path = saxutils.escape(file_path, {'"': "&quot;"})
        escaped_content = saxutils.escape(content)
        files_xml_parts.append(f'  <file path="{escaped_path}">\n{escaped_content}\n  </file>\n')
    files_xml = "".join(files_xml_parts)

    # Escape claude_md_content for XML
    escaped_claude_md = saxutils.escape(claude_md_content)

    prompt = f"""<guidelines>
{escaped_claude_md}
</guidelines>

Check the following files for compliance with the guidelines above.
For each file, evaluate:
1. Pattern compliance - Does the code follow specific patterns mentioned?
2. Principle adherence - Does the code embody the philosophy described?
3. Anti-pattern detection - Does the code contain things warned against?

<files>
{files_xml}</files>

Return results in this JSON format:
{{
  "results": [
    {{
      "file": "path/to/file",
      "violations": [
        {{
          "type": "missing-pattern|principle-violation|anti-pattern",
          "message": "Description of the issue",
          "line": null or line number
        }}
      ]
    }}
  ]
}}

If a file has no violations, include it with an empty violations array.
"""

    return prompt


def create_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split items into batches.

    Args:
        items: List of items to batch
        batch_size: Number of items per batch

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i : i + batch_size])
    return batches


def parse_response(response: str) -> list[FileResult]:
    """Parse Claude API response to extract results.

    Args:
        response: Raw response text from Claude

    Returns:
        List of file results with violations
    """
    # Extract JSON from markdown code blocks if present
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            logger.warning("No JSON found in response")
            return []

    try:
        data = json.loads(json_str)
        results: list[FileResult] = data.get("results", [])
        return results
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}")
        return []
