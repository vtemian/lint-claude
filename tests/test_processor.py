from claude_lint.processor import create_batches, parse_response, build_xml_prompt


def test_build_xml_prompt():
    """Test building XML prompt for Claude API."""
    claude_md_content = "# Guidelines\n\nFollow TDD."
    files = {"src/main.py": "def main():\n    pass", "src/utils.py": "def helper():\n    return 42"}

    prompt = build_xml_prompt(claude_md_content, files)

    # Check structure
    assert "<guidelines>" in prompt
    assert "Follow TDD" in prompt
    assert "</guidelines>" in prompt
    assert "<files>" in prompt
    assert '<file path="src/main.py">' in prompt
    assert '<file path="src/utils.py">' in prompt
    assert "def main()" in prompt
    assert "def helper()" in prompt
    assert "</files>" in prompt


def test_build_xml_prompt_escapes_special_characters():
    """Test that special XML characters are properly escaped."""
    claude_md_content = "Use <template> & avoid >>>"
    files = {'path/with"quote.py': "x = 5 < 10 & y > 3", "normal.py": "return a & b"}

    prompt = build_xml_prompt(claude_md_content, files)

    # Check that claude_md_content is escaped
    assert "Use &lt;template&gt; &amp; avoid &gt;&gt;&gt;" in prompt

    # Check that file paths with quotes are escaped
    assert "path/with&quot;quote.py" in prompt

    # Check that file content is escaped
    assert "x = 5 &lt; 10 &amp; y &gt; 3" in prompt
    assert "return a &amp; b" in prompt

    # Ensure no unescaped special characters in file content
    assert "x = 5 < 10 & y > 3" not in prompt


def test_batch_files():
    """Test batching files into groups."""
    files = [f"file{i}.py" for i in range(25)]
    batch_size = 10

    batches = create_batches(files, batch_size)

    assert len(batches) == 3  # 10, 10, 5
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5


def test_parse_response():
    """Test parsing Claude API response."""
    response = """
    Here are the compliance issues:

    ```json
    {
      "results": [
        {
          "file": "src/main.py",
          "violations": [
            {
              "type": "missing-pattern",
              "message": "No tests found for this module",
              "line": null
            }
          ]
        },
        {
          "file": "src/utils.py",
          "violations": []
        }
      ]
    }
    ```
    """

    results = parse_response(response)

    assert len(results) == 2
    assert results[0]["file"] == "src/main.py"
    assert len(results[0]["violations"]) == 1
    assert results[1]["file"] == "src/utils.py"
    assert len(results[1]["violations"]) == 0
