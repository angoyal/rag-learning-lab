"""Tests for output safety.

Covers OWASP LLM10 — output length bounds, no executable output,
and hallucination on absence.

Tests verify prompt template safety properties without requiring a running LLM.
"""

import pytest
from src.generate.prompt_templates import render_prompt


@pytest.mark.security
def test_output_length_bounds():
    """Prompt length should scale linearly with chunk count, not exponentially."""
    small_chunks = ["Fact A."] * 3
    large_chunks = ["Fact A."] * 100
    small_prompt = render_prompt("structured", small_chunks, "Question?")
    large_prompt = render_prompt("structured", large_chunks, "Question?")
    # Length should scale roughly linearly
    ratio = len(large_prompt) / len(small_prompt)
    assert ratio < 50  # 100/3 ≈ 33x, so 50x is generous


@pytest.mark.security
def test_no_executable_output():
    """Chunks containing code should be passed through as text, not executed."""
    code_chunk = "```python\nimport os\nos.system('rm -rf /')\n```"
    prompt = render_prompt("structured", [code_chunk], "What does this code do?")
    # The code should appear verbatim in the prompt as context
    assert "os.system" in prompt
    assert "rm -rf" in prompt
    # But it's just text in the prompt, not executed


@pytest.mark.security
def test_hallucination_on_absence():
    """The structured template includes a guardrail for missing information."""
    prompt = render_prompt("structured", [], "What is quantum computing?")
    # With no context, the structured template still instructs the LLM to say "I don't know"
    assert "don't know" in prompt.lower()
