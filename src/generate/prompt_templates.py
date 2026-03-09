"""Prompt templates for RAG generation."""

from __future__ import annotations

import jinja2

_ENV = jinja2.Environment(autoescape=False, undefined=jinja2.StrictUndefined)

_TEMPLATES = {
    "default_qa": _ENV.from_string(
        "Answer the question based on the context below.\n"
        "\n"
        "Context:\n"
        "{% for chunk in chunks %}"
        "{{ chunk }}\n"
        "\n"
        "{% endfor %}"
        "Question: {{ question }}\n"
        "\n"
        "Answer:"
    ),
    "structured": _ENV.from_string(
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain enough information "
        "to answer, say \"I don't know.\"\n"
        "\n"
        "Context:\n"
        "{% for chunk in chunks %}"
        "[{{ loop.index }}] {{ chunk }}\n"
        "{% endfor %}"
        "\n"
        "Question: {{ question }}\n"
        "\n"
        "Answer:"
    ),
    "chain_of_thought": _ENV.from_string(
        "Answer the question based on the context below. First identify the "
        "relevant facts from the context, then reason through them step by "
        "step to reach your answer.\n"
        "\n"
        "Context:\n"
        "{% for chunk in chunks %}"
        "{{ chunk }}\n"
        "\n"
        "{% endfor %}"
        "Question: {{ question }}\n"
        "\n"
        "Reasoning and Answer:"
    ),
}


def render_prompt(template_name: str, chunks: list[str], question: str) -> str:
    """Render a RAG prompt from a named template.

    Args:
        template_name: One of "default_qa", "structured", or "chain_of_thought".
        chunks: List of context text chunks from the retriever.
        question: The user's question.

    Returns:
        The rendered prompt string ready to send to an LLM.

    Raises:
        ValueError: If the template name is not recognized.
    """
    template = _TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name!r}")
    return template.render(chunks=chunks, question=question)
