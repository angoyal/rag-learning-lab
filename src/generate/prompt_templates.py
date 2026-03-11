"""Prompt templates for RAG generation."""

from __future__ import annotations

import jinja2
from jinja2.sandbox import SandboxedEnvironment

_ENV = SandboxedEnvironment(undefined=jinja2.StrictUndefined)

_REWRITE_QUERY = _ENV.from_string(
    "Given this conversation history, rewrite the follow-up question "
    "to be a self-contained question that can be understood without "
    "the conversation history.\n"
    "\n"
    "Conversation history:\n"
    "{% for turn in history %}"
    "{{ turn.role | capitalize }}: {{ turn.content }}\n"
    "{% endfor %}"
    "\n"
    "Follow-up question: {{ question }}\n"
    "\n"
    "Rewritten question:"
)

_TEMPLATES = {
    "default_qa": _ENV.from_string(
        "Answer the question based on the context below. "
        "If the context does not contain enough information "
        "to answer, say \"I don't have enough information to answer this.\"\n"
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
    "conversational_qa": _ENV.from_string(
        "Answer the question based on the context and conversation history below. "
        "If the context does not contain enough information "
        "to answer, say \"I don't have enough information to answer this.\"\n"
        "\n"
        "Context:\n"
        "{% for chunk in chunks %}"
        "{{ chunk }}\n"
        "\n"
        "{% endfor %}"
        "{% if history %}"
        "Conversation history:\n"
        "{% for turn in history %}"
        "{{ turn.role | capitalize }}: {{ turn.content }}\n"
        "{% endfor %}"
        "\n"
        "{% endif %}"
        "Question: {{ question }}\n"
        "\n"
        "Answer:"
    ),
    "chain_of_thought": _ENV.from_string(
        "Answer the question based on the context below. First identify the "
        "relevant facts from the context, then reason through them step by "
        "step to reach your answer. If the context does not contain enough "
        "information to answer, say \"I don't have enough information to answer this.\"\n"
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


def render_prompt(
    template_name: str,
    chunks: list[str],
    question: str,
    history: list[dict] | None = None,
) -> str:
    """Render a RAG prompt from a named template.

    Args:
        template_name: One of "default_qa", "structured", "conversational_qa",
            or "chain_of_thought".
        chunks: List of context text chunks from the retriever.
        question: The user's question.
        history: Optional list of conversation turns, each a dict with
            "role" and "content" keys.

    Returns:
        The rendered prompt string ready to send to an LLM.

    Raises:
        ValueError: If the template name is not recognized.
    """
    template = _TEMPLATES.get(template_name)
    if template is None:
        raise ValueError(f"Unknown template: {template_name!r}")
    return template.render(chunks=chunks, question=question, history=history or [])


def render_rewrite(question: str, history: list[dict]) -> str:
    """Render a query-rewrite prompt for conversational follow-ups.

    Uses the conversation history to rewrite an ambiguous follow-up
    question into a self-contained question suitable for RAG retrieval.

    Args:
        question: The user's follow-up question.
        history: List of conversation turns, each a dict with
            "role" and "content" keys.

    Returns:
        The rendered rewrite prompt string.
    """
    return _REWRITE_QUERY.render(question=question, history=history)
