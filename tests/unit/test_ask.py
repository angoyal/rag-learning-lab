"""Tests for scripts/ask.py — conversation persistence, slash commands, display helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.ask import (
    Spinner,
    delete_all_conversations,
    delete_conversation,
    generate_title,
    handle_slash_command,
    list_conversations,
    load_conversation,
    print_chunks,
    print_sources,
    query_with_retry,
    save_conversation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conv_dir(tmp_path: Path) -> Path:
    """Return a temporary conversations directory."""
    return tmp_path / "conversations"


def _sample_turns() -> list[dict]:
    return [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    ]


def _save_samples(conv_dir: Path) -> list[str]:
    """Save three sample conversations and return their IDs in creation order."""
    ids = ["2026-01-01_100000", "2026-01-02_110000", "2026-01-03_120000"]
    titles = ["First question", "Second question", "Third question"]
    for cid, title in zip(ids, titles):
        save_conversation(conv_dir, cid, title, _sample_turns())
    return ids


def _run_cmd(
    command: str, conv_dir: Path,
    show_chunks: bool = False, show_prompt: bool = False,
    show_debug: bool = False, turns: list[dict] | None = None,
    last_results: list[dict] | None = None,
    pipeline=None, last_question: str | None = None,
    conv_id: str = "id", title: str = "",
) -> tuple:
    """Helper to run a slash command with the full signature."""
    return handle_slash_command(
        command, conv_dir, conv_id, title, turns or [],
        show_chunks, show_prompt, show_debug,
        last_results, pipeline, last_question,
    )


# ---------------------------------------------------------------------------
# save_conversation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveConversation:

    def test_creates_directory_and_file(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "test-id", "My title", _sample_turns())
        path = conv_dir / "test-id.yaml"
        assert path.exists()

    def test_saved_data_matches_input(self, conv_dir: Path) -> None:
        turns = _sample_turns()
        save_conversation(conv_dir, "abc", "My title", turns)
        with open(conv_dir / "abc.yaml") as f:
            data = yaml.safe_load(f)
        assert data["id"] == "abc"
        assert data["title"] == "My title"
        assert data["turns"] == turns

    def test_overwrites_existing(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "x", "old", [{"role": "user", "content": "old"}])
        new_turns = _sample_turns()
        save_conversation(conv_dir, "x", "new", new_turns)
        with open(conv_dir / "x.yaml") as f:
            data = yaml.safe_load(f)
        assert data["title"] == "new"
        assert data["turns"] == new_turns

    def test_empty_turns(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "empty", "empty conv", [])
        data = load_conversation(conv_dir, "empty")
        assert data["turns"] == []


# ---------------------------------------------------------------------------
# load_conversation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadConversation:

    def test_load_existing(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "c1", "title", _sample_turns())
        data = load_conversation(conv_dir, "c1")
        assert data is not None
        assert data["id"] == "c1"
        assert len(data["turns"]) == 2

    def test_load_nonexistent_returns_none(self, conv_dir: Path) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        assert load_conversation(conv_dir, "nope") is None

    def test_load_nonexistent_dir_returns_none(self, tmp_path: Path) -> None:
        assert load_conversation(tmp_path / "no-such-dir", "nope") is None


# ---------------------------------------------------------------------------
# list_conversations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListConversations:

    def test_empty_dir(self, conv_dir: Path) -> None:
        assert list_conversations(conv_dir) == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        assert list_conversations(tmp_path / "no-such-dir") == []

    def test_lists_all_sorted_descending(self, conv_dir: Path) -> None:
        ids = _save_samples(conv_dir)
        convos = list_conversations(conv_dir)
        assert len(convos) == 3
        # Most recent first
        assert convos[0]["id"] == ids[2]
        assert convos[1]["id"] == ids[1]
        assert convos[2]["id"] == ids[0]

    def test_includes_turn_count(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "c1", "t", _sample_turns())
        convos = list_conversations(conv_dir)
        assert convos[0]["turns"] == 2

    def test_includes_title(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "c1", "My Title", _sample_turns())
        convos = list_conversations(conv_dir)
        assert convos[0]["title"] == "My Title"


# ---------------------------------------------------------------------------
# delete_conversation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteConversation:

    def test_delete_existing(self, conv_dir: Path) -> None:
        save_conversation(conv_dir, "del-me", "title", _sample_turns())
        assert delete_conversation(conv_dir, "del-me") is True
        assert not (conv_dir / "del-me.yaml").exists()

    def test_delete_nonexistent(self, conv_dir: Path) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        assert delete_conversation(conv_dir, "nope") is False

    def test_delete_does_not_affect_others(self, conv_dir: Path) -> None:
        ids = _save_samples(conv_dir)
        delete_conversation(conv_dir, ids[1])
        remaining = list_conversations(conv_dir)
        assert len(remaining) == 2
        remaining_ids = {c["id"] for c in remaining}
        assert ids[1] not in remaining_ids


# ---------------------------------------------------------------------------
# delete_all_conversations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeleteAllConversations:

    def test_delete_all(self, conv_dir: Path) -> None:
        _save_samples(conv_dir)
        count = delete_all_conversations(conv_dir)
        assert count == 3
        assert list_conversations(conv_dir) == []

    def test_delete_all_empty_dir(self, conv_dir: Path) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        assert delete_all_conversations(conv_dir) == 0

    def test_delete_all_nonexistent_dir(self, tmp_path: Path) -> None:
        assert delete_all_conversations(tmp_path / "nope") == 0


# ---------------------------------------------------------------------------
# handle_slash_command — toggle commands
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashToggleCommands:

    def test_chunks_toggle_on(self, conv_dir: Path) -> None:
        result = _run_cmd("/chunks", conv_dir, show_chunks=False)
        assert result[3] is True   # show_chunks
        assert result[6] is True   # handled

    def test_chunks_toggle_off(self, conv_dir: Path) -> None:
        result = _run_cmd("/chunks", conv_dir, show_chunks=True)
        assert result[3] is False

    def test_prompt_toggle_on(self, conv_dir: Path) -> None:
        result = _run_cmd("/prompt", conv_dir, show_prompt=False)
        assert result[4] is True   # show_prompt

    def test_prompt_toggle_off(self, conv_dir: Path) -> None:
        result = _run_cmd("/prompt", conv_dir, show_prompt=True)
        assert result[4] is False

    def test_debug_toggle_on(self, conv_dir: Path) -> None:
        result = _run_cmd("/debug", conv_dir, show_debug=False)
        assert result[5] is True   # show_debug

    def test_debug_toggle_off(self, conv_dir: Path) -> None:
        result = _run_cmd("/debug", conv_dir, show_debug=True)
        assert result[5] is False


# ---------------------------------------------------------------------------
# handle_slash_command — /new
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashNew:

    def test_new_clears_turns(self, conv_dir: Path) -> None:
        turns = _sample_turns()
        result = _run_cmd(
            "/new", conv_dir, turns=turns, conv_id="old-id", title="old-title",
        )
        conv_id, title, new_turns = result[0], result[1], result[2]
        assert new_turns == []
        assert title == ""
        assert conv_id != "old-id"
        assert result[6] is True  # handled

    def test_new_saves_current_conversation(self, conv_dir: Path) -> None:
        turns = _sample_turns()
        _run_cmd("/new", conv_dir, turns=turns, conv_id="old-id", title="old-title")
        # The old conversation should be saved
        data = load_conversation(conv_dir, "old-id")
        assert data is not None
        assert data["title"] == "old-title"

    def test_new_with_no_turns_does_not_save(self, conv_dir: Path) -> None:
        _run_cmd("/new", conv_dir, conv_id="empty-id")
        assert load_conversation(conv_dir, "empty-id") is None


# ---------------------------------------------------------------------------
# handle_slash_command — /conversations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashConversations:

    def test_lists_empty(self, conv_dir: Path, capsys) -> None:
        _run_cmd("/conversations", conv_dir)
        output = capsys.readouterr().out
        assert "No saved conversations" in output

    def test_lists_existing(self, conv_dir: Path, capsys) -> None:
        _save_samples(conv_dir)
        _run_cmd("/conversations", conv_dir)
        output = capsys.readouterr().out
        assert "3 saved conversation(s)" in output
        assert "First question" in output
        assert "Third question" in output


# ---------------------------------------------------------------------------
# handle_slash_command — /resume with ID
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashResume:

    def test_resume_with_id(self, conv_dir: Path) -> None:
        ids = _save_samples(conv_dir)
        result = _run_cmd(f"/resume {ids[0]}", conv_dir)
        conv_id, title, turns = result[0], result[1], result[2]
        assert conv_id == ids[0]
        assert title == "First question"
        assert len(turns) == 2
        assert result[6] is True  # handled

    def test_resume_nonexistent(self, conv_dir: Path, capsys) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        result = _run_cmd("/resume no-such-id", conv_dir)
        output = capsys.readouterr().out
        assert "not found" in output
        assert result[2] == []  # turns

    def test_resume_saves_current_conversation(self, conv_dir: Path) -> None:
        ids = _save_samples(conv_dir)
        current_turns = [{"role": "user", "content": "current q"}]
        _run_cmd(
            f"/resume {ids[0]}", conv_dir,
            turns=current_turns, conv_id="cur-id", title="cur-title",
        )
        # Current conversation should be saved before resuming
        data = load_conversation(conv_dir, "cur-id")
        assert data is not None
        assert data["turns"] == current_turns


# ---------------------------------------------------------------------------
# handle_slash_command — /delete
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashDelete:

    def test_delete_existing(self, conv_dir: Path, capsys) -> None:
        ids = _save_samples(conv_dir)
        _run_cmd(f"/delete {ids[0]}", conv_dir)
        output = capsys.readouterr().out
        assert "Deleted" in output
        assert load_conversation(conv_dir, ids[0]) is None

    def test_delete_nonexistent(self, conv_dir: Path, capsys) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        _run_cmd("/delete nope", conv_dir)
        output = capsys.readouterr().out
        assert "not found" in output

    def test_delete_no_arg(self, conv_dir: Path, capsys) -> None:
        _run_cmd("/delete", conv_dir)
        output = capsys.readouterr().out
        assert "Usage" in output


# ---------------------------------------------------------------------------
# handle_slash_command — /delete-all
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashDeleteAll:

    def test_delete_all_confirmed(self, conv_dir: Path, monkeypatch, capsys) -> None:
        _save_samples(conv_dir)
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        result = _run_cmd(
            "/delete-all", conv_dir,
            turns=_sample_turns(), conv_id="cur-id", title="cur",
        )
        output = capsys.readouterr().out
        assert "Deleted 3" in output
        assert list_conversations(conv_dir) == []
        assert result[2] == []  # turns reset

    def test_delete_all_cancelled(self, conv_dir: Path, monkeypatch, capsys) -> None:
        _save_samples(conv_dir)
        monkeypatch.setattr("builtins.input", lambda _: "no")
        _run_cmd("/delete-all", conv_dir)
        output = capsys.readouterr().out
        assert "Cancelled" in output
        assert len(list_conversations(conv_dir)) == 3

    def test_delete_all_eof(self, conv_dir: Path, monkeypatch, capsys) -> None:
        _save_samples(conv_dir)

        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        _run_cmd("/delete-all", conv_dir)
        # Should not crash, conversations preserved
        assert len(list_conversations(conv_dir)) == 3


# ---------------------------------------------------------------------------
# handle_slash_command — /help and unknown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashHelp:

    def test_help(self, conv_dir: Path, capsys) -> None:
        result = _run_cmd("/help", conv_dir)
        output = capsys.readouterr().out
        assert "/chunks" in output
        assert "/resume" in output
        assert "/delete-all" in output
        assert "/debug" in output
        assert "/regenerate" in output
        assert "/compact" in output
        assert result[6] is True  # handled

    def test_unknown_command(self, conv_dir: Path, capsys) -> None:
        result = _run_cmd("/foobar", conv_dir)
        output = capsys.readouterr().out
        assert "Unknown command" in output
        assert result[6] is True  # handled


# ---------------------------------------------------------------------------
# handle_slash_command — /regenerate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashRegenerate:

    def test_regenerate_returns_question(self, conv_dir: Path) -> None:
        turns = _sample_turns()
        result = _run_cmd(
            "/regenerate", conv_dir, turns=turns, last_question="What is RAG?",
        )
        assert result[7] == "What is RAG?"  # regenerate_question
        # Last Q&A pair should be removed from turns
        assert len(result[2]) == 0

    def test_regenerate_no_previous_question(self, conv_dir: Path, capsys) -> None:
        result = _run_cmd("/regenerate", conv_dir)
        output = capsys.readouterr().out
        assert "No previous question" in output
        assert result[7] is None

    def test_regenerate_preserves_earlier_turns(self, conv_dir: Path) -> None:
        turns = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer"},
        ]
        result = _run_cmd(
            "/regenerate", conv_dir, turns=turns, last_question="Second question",
        )
        assert len(result[2]) == 2  # Only first Q&A pair remains
        assert result[2][0]["content"] == "First question"
        assert result[7] == "Second question"


# ---------------------------------------------------------------------------
# print_sources
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrintSources:

    def test_prints_unique_sources(self, capsys) -> None:
        results = [
            {"metadata": {"source": "/data/paper_a.pdf"}},
            {"metadata": {"source": "/data/paper_b.pdf"}},
            {"metadata": {"source": "/data/paper_a.pdf"}},
        ]
        print_sources(results)
        output = capsys.readouterr().out
        assert "paper_a.pdf" in output
        assert "paper_b.pdf" in output
        # Should appear on only one line (deduplicated)
        lines_with_a = [line for line in output.splitlines() if "paper_a.pdf" in line]
        assert len(lines_with_a) == 1

    def test_no_sources(self, capsys) -> None:
        print_sources([{"metadata": {}}])
        output = capsys.readouterr().out
        assert output == ""

    def test_empty_results(self, capsys) -> None:
        print_sources([])
        output = capsys.readouterr().out
        assert output == ""


# ---------------------------------------------------------------------------
# print_chunks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPrintChunks:

    def test_prints_chunks_with_sources(self, capsys) -> None:
        results = [
            {"metadata": {"source": "/data/doc.pdf"}},
            {"metadata": {"source": "/data/other.pdf"}},
        ]
        chunks = ["First chunk text here", "Second chunk text here"]
        print_chunks(results, chunks)
        output = capsys.readouterr().out
        assert "[1] [doc.pdf]" in output
        assert "[2] [other.pdf]" in output
        assert "First chunk" in output

    def test_truncates_long_chunks(self, capsys) -> None:
        results = [{"metadata": {"source": "a.pdf"}}]
        chunks = ["x" * 300]
        print_chunks(results, chunks)
        output = capsys.readouterr().out
        assert "..." in output
        # Should be truncated to 200 chars + ...
        lines = [line for line in output.splitlines() if "x" in line]
        assert len(lines) == 1

    def test_handles_missing_source(self, capsys) -> None:
        results = [{"metadata": {}}]
        chunks = ["some text"]
        print_chunks(results, chunks)
        output = capsys.readouterr().out
        assert "[1]" in output
        assert "some text" in output


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpinner:

    def test_start_and_stop(self) -> None:
        spinner = Spinner("Testing")
        spinner.start()
        assert spinner._thread is not None
        assert spinner._thread.is_alive()
        spinner.stop()
        assert not spinner._thread.is_alive()

    def test_stop_is_idempotent(self) -> None:
        spinner = Spinner()
        spinner.start()
        spinner.stop()
        spinner.stop()  # Should not raise

    def test_spinner_is_daemon(self) -> None:
        spinner = Spinner()
        spinner.start()
        assert spinner._thread.daemon is True
        spinner.stop()


# ---------------------------------------------------------------------------
# query_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryWithRetry:

    def test_success_on_first_try(self) -> None:
        class FakePipeline:
            def query(self, question, history=None):
                return {"answer": "ok", "chunks": [], "results": []}

        result = query_with_retry(FakePipeline(), "test", [], False)
        assert result["answer"] == "ok"

    def test_retries_on_connection_error(self) -> None:
        call_count = 0

        class FakePipeline:
            def query(self, question, history=None):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("timeout")
                return {"answer": "recovered"}

        result = query_with_retry(FakePipeline(), "test", [], False)
        assert result["answer"] == "recovered"
        assert call_count == 3

    def test_raises_after_max_retries(self) -> None:
        class FakePipeline:
            def query(self, question, history=None):
                raise ConnectionError("always fails")

        with pytest.raises(ConnectionError, match="always fails"):
            query_with_retry(FakePipeline(), "test", [], False)

    def test_non_retryable_error_not_caught(self) -> None:
        class FakePipeline:
            def query(self, question, history=None):
                raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            query_with_retry(FakePipeline(), "test", [], False)


# ---------------------------------------------------------------------------
# generate_title
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateTitle:

    def test_returns_llm_title(self) -> None:
        class FakeLLM:
            def generate(self, prompt):
                return "  RAG Overview  "

        class FakePipeline:
            llm = FakeLLM()

        title = generate_title(FakePipeline(), "What is RAG?", "RAG is...")
        assert title == "RAG Overview"

    def test_strips_quotes(self) -> None:
        class FakeLLM:
            def generate(self, prompt):
                return '"My Title"'

        class FakePipeline:
            llm = FakeLLM()

        title = generate_title(FakePipeline(), "q", "a")
        assert title == "My Title"

    def test_fallback_on_error(self) -> None:
        class FakeLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM down")

        class FakePipeline:
            llm = FakeLLM()

        title = generate_title(FakePipeline(), "What is RAG?", "answer")
        assert title == "What is RAG?"

    def test_truncates_long_title(self) -> None:
        class FakeLLM:
            def generate(self, prompt):
                return "A" * 200

        class FakePipeline:
            llm = FakeLLM()

        title = generate_title(FakePipeline(), "q", "a")
        assert len(title) == 80


# ---------------------------------------------------------------------------
# Conversation persistence round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConversationRoundTrip:

    def test_save_load_delete_lifecycle(self, conv_dir: Path) -> None:
        turns = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        save_conversation(conv_dir, "lifecycle", "First Q", turns)

        # Load and verify
        data = load_conversation(conv_dir, "lifecycle")
        assert data["turns"] == turns
        assert data["title"] == "First Q"

        # Appears in list
        convos = list_conversations(conv_dir)
        assert any(c["id"] == "lifecycle" for c in convos)

        # Delete
        assert delete_conversation(conv_dir, "lifecycle") is True
        assert load_conversation(conv_dir, "lifecycle") is None
        assert not any(c["id"] == "lifecycle" for c in list_conversations(conv_dir))

    def test_unicode_content_preserved(self, conv_dir: Path) -> None:
        turns = [
            {"role": "user", "content": "What about 日本語?"},
            {"role": "assistant", "content": "日本語 means Japanese."},
        ]
        save_conversation(conv_dir, "unicode", "日本語 question", turns)
        data = load_conversation(conv_dir, "unicode")
        assert data["turns"][0]["content"] == "What about 日本語?"
        assert data["title"] == "日本語 question"

    def test_special_chars_in_content(self, conv_dir: Path) -> None:
        turns = [
            {"role": "user", "content": "What about <script>alert('xss')</script>?"},
            {"role": "assistant", "content": "That's a & dangerous < pattern."},
        ]
        save_conversation(conv_dir, "special", "XSS test", turns)
        data = load_conversation(conv_dir, "special")
        assert "<script>" in data["turns"][0]["content"]
        assert "&" in data["turns"][1]["content"]
