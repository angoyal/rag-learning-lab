"""Tests for scripts/ask.py — conversation persistence, slash commands, display helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from scripts.ask import (
    delete_all_conversations,
    delete_conversation,
    handle_slash_command,
    list_conversations,
    load_conversation,
    print_chunks,
    print_sources,
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

    def _run(
        self, command: str, conv_dir: Path,
        show_chunks: bool = False, show_prompt: bool = False,
    ) -> tuple:
        return handle_slash_command(
            command, conv_dir, "id", "title", [], show_chunks, show_prompt,
        )

    def test_chunks_toggle_on(self, conv_dir: Path) -> None:
        result = self._run("/chunks", conv_dir, show_chunks=False)
        assert result[3] is True   # show_chunks
        assert result[5] is True   # handled

    def test_chunks_toggle_off(self, conv_dir: Path) -> None:
        result = self._run("/chunks", conv_dir, show_chunks=True)
        assert result[3] is False

    def test_prompt_toggle_on(self, conv_dir: Path) -> None:
        result = self._run("/prompt", conv_dir, show_prompt=False)
        assert result[4] is True   # show_prompt

    def test_prompt_toggle_off(self, conv_dir: Path) -> None:
        result = self._run("/prompt", conv_dir, show_prompt=True)
        assert result[4] is False


# ---------------------------------------------------------------------------
# handle_slash_command — /new
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashNew:

    def test_new_clears_turns(self, conv_dir: Path) -> None:
        turns = _sample_turns()
        conv_id, title, new_turns, _, _, handled = handle_slash_command(
            "/new", conv_dir, "old-id", "old-title", turns, False, False,
        )
        assert new_turns == []
        assert title == ""
        assert conv_id != "old-id"
        assert handled is True

    def test_new_saves_current_conversation(self, conv_dir: Path) -> None:
        turns = _sample_turns()
        handle_slash_command(
            "/new", conv_dir, "old-id", "old-title", turns, False, False,
        )
        # The old conversation should be saved
        data = load_conversation(conv_dir, "old-id")
        assert data is not None
        assert data["title"] == "old-title"

    def test_new_with_no_turns_does_not_save(self, conv_dir: Path) -> None:
        handle_slash_command(
            "/new", conv_dir, "empty-id", "", [], False, False,
        )
        assert load_conversation(conv_dir, "empty-id") is None


# ---------------------------------------------------------------------------
# handle_slash_command — /conversations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashConversations:

    def test_lists_empty(self, conv_dir: Path, capsys) -> None:
        handle_slash_command(
            "/conversations", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "No saved conversations" in output

    def test_lists_existing(self, conv_dir: Path, capsys) -> None:
        _save_samples(conv_dir)
        handle_slash_command(
            "/conversations", conv_dir, "id", "", [], False, False,
        )
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
        conv_id, title, turns, _, _, handled = handle_slash_command(
            f"/resume {ids[0]}", conv_dir, "cur-id", "cur", [], False, False,
        )
        assert conv_id == ids[0]
        assert title == "First question"
        assert len(turns) == 2
        assert handled is True

    def test_resume_nonexistent(self, conv_dir: Path, capsys) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        _, _, turns, _, _, _ = handle_slash_command(
            "/resume no-such-id", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "not found" in output
        assert turns == []

    def test_resume_saves_current_conversation(self, conv_dir: Path) -> None:
        ids = _save_samples(conv_dir)
        current_turns = [{"role": "user", "content": "current q"}]
        handle_slash_command(
            f"/resume {ids[0]}", conv_dir, "cur-id", "cur-title",
            current_turns, False, False,
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
        handle_slash_command(
            f"/delete {ids[0]}", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "Deleted" in output
        assert load_conversation(conv_dir, ids[0]) is None

    def test_delete_nonexistent(self, conv_dir: Path, capsys) -> None:
        conv_dir.mkdir(parents=True, exist_ok=True)
        handle_slash_command(
            "/delete nope", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "not found" in output

    def test_delete_no_arg(self, conv_dir: Path, capsys) -> None:
        handle_slash_command(
            "/delete", conv_dir, "id", "", [], False, False,
        )
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
        conv_id, title, turns, _, _, _ = handle_slash_command(
            "/delete-all", conv_dir, "cur-id", "cur", _sample_turns(), False, False,
        )
        output = capsys.readouterr().out
        assert "Deleted 3" in output
        assert list_conversations(conv_dir) == []
        # Current conversation should be reset
        assert turns == []

    def test_delete_all_cancelled(self, conv_dir: Path, monkeypatch, capsys) -> None:
        _save_samples(conv_dir)
        monkeypatch.setattr("builtins.input", lambda _: "no")
        handle_slash_command(
            "/delete-all", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "Cancelled" in output
        assert len(list_conversations(conv_dir)) == 3

    def test_delete_all_eof(self, conv_dir: Path, monkeypatch, capsys) -> None:
        _save_samples(conv_dir)

        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        handle_slash_command(
            "/delete-all", conv_dir, "id", "", [], False, False,
        )
        # Should not crash, conversations preserved
        assert len(list_conversations(conv_dir)) == 3


# ---------------------------------------------------------------------------
# handle_slash_command — /help and unknown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlashHelp:

    def test_help(self, conv_dir: Path, capsys) -> None:
        _, _, _, _, _, handled = handle_slash_command(
            "/help", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "/chunks" in output
        assert "/resume" in output
        assert "/delete-all" in output
        assert handled is True

    def test_unknown_command(self, conv_dir: Path, capsys) -> None:
        _, _, _, _, _, handled = handle_slash_command(
            "/foobar", conv_dir, "id", "", [], False, False,
        )
        output = capsys.readouterr().out
        assert "Unknown command" in output
        assert handled is True


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
        # Should appear only once
        assert output.count("paper_a.pdf") == 1

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
