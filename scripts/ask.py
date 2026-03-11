"""Interactive Q&A with your RAG pipeline.

Supports conversational follow-ups with query rewriting,
persistent conversation history, and slash commands.

Usage:
    uv run python scripts/ask.py --config configs/experiments/arxiv_auto_scaling.yaml
    uv run python scripts/ask.py --config configs/experiments/01_baseline.yaml --show-chunks
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from src.pipeline import RAGPipeline, load_config

DEFAULT_CONVERSATIONS_DIR = "data/conversations"


def conversation_id() -> str:
    """Generate a conversation ID from the current UTC timestamp."""
    return datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")


def save_conversation(
    conv_dir: Path, conv_id: str, title: str, turns: list[dict]
) -> None:
    """Save a conversation to a YAML file.

    Args:
        conv_dir: Directory to save conversations in.
        conv_id: Unique conversation identifier.
        title: Conversation title (first question asked).
        turns: List of turn dicts with "role" and "content" keys.
    """
    conv_dir.mkdir(parents=True, exist_ok=True)
    data = {"id": conv_id, "title": title, "created": conv_id, "turns": turns}
    with open(conv_dir / f"{conv_id}.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_conversation(conv_dir: Path, conv_id: str) -> dict | None:
    """Load a conversation from a YAML file.

    Args:
        conv_dir: Directory containing conversation files.
        conv_id: Conversation identifier to load.

    Returns:
        Conversation dict with id, title, turns, or None if not found.
    """
    path = conv_dir / f"{conv_id}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def list_conversations(conv_dir: Path) -> list[dict]:
    """List all saved conversations, sorted by date descending.

    Args:
        conv_dir: Directory containing conversation files.

    Returns:
        List of conversation dicts with id, title, and turn count.
    """
    if not conv_dir.exists():
        return []
    conversations = []
    for path in sorted(conv_dir.glob("*.yaml"), reverse=True):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data:
            conversations.append({
                "id": data["id"],
                "title": data.get("title", "untitled"),
                "turns": len(data.get("turns", [])),
            })
    return conversations


def delete_conversation(conv_dir: Path, conv_id: str) -> bool:
    """Delete a specific conversation file.

    Args:
        conv_dir: Directory containing conversation files.
        conv_id: Conversation identifier to delete.

    Returns:
        True if deleted, False if not found.
    """
    path = conv_dir / f"{conv_id}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False


def delete_all_conversations(conv_dir: Path) -> int:
    """Delete all conversation files.

    Args:
        conv_dir: Directory containing conversation files.

    Returns:
        Number of conversations deleted.
    """
    if not conv_dir.exists():
        return 0
    count = 0
    for path in conv_dir.glob("*.yaml"):
        path.unlink()
        count += 1
    return count


def print_sources(results: list[dict]) -> None:
    """Print unique source filenames from retrieval results."""
    sources = []
    for r in results:
        source = r.get("metadata", {}).get("source", "")
        if source:
            sources.append(Path(source).name)
    unique_sources = list(dict.fromkeys(sources))
    if unique_sources:
        print(f"\nSources: {', '.join(unique_sources)}")


def print_chunks(results: list[dict], chunks: list[str]) -> None:
    """Print retrieved chunks with source attribution."""
    print("--- Retrieved Chunks ---")
    for i, chunk in enumerate(chunks, 1):
        source_name = ""
        if i <= len(results):
            s = results[i - 1].get("metadata", {}).get("source", "")
            if s:
                source_name = f" [{Path(s).name}]"
        print(f"  [{i}]{source_name} {chunk[:200]}...")
    print()


def handle_slash_command(
    command: str, conv_dir: Path, conv_id: str, title: str, turns: list[dict],
    show_chunks: bool, show_prompt: bool,
) -> tuple[str, str, list[dict], bool, bool, bool]:
    """Handle a slash command and return updated state.

    Args:
        command: The slash command string (e.g., "/conversations").
        conv_dir: Directory containing conversation files.
        conv_id: Current conversation ID.
        title: Current conversation title.
        turns: Current conversation turns.
        show_chunks: Current show-chunks toggle state.
        show_prompt: Current show-prompt toggle state.

    Returns:
        Tuple of (conv_id, title, turns, show_chunks, show_prompt, handled).
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/chunks":
        show_chunks = not show_chunks
        print(f"  show-chunks: {'ON' if show_chunks else 'OFF'}\n")
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/prompt":
        show_prompt = not show_prompt
        print(f"  show-prompt: {'ON' if show_prompt else 'OFF'}\n")
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/new":
        if turns:
            save_conversation(conv_dir, conv_id, title, turns)
            print(f"  Saved conversation '{title}' ({conv_id})")
        conv_id = conversation_id()
        title = ""
        turns = []
        print("  Started new conversation\n")
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/conversations":
        convos = list_conversations(conv_dir)
        if not convos:
            print("  No saved conversations\n")
        else:
            print(f"  {len(convos)} saved conversation(s):")
            for c in convos:
                print(f"    {c['id']}  {c['title']}  ({c['turns']} turns)")
            print()
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/resume":
        if arg:
            target_id = arg
        else:
            convos = list_conversations(conv_dir)
            if not convos:
                print("  No saved conversations\n")
                return conv_id, title, turns, show_chunks, show_prompt, True
            print("  Saved conversations:")
            for i, c in enumerate(convos, 1):
                print(f"    {i}. {c['id']}  {c['title']}  ({c['turns']} turns)")
            try:
                choice = input("  Enter number or ID: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return conv_id, title, turns, show_chunks, show_prompt, True
            if choice.isdigit() and 1 <= int(choice) <= len(convos):
                target_id = convos[int(choice) - 1]["id"]
            else:
                target_id = choice

        data = load_conversation(conv_dir, target_id)
        if data is None:
            print(f"  Conversation '{target_id}' not found\n")
        else:
            if turns:
                save_conversation(conv_dir, conv_id, title, turns)
            conv_id = data["id"]
            title = data.get("title", "")
            turns = data.get("turns", [])
            print(f"  Resumed '{title}' ({len(turns)} turns)\n")
            for turn in turns[-6:]:
                role = "You" if turn["role"] == "user" else "Bot"
                content = turn["content"]
                if len(content) > 120:
                    content = content[:120] + "..."
                print(f"    {role}: {content}")
            if len(turns) > 6:
                print(f"    ... ({len(turns) - 6} earlier turns)")
            print()
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/delete":
        if not arg:
            print("  Usage: /delete <conversation-id>\n")
            return conv_id, title, turns, show_chunks, show_prompt, True
        if delete_conversation(conv_dir, arg):
            print(f"  Deleted conversation '{arg}'\n")
        else:
            print(f"  Conversation '{arg}' not found\n")
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/delete-all":
        try:
            confirm = input("  Delete all conversations? (yes/no): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return conv_id, title, turns, show_chunks, show_prompt, True
        if confirm.lower() == "yes":
            count = delete_all_conversations(conv_dir)
            print(f"  Deleted {count} conversation(s)\n")
            if conv_id and turns:
                conv_id = conversation_id()
                title = ""
                turns = []
        else:
            print("  Cancelled\n")
        return conv_id, title, turns, show_chunks, show_prompt, True

    if cmd == "/help":
        print("  Commands:")
        print("    /chunks         Toggle chunk display")
        print("    /prompt         Toggle prompt display")
        print("    /new            Start a new conversation")
        print("    /conversations  List saved conversations")
        print("    /resume [id]    Resume a previous conversation")
        print("    /delete <id>    Delete a conversation")
        print("    /delete-all     Delete all conversations")
        print("    /help           Show this help")
        print()
        return conv_id, title, turns, show_chunks, show_prompt, True

    print(f"  Unknown command: {cmd}. Type /help for commands.\n")
    return conv_id, title, turns, show_chunks, show_prompt, True


def main() -> None:
    """Start interactive Q&A loop against persisted ChromaDB embeddings."""
    parser = argparse.ArgumentParser(description="Interactive RAG Q&A")
    parser.add_argument(
        "--config",
        default="configs/experiments/01_baseline.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--show-chunks", action="store_true", help="Show retrieved chunks",
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="Show the full prompt",
    )
    parser.add_argument(
        "--conversations-dir",
        default=DEFAULT_CONVERSATIONS_DIR,
        help="Directory for conversation history (default: data/conversations)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    collection = config.get("experiment_name", "default")
    n_chunks = pipeline.store.count()
    if n_chunks == 0:
        print(
            f"No embeddings found in collection '{collection}'. "
            "Run ingest_docs.py first."
        )
        return

    conv_dir = Path(args.conversations_dir)
    conv_id = conversation_id()
    title = ""
    turns: list[dict] = []
    show_chunks = args.show_chunks
    show_prompt = args.show_prompt

    print(f"Connected to '{collection}' ({n_chunks} chunks)")
    print("Ask questions. Type 'quit' to exit, /help for commands.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q", "bye"):
            print("Bye!")
            break

        if question.startswith("/"):
            conv_id, title, turns, show_chunks, show_prompt, _ = (
                handle_slash_command(
                    question, conv_dir, conv_id, title, turns,
                    show_chunks, show_prompt,
                )
            )
            continue

        start = time.perf_counter()
        result = pipeline.query(question, history=turns or None)
        elapsed = time.perf_counter() - start

        if result.get("rewritten_query"):
            print(f"  (rewritten: {result['rewritten_query']})")

        print(f"\nAnswer: {result['answer']}")
        print_sources(result["results"])
        print(f"({len(result['chunks'])} chunks, {elapsed:.2f}s)\n")

        if show_chunks:
            print_chunks(result["results"], result["chunks"])

        if show_prompt:
            print("--- Full Prompt ---")
            print(result["prompt"])
            print()

        # Update history
        if not title:
            title = question[:80]
        turns.append({"role": "user", "content": question})
        turns.append({"role": "assistant", "content": result["answer"]})

        # Auto-save
        save_conversation(conv_dir, conv_id, title, turns)

    # Save on exit if there's an active conversation
    if turns:
        save_conversation(conv_dir, conv_id, title, turns)
        print(f"Conversation saved ({conv_id})")


if __name__ == "__main__":
    main()
