"""Interactive Q&A with your RAG pipeline.

Supports conversational follow-ups with query rewriting,
persistent conversation history, and slash commands.

Usage:
    uv run python scripts/ask.py --config configs/experiments/arxiv_auto_scaling.yaml
    uv run python scripts/ask.py --config configs/experiments/01_baseline.yaml --show-chunks
"""

from __future__ import annotations

import argparse
import atexit
import readline
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.markdown import Markdown
from src.generate.prompt_templates import render_compact, render_title
from src.pipeline import RAGPipeline, load_config

DEFAULT_CONVERSATIONS_DIR = "data/conversations"
METADATA_FILENAME = "arxiv_metadata.yaml"
HISTORY_FILE = Path("data/conversations/.ask_history")
MAX_RETRIES = 3
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

console = Console()


def setup_readline() -> None:
    """Configure readline for interactive prompt editing and history.

    Sets up persistent history across sessions and custom key bindings:
    - Up/Down: navigate prompt history
    - Ctrl+R: reverse search through history
    - Ctrl+A: move cursor to beginning of line
    - Ctrl+E: move cursor to end of line
    - Ctrl+W: move cursor forward one word
    - Ctrl+B: move cursor backward one word
    """
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, HISTORY_FILE)

    # Detect libedit (macOS) vs GNU readline (Linux)
    if "libedit" in readline.__doc__:
        readline.parse_and_bind("bind ^W forward-word")
        readline.parse_and_bind("bind ^B backward-word")
    else:
        readline.parse_and_bind(r'"\C-w": forward-word')
        readline.parse_and_bind(r'"\C-b": backward-word')


class Spinner:
    """Animated terminal spinner that runs in a background thread.

    Args:
        message: Text to display alongside the spinner animation.
    """

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the spinner animation in a daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def _spin(self) -> None:
        idx = 0
        while not self._stop_event.is_set():
            frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
            sys.stderr.write(f"\r{frame} {self._message}...")
            sys.stderr.flush()
            idx += 1
            self._stop_event.wait(0.08)


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


def load_arxiv_metadata(source_paths: list[str]) -> dict[str, dict]:
    """Load arXiv metadata from YAML sidecars found alongside source files.

    Searches for arxiv_metadata.yaml in the directories containing the source
    files. Returns a dict mapping pdf_file name to paper metadata.

    Args:
        source_paths: List of source file paths from retrieval results.

    Returns:
        Dict mapping filename to paper metadata dict.
    """
    metadata = {}
    searched_dirs: set[str] = set()
    for source in source_paths:
        parent = str(Path(source).parent)
        if parent in searched_dirs:
            continue
        searched_dirs.add(parent)
        meta_path = Path(parent) / METADATA_FILENAME
        if meta_path.exists():
            with open(meta_path) as f:
                data = yaml.safe_load(f)
            for paper in data.get("papers", []):
                pdf_file = paper.get("pdf_file", "")
                if pdf_file:
                    metadata[pdf_file] = paper
    return metadata


def print_papers(last_results: list[dict]) -> None:
    """Print title and abstract for papers cited in the last answer.

    Args:
        last_results: Retrieval results from the most recent query.
    """
    if not last_results:
        print("  No results yet — ask a question first.\n")
        return

    sources = []
    seen: set[str] = set()
    for r in last_results:
        source = r.get("metadata", {}).get("source", "")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)

    metadata = load_arxiv_metadata(sources)
    if not metadata:
        print("  No arxiv_metadata.yaml found alongside source files.\n")
        return

    printed = 0
    for source in sources:
        filename = Path(source).name
        paper = metadata.get(filename)
        if not paper:
            continue
        printed += 1
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "No abstract available.")
        authors = ", ".join(paper.get("authors", []))
        arxiv_id = paper.get("arxiv_id", "")
        print(f"  [{printed}] {title}")
        if authors:
            print(f"      Authors: {authors}")
        if arxiv_id:
            print(f"      https://arxiv.org/abs/{arxiv_id}")
        print(f"      Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        print()

    if printed == 0:
        print("  No metadata found for the cited sources.\n")


def print_sources(results: list[dict]) -> None:
    """Print unique sources as clickable file:// URLs.

    Args:
        results: Retrieval results with metadata containing source paths.
    """
    seen = set()
    sources = []
    for r in results:
        source = r.get("metadata", {}).get("source", "")
        if source and source not in seen:
            seen.add(source)
            sources.append(source)
    if sources:
        print("\nSources:")
        for source in sources:
            abs_path = Path(source).resolve()
            print(f"  • {abs_path.name}  file://{abs_path}")


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


def query_with_retry(
    pipeline: RAGPipeline,
    question: str,
    turns: list[dict],
    show_debug: bool,
) -> dict:
    """Run a pipeline query with automatic retry on transient errors.

    Retries up to MAX_RETRIES times with exponential backoff on connection
    and server errors. Shows stack traces when debug mode is enabled.

    Args:
        pipeline: The RAG pipeline instance.
        question: The user's question.
        turns: Conversation history for query rewriting.
        show_debug: Whether to print stack traces on error.

    Returns:
        Query result dict from the pipeline.

    Raises:
        Exception: Re-raises the last exception after all retries are exhausted.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return pipeline.query(question, history=turns or None)
        except (ConnectionError, OSError, RuntimeError) as e:
            last_error = e
            if show_debug:
                traceback.print_exc()
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  Retry {attempt}/{MAX_RETRIES} in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}")
    raise last_error  # type: ignore[misc]


def generate_title(pipeline: RAGPipeline, question: str, answer: str) -> str:
    """Use the LLM to generate a short conversation title.

    Args:
        pipeline: The RAG pipeline instance (for LLM access).
        question: The user's first question.
        answer: The assistant's first answer.

    Returns:
        A short title string, or first 80 chars of the question on failure.
    """
    try:
        prompt = render_title(question, answer)
        title = pipeline.llm.generate(prompt).strip().strip('"\'')
        return title[:80] if title else question[:80]
    except Exception:
        return question[:80]


def handle_slash_command(
    command: str, conv_dir: Path, conv_id: str, title: str, turns: list[dict],
    show_chunks: bool, show_prompt: bool, show_debug: bool,
    last_results: list[dict] | None = None,
    pipeline: RAGPipeline | None = None,
    last_question: str | None = None,
) -> tuple[str, str, list[dict], bool, bool, bool, bool, str | None]:
    """Handle a slash command and return updated state.

    Args:
        command: The slash command string (e.g., "/conversations").
        conv_dir: Directory containing conversation files.
        conv_id: Current conversation ID.
        title: Current conversation title.
        turns: Current conversation turns.
        show_chunks: Current show-chunks toggle state.
        show_prompt: Current show-prompt toggle state.
        show_debug: Current show-debug toggle state.
        last_results: Retrieval results from the most recent query.
        pipeline: RAG pipeline for /regenerate and /compact.
        last_question: The last question asked (for /regenerate).

    Returns:
        Tuple of (conv_id, title, turns, show_chunks, show_prompt,
        show_debug, handled, regenerate_question).
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""
    no_regen: str | None = None

    if cmd == "/chunks":
        show_chunks = not show_chunks
        print(f"  show-chunks: {'ON' if show_chunks else 'OFF'}\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/prompt":
        show_prompt = not show_prompt
        print(f"  show-prompt: {'ON' if show_prompt else 'OFF'}\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/debug":
        show_debug = not show_debug
        print(f"  debug: {'ON' if show_debug else 'OFF'}\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/new":
        if turns:
            save_conversation(conv_dir, conv_id, title, turns)
            print(f"  Saved conversation '{title}' ({conv_id})")
        conv_id = conversation_id()
        title = ""
        turns = []
        print("  Started new conversation\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/conversations":
        convos = list_conversations(conv_dir)
        if not convos:
            print("  No saved conversations\n")
        else:
            print(f"  {len(convos)} saved conversation(s):")
            for c in convos:
                print(f"    {c['id']}  {c['title']}  ({c['turns']} turns)")
            print()
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/resume":
        if arg:
            target_id = arg
        else:
            convos = list_conversations(conv_dir)
            if not convos:
                print("  No saved conversations\n")
                return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
            print("  Saved conversations:")
            for i, c in enumerate(convos, 1):
                print(f"    {i}. {c['id']}  {c['title']}  ({c['turns']} turns)")
            try:
                choice = input("  Enter number or ID: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
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
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/delete":
        if not arg:
            print("  Usage: /delete <conversation-id>\n")
            return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
        if delete_conversation(conv_dir, arg):
            print(f"  Deleted conversation '{arg}'\n")
        else:
            print(f"  Conversation '{arg}' not found\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/delete-all":
        try:
            confirm = input("  Delete all conversations? (yes/no): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
        if confirm.lower() == "yes":
            count = delete_all_conversations(conv_dir)
            print(f"  Deleted {count} conversation(s)\n")
            if conv_id and turns:
                conv_id = conversation_id()
                title = ""
                turns = []
        else:
            print("  Cancelled\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/papers":
        print_papers(last_results or [])
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/regenerate":
        if not last_question:
            print("  No previous question to regenerate.\n")
            return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
        # Remove the last Q&A pair from turns before regenerating
        if len(turns) >= 2 and turns[-2]["role"] == "user":
            turns = turns[:-2]
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, last_question

    if cmd == "/compact":
        if not pipeline or len(turns) < 4:
            print("  Not enough conversation history to compact.\n")
            return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
        spinner = Spinner("Compacting")
        spinner.start()
        try:
            prompt = render_compact(turns)
            summary = pipeline.llm.generate(prompt).strip()
        except Exception as e:
            spinner.stop()
            print(f"  Compact failed: {e}\n")
            return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen
        spinner.stop()
        old_count = len(turns)
        turns = [
            {"role": "user", "content": "[Previous conversation summary]"},
            {"role": "assistant", "content": summary},
        ]
        print(f"  Compacted {old_count} turns → 2 (summary)\n")
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    if cmd == "/help":
        print("  Commands:")
        print("    /chunks         Toggle chunk display")
        print("    /prompt         Toggle prompt display")
        print("    /debug          Toggle debug traces")
        print("    /papers         Show title & abstract of cited papers")
        print("    /regenerate     Re-generate last answer")
        print("    /compact        Compact conversation history")
        print("    /new            Start a new conversation")
        print("    /conversations  List saved conversations")
        print("    /resume [id]    Resume a previous conversation")
        print("    /delete <id>    Delete a conversation")
        print("    /delete-all     Delete all conversations")
        print("    /help           Show this help")
        print()
        return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen

    print(f"  Unknown command: {cmd}. Type /help for commands.\n")
    return conv_id, title, turns, show_chunks, show_prompt, show_debug, True, no_regen


def main() -> None:
    """Start interactive Q&A loop against persisted ChromaDB embeddings."""
    setup_readline()
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
    last_results: list[dict] = []
    last_question: str | None = None
    show_chunks = args.show_chunks
    show_prompt = args.show_prompt
    show_debug = False

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
            conv_id, title, turns, show_chunks, show_prompt, show_debug, _, regen_q = (
                handle_slash_command(
                    question, conv_dir, conv_id, title, turns,
                    show_chunks, show_prompt, show_debug, last_results,
                    pipeline, last_question,
                )
            )
            if regen_q:
                question = regen_q
            else:
                continue

        spinner = Spinner()
        spinner.start()
        try:
            start = time.perf_counter()
            result = query_with_retry(pipeline, question, turns, show_debug)
            elapsed = time.perf_counter() - start
        except KeyboardInterrupt:
            spinner.stop()
            print("\n  Interrupted.\n")
            continue
        except Exception as e:
            spinner.stop()
            if show_debug:
                traceback.print_exc()
            print(f"\n  Error: {e}\n")
            continue
        spinner.stop()

        if result.get("rewritten_query"):
            print(f"  (rewritten: {result['rewritten_query']})")

        last_results = result["results"]
        last_question = question

        print()
        console.print(Markdown(result["answer"]))
        print_sources(last_results)
        print(f"\n({len(result['chunks'])} chunks, {elapsed:.2f}s)\n")

        if show_chunks:
            print_chunks(result["results"], result["chunks"])

        if show_prompt:
            print("--- Full Prompt ---")
            print(result["prompt"])
            print()

        # Generate title from LLM on the first exchange
        if not title:
            title = generate_title(pipeline, question, result["answer"])

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
