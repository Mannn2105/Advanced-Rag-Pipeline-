"""Command-line interface for asking questions against the local RAG index."""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import sys
import pickle
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

import config
from pipeline import init_models, load_index, run_pipeline

console = Console()

def check_setup() -> None:
    problems = config.validate_runtime()
    if not config.GROQ_API_KEY:
        problems.append("GROQ_API_KEY is not configured")
    if not config.OPENROUTER_API_KEY:
        problems.append("OPENROUTER_API_KEY is not configured")
    node_file = os.path.join(config.QDRANT_DIR, "nodes.pkl")
    if not os.path.isfile(node_file):
        problems.append("No node catalogue found; run python ingest.py first")
    if problems:
        raise RuntimeError("\n".join(problems))

def show_result(result: dict) -> None:
    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
    sources = result.get("sources") or []
    if sources:
        table = Table(title="Retrieved evidence")
        table.add_column("Source")
        table.add_column("Page")
        table.add_column("Score")
        for item in sources:
            table.add_row(str(item["file"]), str(item["page"]), str(item["score"] or "—"))
        console.print(table)
    console.print(
        f"Retrieved {result.get('chunks_retrieved', 0)} chunks; "
        f"kept {result.get('chunks_after_rerank', 0)}; "
        f"mode: {result.get('source_type', 'unknown')}"
    )

def main() -> int:
    console.print(Panel("Advanced RAG Pipeline\nHybrid search • rank fusion • reranking • grounded answers"))
    try:
        check_setup()
        _, fast_llm, answer_llm = init_models()
        index = load_index()
        with open(os.path.join(config.QDRANT_DIR, "nodes.pkl"), "rb") as stream:
            nodes = pickle.load(stream)
        if not nodes:
            raise RuntimeError("The node catalogue is empty")
    except Exception as exc:
        console.print(f"[red]Startup failed: {exc}[/red]")
        return 1

    console.print("[green]Ready. Ask a question or type exit.[/green]")
    while True:
        try:
            question = console.input("[bold blue]Question:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            return 0
        if question.lower() in {"exit", "quit", "q"}:
            return 0
        if not question:
            continue
        try:
            show_result(run_pipeline(question, index, nodes, fast_llm, answer_llm))
        except Exception as exc:
            console.print(f"[red]Query failed: {type(exc).__name__}: {exc}[/red]")

if __name__ == "__main__":
    raise SystemExit(main())
