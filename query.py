# TF guard — stops broken TensorFlow DLL from crashing sentence-transformers on Windows
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ============================================================
#  query.py — Interactive query interface
#  Run: python query.py
# ============================================================

import sys
import pickle

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

import config
from pipeline import init_models, load_index, run_pipeline

console = Console()


def check_setup():
    errors = []
    if config.GROQ_API_KEY == "your_groq_key_here":
        errors.append("GROQ_API_KEY not set in config.py")
    if config.OPENROUTER_API_KEY == "your_openrouter_key_here":
        errors.append("OPENROUTER_API_KEY not set in config.py")
    if config.COHERE_API_KEY == "your_cohere_key_here":
        errors.append("COHERE_API_KEY not set in config.py")
    nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
    if not os.path.exists(nodes_path):
        errors.append("No index found — run: python ingest.py first")
    if errors:
        console.print("\n[bold red]Setup issues found:[/bold red]")
        for e in errors:
            console.print(f"  ✗ {e}")
        sys.exit(1)


def print_result(result: dict):
    # Answer panel
    console.print()
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]Answer[/bold green]",
        border_style="green"
    ))

    # Sources table
    if result["sources"]:
        table = Table(title="📚 Sources", show_header=True, header_style="bold cyan")
        table.add_column("File", style="white")
        table.add_column("Page", style="dim")
        table.add_column("Relevance Score", style="yellow")
        for s in result["sources"]:
            table.add_row(s["file"], str(s["page"]), str(s["score"]) if s["score"] else "n/a")
        console.print(table)
    elif result.get("source_type", "").startswith("web"):
        console.print("[yellow]📡 Answer supplemented with web search[/yellow]")

    # Sub-queries used
    if result.get("sub_queries") and len(result["sub_queries"]) > 1:
        console.print("\n[dim]Sub-queries used:[/dim]")
        for i, q in enumerate(result["sub_queries"], 1):
            console.print(f"  [dim]{i}. {q}[/dim]")

    # Stats
    console.print(
        f"\n[dim]Chunks retrieved: {result['chunks_retrieved']} → "
        f"after rerank: {result['chunks_after_rerank']} | "
        f"source: {result.get('source_type', '?')}[/dim]"
    )


def main():
    console.print(Panel(
        "[bold white]🧠 Max-Accuracy RAG v2[/bold white]\n"
        "[dim]Decompose • HyDE • Fusion • Rerank • Compress • CRAG • OpenRouter[/dim]",
        style="blue"
    ))

    check_setup()

    console.print("\n[cyan]Loading models and index...[/cyan]")
    embed_model, fast_llm, final_llm = init_models()
    index = load_index()

    with open(os.path.join(config.QDRANT_DIR, "nodes.pkl"), "rb") as f:
        nodes_store = pickle.load(f)

    console.print(f"[green]Ready! Index loaded with {len(nodes_store)} chunks.[/green]")
    console.print("[dim]Type 'exit' to stop. Type 'help' for tips.[/dim]\n")

    while True:
        try:
            query = console.input("[bold blue]Question:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            console.print("[dim]Bye![/dim]")
            break
        if query.lower() == "help":
            console.print(Panel(
                "[bold]Tips:[/bold]\n"
                "• Ask specific questions for best precision\n"
                "• Complex multi-part questions work well\n"
                "• VERBOSE=True in config.py shows every pipeline step\n"
                "• Add docs to /docs and re-run ingest.py anytime",
                title="Help", style="dim"
            ))
            continue

        try:
            result = run_pipeline(
                query=query,
                index=index,
                nodes_store=nodes_store,
                fast_llm=fast_llm,
                final_llm=final_llm
            )
            print_result(result)
        except Exception as e:
            console.print(f"\n[red]Pipeline error: {e}[/red]")
            console.print("[dim]Check your API keys in config.py and try again.[/dim]")

        console.print("\n" + "─" * 60 + "\n")


if __name__ == "__main__":
    main()