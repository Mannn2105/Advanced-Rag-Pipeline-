# TF guard — stops broken TensorFlow DLL from crashing sentence-transformers on Windows
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ============================================================
#  tests.py — Automated Test Runner
#  Runs all test questions, captures full logs + answers,
#  writes timestamped report to test_results/
#
#  Run: python tests.py
#  Run specific group: python tests.py --group R
#  Run single test: python tests.py --test R-01
# ============================================================

import sys
import time
import pickle
import argparse
from datetime import datetime
from io import StringIO

import config
from pipeline import init_models, load_index, run_pipeline

# ── Rich for terminal output ──────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

terminal = Console()
file_console = Console(file=StringIO(), highlight=False)

# ============================================================
#  TEST DEFINITIONS
# ============================================================

TESTS = [
    # ── GROUP 1: REGRESSION ───────────────────────────────────
    {
        "id": "R-01", "group": "R",
        "name": "Eco mode power",
        "question": "What is the power consumption of the Cortex-X9 in Eco mode?",
        "expected": "0.8 watts / 0.8W",
        "pass_if": "Contains '0.8'",
        "keywords_pass": ["0.8"],
        "keywords_fail": ["not found", "not contain", "unable"],
    },
    {
        "id": "R-02", "group": "R",
        "name": "Transistor count",
        "question": "How many transistors does the Cortex-X9 integrate?",
        "expected": "18 billion transistors",
        "pass_if": "Contains '18 billion'",
        "keywords_pass": ["18 billion", "18billion"],
        "keywords_fail": [],
    },
    {
        "id": "R-03", "group": "R",
        "name": "Veranox NNT",
        "question": "What is the NNT for remission in the Veranox trial?",
        "expected": "4.5",
        "pass_if": "Contains '4.5'",
        "keywords_pass": ["4.5"],
        "keywords_fail": [],
    },
    {
        "id": "R-04", "group": "R",
        "name": "P1 SLA response time",
        "question": "What is the P1 response time SLA under Platinum Support?",
        "expected": "30 minutes",
        "pass_if": "Contains '30 minutes' or '30-minute'",
        "keywords_pass": ["30 minute", "30-minute"],
        "keywords_fail": ["4 hour", "2 hour", "1 hour"],
    },

    # ── GROUP 2D: DECOMPOSITION ───────────────────────────────
    {
        "id": "D-01", "group": "D",
        "name": "Power modes + founders",
        "question": "What are the power modes of the Cortex-X9 and which uses least energy, and who founded the company that makes it?",
        "expected": "Active/Balanced/Eco wattages + Dr. Priya Nair + Marcus Chen + March 2019",
        "pass_if": "Contains wattages AND founder names",
        "keywords_pass": ["0.8", "4.2", "priya", "marcus"],
        "keywords_fail": [],
    },
    {
        "id": "D-02", "group": "D",
        "name": "AxonOS bug + gross margin",
        "question": "What bug affects AxonOS memory and what is Axon's gross margin?",
        "expected": "Bug AX-2041 memory leak + 58.4% gross margin",
        "pass_if": "Contains AX-2041 AND 58.4",
        "keywords_pass": ["ax-2041", "58.4"],
        "keywords_fail": [],
    },
    {
        "id": "D-03", "group": "D",
        "name": "Veranox remission rate + NDA date",
        "question": "What was the remission rate for Veranox-40 versus placebo, and when does Meridian plan to submit the NDA?",
        "expected": "38.4% vs 16.2% + NDA Q1 2025",
        "pass_if": "Contains remission rates AND NDA date",
        "keywords_pass": ["38.4", "16.2", "q1 2025"],
        "keywords_fail": [],
    },
    {
        "id": "D-04", "group": "D",
        "name": "SLA credits + liability cap",
        "question": "What are the SLA credit percentages for uptime failures and what is the liability cap amount?",
        "expected": "10%/25%/50% SLA tiers + CHF 840,000 cap",
        "pass_if": "Contains SLA tiers AND CHF amount",
        "keywords_pass": ["840,000", "840000", "25%", "50%"],
        "keywords_fail": [],
    },

    # ── GROUP 2F: RAG-FUSION ──────────────────────────────────
    {
        "id": "F-01", "group": "F",
        "name": "Energy efficiency vs rivals",
        "question": "How energy efficient is Axon's flagship chip compared to rivals?",
        "expected": "48 TOPS at 4.2W + competitor wattages",
        "pass_if": "Contains specific wattage numbers AND competitor names",
        "keywords_pass": ["48", "4.2", "5.8", "mediatek"],
        "keywords_fail": ["very efficient", "highly efficient"],
    },
    {
        "id": "F-02", "group": "F",
        "name": "Veranox faster onset evidence",
        "question": "What clinical evidence shows Veranox works faster than typical antidepressants?",
        "expected": "Week 2 separation (p=0.018)",
        "pass_if": "Contains Week 2 AND p-value",
        "keywords_pass": ["week 2", "0.018"],
        "keywords_fail": [],
    },
    {
        "id": "F-03", "group": "F",
        "name": "Data location + breach notification",
        "question": "What does the Nexalink contract say about data location and breach notification?",
        "expected": "AWS Frankfurt eu-central-1 + 48 hours breach notification",
        "pass_if": "Contains Frankfurt AND 48 hours",
        "keywords_pass": ["frankfurt", "48 hour"],
        "keywords_fail": [],
    },

    # ── GROUP 2C: CONTEXT COMPRESSION ────────────────────────
    {
        "id": "C-01", "group": "C",
        "name": "Cash runway",
        "question": "What is Axon's cash runway in months?",
        "expected": "~30 months ($143.2M / $4.8M burn)",
        "pass_if": "Contains ~30 months with supporting figures",
        "keywords_pass": ["30 month", "30months", "143"],
        "keywords_fail": [],
    },
    {
        "id": "C-02", "group": "C",
        "name": "Memory bandwidth",
        "question": "What is the memory bandwidth of the Cortex-X9?",
        "expected": "68.3 GB/s",
        "pass_if": "Contains 68.3",
        "keywords_pass": ["68.3"],
        "keywords_fail": [],
    },
    {
        "id": "C-03", "group": "C",
        "name": "Data export window",
        "question": "How many days does Feldberg have to export their data after contract termination?",
        "expected": "90 days + free CSV/JSON export",
        "pass_if": "Contains 90 days",
        "keywords_pass": ["90 day", "90days"],
        "keywords_fail": [],
    },

    # ── GROUP 2B: BM25 + DIVERSITY ────────────────────────────
    {
        "id": "B-01", "group": "B",
        "name": "Process node (BM25 junk filter)",
        "question": "What is the Cortex-X9 process node?",
        "expected": "TSMC 3nm N3E",
        "pass_if": "Contains TSMC and 3nm",
        "keywords_pass": ["tsmc", "3nm", "n3e"],
        "keywords_fail": [],
    },
    {
        "id": "SD-01", "group": "SD",
        "name": "Cross-document European presence",
        "question": "Compare the European presence mentioned across all three documents.",
        "expected": "Amsterdam + Stuttgart + Olten/Frankfurt + Stockholm",
        "pass_if": "Mentions locations from at least 2 of 3 documents",
        "keywords_pass": ["amsterdam", "frankfurt", "stockholm"],
        "keywords_fail": [],
    },

    # ── GROUP 2P: PROMPT QUALITY ──────────────────────────────
    {
        "id": "P-01", "group": "P",
        "name": "FP32 hallucination trap",
        "question": "What is the Cortex-X9's performance in FP32 precision?",
        "expected": "Clean refusal — FP32 not in documents",
        "pass_if": "Refuses cleanly without fabricating a number",
        "keywords_pass": ["not contain", "not found", "does not", "no information"],
        "keywords_fail": ["tops", "tflops", "33", "fp32 performance"],
    },
    {
        "id": "P-02", "group": "P",
        "name": "Stock price (out of scope)",
        "question": "What is the current stock price of Axon Systems?",
        "expected": "Not in documents / private company",
        "pass_if": "Doesn't return a real stock price",
        "keywords_pass": ["not contain", "private", "not publicly", "no information", "not found"],
        "keywords_fail": ["$", "usd", "per share"],
    },

    # ── GROUP 2SC: SELF-CRITIQUE ──────────────────────────────
    {
        "id": "SC-01", "group": "SC",
        "name": "Signature block precision",
        "question": "Who signed the Nexalink contract and on what dates?",
        "expected": "Rajesh Subramaniam Jul 28 + Klaus-Dieter Hofmann Jul 29",
        "pass_if": "Contains both names AND dates without meta-commentary",
        "keywords_pass": ["rajesh", "klaus", "july 28", "july 29", "28, 2024", "29, 2024"],
        "keywords_fail": ["based on context", "upon reviewing", "pass"],
    },
    {
        "id": "SC-02", "group": "SC",
        "name": "Memory interface + unsupported frameworks trap",
        "question": "What is the memory interface speed of the Cortex-X9 and what framework does it NOT support?",
        "expected": "LPDDR5X at 8533 MT/s + clean handling of unsupported frameworks trap",
        "pass_if": "Contains 8533 and doesn't invent unsupported frameworks",
        "keywords_pass": ["8533", "lpddr5x"],
        "keywords_fail": ["tensorflow", "keras", "caffe", "mxnet"],
    },

    # ── GROUP 3: PARALLEL EXECUTION ───────────────────────────
    {
        "id": "PAR-01", "group": "PAR",
        "name": "Dry mouth percentage (parallel CRAG YES path)",
        "question": "What percentage of Veranox-40 patients experienced dry mouth?",
        "expected": "17.9% (vs 8.2% placebo)",
        "pass_if": "Contains 17.9",
        "keywords_pass": ["17.9"],
        "keywords_fail": [],
    },
    {
        "id": "PAR-02", "group": "PAR",
        "name": "Qualcomm CEO (parallel CRAG NO path)",
        "question": "Who is the current CEO of Qualcomm?",
        "expected": "Real name from web search",
        "pass_if": "Returns a person's name (web fallback worked)",
        "keywords_pass": ["cristiano", "amon"],  # Cristiano Amon as of 2025
        "keywords_fail": ["not contain", "documents do not"],
    },
]

# ============================================================
#  AUTO-SCORING
# ============================================================

def auto_score(answer: str, test: dict) -> tuple[int, str]:
    """
    Returns (score 0-5, reason string)
    5 = all pass keywords found, no fail keywords
    4 = most pass keywords found, no fail keywords
    3 = some pass keywords found
    2 = fail keywords found
    1 = no pass keywords, no fail keywords (unclear)
    0 = crash / empty
    """
    if not answer or len(answer.strip()) < 5:
        return 0, "Empty or crash"

    answer_lower = answer.lower()
    pass_kw = test.get("keywords_pass", [])
    fail_kw = test.get("keywords_fail", [])

    # Check fail keywords first
    for kw in fail_kw:
        if kw.lower() in answer_lower:
            return 2, f"FAIL keyword found: '{kw}'"

    if not pass_kw:
        return 3, "No keywords to check — manual review needed"

    hits = [kw for kw in pass_kw if kw.lower() in answer_lower]
    ratio = len(hits) / len(pass_kw)

    if ratio == 1.0:
        return 5, f"All {len(hits)} pass keywords found"
    elif ratio >= 0.6:
        missing = [kw for kw in pass_kw if kw.lower() not in answer_lower]
        return 4, f"{len(hits)}/{len(pass_kw)} keywords found. Missing: {missing}"
    elif ratio > 0:
        missing = [kw for kw in pass_kw if kw.lower() not in answer_lower]
        return 3, f"Only {len(hits)}/{len(pass_kw)} keywords found. Missing: {missing}"
    else:
        return 1, f"No pass keywords found. Expected to contain: {pass_kw}"


# ============================================================
#  LOG CAPTURE
# ============================================================

class LogCapture:
    """Captures Rich console output to a string buffer."""
    def __init__(self):
        self.buffer = StringIO()
        self._console = Console(file=self.buffer, highlight=False, width=120)

    def print(self, *args, **kwargs):
        self._console.print(*args, **kwargs)

    def getvalue(self):
        return self.buffer.getvalue()


# ============================================================
#  REPORT WRITER
# ============================================================

def write_report(results: list, output_path: str, total_time: float):
    groups = {}
    for r in results:
        g = r["group"]
        groups.setdefault(g, []).append(r)

    group_names = {
        "R": "GROUP 1 — REGRESSION TESTS",
        "D": "GROUP 2D — QUERY DECOMPOSITION",
        "F": "GROUP 2F — RAG-FUSION",
        "C": "GROUP 2C — CONTEXT COMPRESSION",
        "B": "GROUP 2B — BM25 THRESHOLD",
        "SD": "GROUP 2SD — SOURCE DIVERSITY",
        "P": "GROUP 2P — PROMPT QUALITY",
        "SC": "GROUP 2SC — SELF-CRITIQUE",
        "PAR": "GROUP 3 — PARALLEL EXECUTION",
    }

    score_emoji = {5: "✅", 4: "🟢", 3: "🟡", 2: "🔴", 1: "❌", 0: "💥"}

    lines = []
    lines.append("=" * 80)
    lines.append("RAG PIPELINE v2 — AUTOMATED TEST RESULTS")
    lines.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total time: {total_time:.1f}s")
    lines.append("=" * 80)

    # Summary table
    lines.append("\n📊 SUMMARY\n")
    lines.append(f"{'ID':<8} {'Name':<40} {'Score':<8} {'Time':<8} {'Auto-verdict'}")
    lines.append("-" * 90)

    total_score = 0
    total_tests = 0
    for r in results:
        emoji = score_emoji.get(r["score"], "?")
        lines.append(
            f"{r['id']:<8} {r['name']:<40} {emoji} {r['score']}/5   "
            f"{r['elapsed']:.1f}s    {r['score_reason']}"
        )
        total_score += r["score"]
        total_tests += 1

    avg = total_score / max(total_tests, 1)
    lines.append("-" * 90)
    lines.append(f"{'TOTAL':<8} {total_tests} tests{'':<33} {total_score}/{total_tests*5}  avg={avg:.1f}/5")

    # Group averages
    lines.append("\n📈 GROUP AVERAGES\n")
    for group_id, group_results in groups.items():
        g_scores = [r["score"] for r in group_results]
        g_avg = sum(g_scores) / len(g_scores)
        g_name = group_names.get(group_id, group_id)
        bar = "█" * int(g_avg) + "░" * (5 - int(g_avg))
        lines.append(f"  {bar} {g_avg:.1f}/5  {g_name}")

    # Detailed results per group
    lines.append("\n\n" + "=" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 80)

    for group_id, group_results in groups.items():
        lines.append(f"\n{'='*80}")
        lines.append(group_names.get(group_id, group_id))
        lines.append(f"{'='*80}")

        for r in group_results:
            emoji = score_emoji.get(r["score"], "?")
            lines.append(f"\n{'─'*60}")
            lines.append(f"{emoji} {r['id']} | {r['name']} | Score: {r['score']}/5 | Time: {r['elapsed']:.1f}s")
            lines.append(f"{'─'*60}")
            lines.append(f"QUESTION:  {r['question']}")
            lines.append(f"EXPECTED:  {r['expected']}")
            lines.append(f"AUTO-SCORE: {r['score_reason']}")
            lines.append(f"\nANSWER:\n{r['answer']}")
            if r.get("sub_queries"):
                lines.append("\nSUB-QUERIES USED:")
                for i, q in enumerate(r["sub_queries"], 1):
                    lines.append(f"  {i}. {q}")
            lines.append(f"\nSOURCES: {r['sources']}")
            lines.append(f"source_type: {r['source_type']} | chunks_retrieved: {r['chunks_retrieved']} | after_rerank: {r['chunks_after_rerank']}")
            lines.append(f"\nPIPELINE LOGS:\n{r['logs']}")

    # Failure analysis
    failures = [r for r in results if r["score"] <= 2]
    if failures:
        lines.append("\n" + "=" * 80)
        lines.append("⚠️  FAILURES REQUIRING ATTENTION")
        lines.append("=" * 80)
        for r in failures:
            lines.append(f"\n{r['id']} — {r['name']} (score {r['score']}/5)")
            lines.append(f"  Expected: {r['expected']}")
            lines.append(f"  Got:      {r['answer'][:200]}...")
            lines.append(f"  Reason:   {r['score_reason']}")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return avg


# ============================================================
#  MAIN RUNNER
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Automated Test Runner")
    parser.add_argument("--group", type=str, help="Run only tests in this group (R/D/F/C/B/SD/P/SC/PAR)")
    parser.add_argument("--test", type=str, help="Run only this specific test ID (e.g. R-01)")
    parser.add_argument("--no-auto-score", action="store_true", help="Skip auto-scoring")
    args = parser.parse_args()

    # Filter tests
    tests_to_run = TESTS
    if args.test:
        tests_to_run = [t for t in TESTS if t["id"] == args.test.upper()]
        if not tests_to_run:
            terminal.print(f"[red]Test '{args.test}' not found.[/red]")
            sys.exit(1)
    elif args.group:
        tests_to_run = [t for t in TESTS if t["group"] == args.group.upper()]
        if not tests_to_run:
            terminal.print(f"[red]Group '{args.group}' not found.[/red]")
            sys.exit(1)

    terminal.print(Panel(
        f"[bold white]🧪 RAG Pipeline v2 — Automated Test Runner[/bold white]\n"
        f"[dim]Running {len(tests_to_run)} tests | Results saved to test_results/[/dim]",
        style="blue"
    ))

    # Validate setup
    if config.GROQ_API_KEY == "your_groq_key_here":
        terminal.print("[red]GROQ_API_KEY not set in config.py[/red]")
        sys.exit(1)
    nodes_path = os.path.join(config.QDRANT_DIR, "nodes.pkl")
    if not os.path.exists(nodes_path):
        terminal.print("[red]No index found — run: python ingest.py first[/red]")
        sys.exit(1)

    # Load models
    terminal.print("\n[cyan]Loading models and index...[/cyan]")
    embed_model, fast_llm, final_llm = init_models()
    index = load_index()
    with open(nodes_path, "rb") as f:
        nodes_store = pickle.load(f)
    terminal.print(f"[green]Ready! {len(nodes_store)} chunks indexed.[/green]\n")

    # Output directory
    os.makedirs("test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_tag = f"_{args.group}" if args.group else ""
    test_tag = f"_{args.test}" if args.test else ""
    output_path = f"test_results/results{group_tag}{test_tag}_{timestamp}.txt"

    # Run tests
    results = []
    suite_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=terminal
    ) as progress:

        task = progress.add_task("Running tests...", total=len(tests_to_run))

        for test in tests_to_run:
            progress.update(task, description=f"[cyan]{test['id']}[/cyan] {test['name']}")

            # Capture pipeline logs by temporarily redirecting config.VERBOSE output
            log_buffer = StringIO()
            log_console = Console(file=log_buffer, highlight=False, width=100)

            # Monkey-patch the pipeline's console temporarily
            import pipeline as pipeline_module
            original_console = pipeline_module.console
            pipeline_module.console = log_console

            t_start = time.time()
            answer = ""
            sub_queries = []
            sources = []
            source_type = "unknown"
            chunks_retrieved = 0
            chunks_after_rerank = 0
            error = None

            try:
                result = run_pipeline(
                    query=test["question"],
                    index=index,
                    nodes_store=nodes_store,
                    fast_llm=fast_llm,
                    final_llm=final_llm
                )
                answer = result["answer"]
                sub_queries = result.get("sub_queries", [])
                sources = result.get("sources", [])
                source_type = result.get("source_type", "unknown")
                chunks_retrieved = result.get("chunks_retrieved", 0)
                chunks_after_rerank = result.get("chunks_after_rerank", 0)
            except Exception as e:
                answer = f"PIPELINE ERROR: {e}"
                error = str(e)
            finally:
                pipeline_module.console = original_console

            elapsed = time.time() - t_start
            logs = log_buffer.getvalue()

            # Auto-score
            score, score_reason = auto_score(answer, test)

            # Format sources for report
            sources_str = ", ".join([
                f"{s['file']} (score: {s['score']})"
                for s in sources
            ]) if sources else source_type

            results.append({
                **test,
                "answer": answer,
                "sub_queries": sub_queries,
                "sources": sources_str,
                "source_type": source_type,
                "chunks_retrieved": chunks_retrieved,
                "chunks_after_rerank": chunks_after_rerank,
                "elapsed": elapsed,
                "logs": logs,
                "score": score,
                "score_reason": score_reason,
                "error": error,
            })

            # Live terminal feedback
            score_colors = {5: "green", 4: "green", 3: "yellow", 2: "red", 1: "red", 0: "red"}
            score_emojis = {5: "✅", 4: "🟢", 3: "🟡", 2: "🔴", 1: "❌", 0: "💥"}
            color = score_colors.get(score, "white")
            emoji = score_emojis.get(score, "?")
            terminal.print(
                f"  {emoji} [{color}]{test['id']}[/{color}] {test['name']:<38} "
                f"[dim]{elapsed:.1f}s[/dim]  score=[bold {color}]{score}/5[/bold {color}]"
            )
            progress.advance(task)

    total_time = time.time() - suite_start

    # Write report
    avg = write_report(results, output_path, total_time)

    # Final summary table in terminal
    terminal.print()
    summary_table = Table(title="📊 Final Scores", show_header=True, header_style="bold cyan")
    summary_table.add_column("Group", style="white")
    summary_table.add_column("Tests", style="dim")
    summary_table.add_column("Avg Score", style="yellow")
    summary_table.add_column("Status")

    group_ids_ordered = ["R", "D", "F", "C", "B", "SD", "P", "SC", "PAR"]
    group_labels = {
        "R": "Regression", "D": "Decomposition", "F": "RAG-Fusion",
        "C": "Compression", "B": "BM25 Filter", "SD": "Source Diversity",
        "P": "Prompt Quality", "SC": "Self-Critique", "PAR": "Parallel"
    }
    targets = {"R": 5.0, "D": 4.0, "F": 4.0, "C": 4.0, "B": 4.0,
               "SD": 4.0, "P": 4.0, "SC": 4.0, "PAR": 3.5}

    for gid in group_ids_ordered:
        group_res = [r for r in results if r["group"] == gid]
        if not group_res:
            continue
        g_avg = sum(r["score"] for r in group_res) / len(group_res)
        target = targets.get(gid, 4.0)
        status = "✅ Pass" if g_avg >= target else f"⚠️  Target {target}"
        color = "green" if g_avg >= target else "yellow"
        summary_table.add_row(
            group_labels.get(gid, gid),
            str(len(group_res)),
            f"{g_avg:.1f}/5",
            f"[{color}]{status}[/{color}]"
        )

    terminal.print(summary_table)
    terminal.print(f"\n[bold]Overall: {sum(r['score'] for r in results)}/{len(results)*5} "
                   f"(avg {avg:.1f}/5) in {total_time:.1f}s[/bold]")
    terminal.print(f"\n[green]Full report saved to:[/green] [bold]{output_path}[/bold]")


if __name__ == "__main__":
    main()