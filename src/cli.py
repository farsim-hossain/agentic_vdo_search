import typer
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from src.agent.router import AgenticRouter

app = typer.Typer(help="Rate-Limited Agentic Video Analytics CLI")
console = Console()

def verbose_print(msg: str):
    console.print(f"[bold blue][INFO][/bold blue] {msg}")

@app.command()
def index(
    video_path: str = typer.Argument(..., help="Path to input video file (.mp4, .mkv, .avi)")
):
    """Ingest video clip, extract keyframes, and build CPU frame vector index."""
    console.print(f"[bold green]Ingesting video:[bold green] {video_path}")
    router = AgenticRouter()
    video_id = router.ensure_indexed(video_path, verbose_callback=verbose_print)
    console.print(Panel(f"Successfully indexed video '[bold yellow]{video_id}[/bold yellow]'", title="Index Status", border_style="green"))

@app.command()
def ask(
    video_path: str = typer.Argument(..., help="Path to input video file"),
    question: str = typer.Argument(..., help="Question about video content")
):
    """Ask a question about the video content."""
    console.print(Panel(f"Question: [bold cyan]{question}[/bold cyan]", title="Video Analytics Query", border_style="cyan"))
    router = AgenticRouter()
    
    result = router.answer_query(video_path, question, verbose_callback=verbose_print)

    answer_text = result.get("answer", "No response generated.")
    source = result.get("source", "unknown")
    shot_id = result.get("shot_id", "N/A")

    console.print("\n")
    console.print(Panel(answer_text, title=f"Answer (Source: {source} | Shot: {shot_id})", border_style="bold green"))

    if "observations" in result:
        table = Table(title="Visual Observations Detail", show_header=True, header_style="bold magenta")
        table.add_column("Time Range", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Visible Objects", style="yellow")

        for ev in result["observations"].get("events", []):
            table.add_row(
                f"{ev.get('start_time')}-{ev.get('end_time')}",
                ev.get("description", ""),
                ", ".join(ev.get("visible_objects", []))
            )
        console.print(table)

@app.command()
def summary(
    video_path: str = typer.Argument(..., help="Path to input video file")
):
    """Summarize overall video events using cached visual observations."""
    console.print(f"[bold green]Generating summary for video:[bold green] {video_path}")
    router = AgenticRouter()
    sum_text = router.summarize_video(video_path, verbose_callback=verbose_print)
    console.print(Panel(sum_text, title="Video Event Summary", border_style="bold yellow"))

if __name__ == "__main__":
    app()
