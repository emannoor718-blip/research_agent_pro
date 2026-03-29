# hitl.py
import json
from rich.console import Console
from rich.panel   import Panel
from rich.prompt  import Prompt
from rich.table   import Table

console = Console()

def display_tool_call(tool_call: dict):
    """Pretty-print the pending tool call for human review."""
    table = Table(title="⏸  HITL — Pending Tool Call", border_style="yellow")
    table.add_column("Field", style="cyan",  width=15)
    table.add_column("Value", style="white")

    table.add_row("Tool",    tool_call.get("name", "unknown"))
    table.add_row("Args",    json.dumps(tool_call.get("args", {}), indent=2))
    table.add_row("Call ID", tool_call.get("id", "N/A"))
    console.print(table)

def get_human_decision(tool_call: dict) -> tuple[str, dict]:
    """
    Returns: (decision, possibly_edited_tool_call)
    decision: "approve" | "edit" | "reject"
    """
    console.print(Panel(
        "[bold yellow]Review the tool call above.[/]\n"
        "[green]approve[/] — run as-is\n"
        "[blue]edit[/]    — modify args before running\n"
        "[red]reject[/]  — skip this tool call",
        title="Your Decision"
    ))

    choice = Prompt.ask(
        "Decision",
        choices=["approve", "edit", "reject"],
        default="approve"
    )

    if choice == "edit":
        console.print("[blue]Enter new args as JSON (e.g. {\"query\": \"new topic\"}):[/]")
        raw = input("New args > ").strip()
        try:
            new_args = json.loads(raw)
            tool_call = {**tool_call, "args": new_args}
            console.print("[green]✓ Args updated.[/]")
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON — keeping original args.[/]")

    elif choice == "reject":
        reason = Prompt.ask("Rejection reason", default="Not relevant")
        return "reject", {"reason": reason}

    return choice, tool_call