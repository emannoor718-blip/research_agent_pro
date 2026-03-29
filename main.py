# main.py
from rich.console import Console
from rich.panel   import Panel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent import build_agent
from hitl  import display_tool_call, get_human_decision

console = Console()
import time
config  = {"configurable": {"thread_id": f"session-{int(time.time())}"}}
def run():
    graph = build_agent()
    console.print(Panel.fit(
        "[bold cyan]Research Agent Pro[/]\n[dim]HITL-Powered · Type 'quit' to exit[/]",
        border_style="cyan"
    ))

    while True:
        query = console.input("\n[bold green]You >[/] ").strip()
        if query.lower() in ("quit", "exit"):
            break

        # ── Stream until interrupt ─────────────────────────────
        state  = {"messages": [HumanMessage(content=query)]}
        events = graph.stream(state, config, stream_mode="values")

        for event in events:
            last_msg = event["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                console.print(f"\n[dim cyan]Agent:[/] {last_msg.content}")

        # ── HITL Loop ──────────────────────────────────────────
        while True:
            current = graph.get_state(config)

            if not current.next:
                break

            last = current.values["messages"][-1]
            if not (isinstance(last, AIMessage) and last.tool_calls):
                break

            for tool_call in last.tool_calls:
                console.rule("[yellow]HITL Gate[/]")
                display_tool_call(tool_call)
                decision, result = get_human_decision(tool_call)

                if decision == "reject":
                    skip_msg = ToolMessage(
                        content=f"Tool call rejected: {result['reason']}",
                        tool_call_id=tool_call["id"]
                    )
                    graph.update_state(config, {"messages": [skip_msg]})
                    console.print(f"[red]✗ Rejected:[/] {result['reason']}")

                elif decision == "edit":
                    updated_tc = {**tool_call, "args": result["args"]}
                    updated_ai = AIMessage(
                        content=last.content,
                        tool_calls=[updated_tc]
                    )
                    graph.update_state(config, {"messages": [updated_ai]})
                    console.print("[blue]✎ Edited args applied.[/]")

                else:
                    console.print("[green]✓ Approved.[/]")

            # Resume from checkpoint
            resume_events = graph.stream(None, config, stream_mode="values")
            for event in resume_events:
                msg = event["messages"][-1]
                if isinstance(msg, AIMessage) and msg.content:
                    console.print(f"\n[dim cyan]Agent:[/] {msg.content}")

        # ── Final answer ───────────────────────────────────────
        final = graph.get_state(config).values["messages"][-1]
        if isinstance(final, AIMessage) and final.content:
            console.print(Panel(
                final.content,
                title="[bold green]Final Answer[/]",
                border_style="green"
            ))

if __name__ == "__main__":
    run()