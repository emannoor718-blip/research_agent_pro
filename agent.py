# agent.py
import sqlite3
import operator
from typing import TypedDict, Annotated

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage

from tools import get_tools

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def build_agent():
    tools     = get_tools()
    llm       = ChatOllama(model="qwen2.5:3b", temperature=0).bind_tools(tools)
    tool_node = ToolNode(tools)

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "hitl_check"
        return END

    def call_model(state: AgentState):
        system = SystemMessage(content=(
            "You are a research assistant. You MUST ALWAYS use tools to answer questions. "
            "NEVER answer from your own knowledge. "
            "For ANY question, ALWAYS call web_search first, then use arxiv_search or "
            "wikipedia_search if needed. "
            "Only provide a final answer AFTER using at least one tool."
        ))
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")

    graph.add_conditional_edges("agent", should_continue, {
        "hitl_check": "tools",
        END: END
    })
    graph.add_edge("tools", "agent")

    conn   = sqlite3.connect("checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["tools"]
    )