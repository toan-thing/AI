from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from agent.utils.state import AgentState
from agent.utils.nodes import Parse, parse_runnable, Reason, reason_runnable, Resolve_products
from agent.utils.tools import tool_node



builder = StateGraph(AgentState)

builder.add_node(
    "parse",
    Parse(parse_runnable)
)

builder.add_node(
    "resolve",
    Resolve_products
)

builder.add_node(
    "reason",
    Reason(reason_runnable)
)

builder.add_node(
    "tools",
    tool_node
)

builder.add_edge(START, "parse")
builder.add_edge("parse", "resolve")
builder.add_edge("resolve", "reason")
builder.add_conditional_edges(
    "reason",
    tools_condition
)
builder.add_edge("tools", "reason")

graph = builder.compile()
