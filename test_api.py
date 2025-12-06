import os
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# --- 1. DEFINE THE STATE ---
# This acts as the "memory" passed between nodes
class AgentState(TypedDict):
    bool1: bool
    bool2: bool
    step_log: list[str]

# --- 2. DEFINE THE NODES ---
def node_1(state: AgentState):
    print("--- Executing Node 1 ---")
    # In a real agent, you might use OpenAI here to decide the booleans
    # e.g., result = llm.invoke(...)
    return {"step_log": ["Visited Node 1"]}

def node_2(state: AgentState):
    print("--- Executing Node 2 ---")
    return {"step_log": ["Visited Node 2"]}

def node_3(state: AgentState):
    print("--- Executing Node 3 ---")
    return {"step_log": ["Visited Node 3"]}

# --- 3. DEFINE THE LOGIC (ROUTER) ---
def decide_next_step(state: AgentState) -> Literal["node_2", "node_3", END]:
    b1 = state.get("bool1")
    b2 = state.get("bool2")
    
    # Logic A: Node 1 -> Node 2
    if b1 is True and b2 is False:
        return "node_2"
    
    # Logic B: Node 1 -> Node 3
    elif b1 is True and b2 is True:
        return "node_3"
    
    # Fallback (e.g. if bool1 is False)
    return END

# --- 4. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("node_1", node_1)
workflow.add_node("node_2", node_2)
workflow.add_node("node_3", node_3)

# Set Entry Point
workflow.set_entry_point("node_1")

# Add Conditional Edges (The "Router" from Node 1)
workflow.add_conditional_edges(
    "node_1",          # Start at Node 1
    decide_next_step,  # Run this function to decide
    {                  # Map output to actual node names
        "node_2": "node_2",
        "node_3": "node_3",
        END: END
    }
)

# Add Normal Edge (Node 2 -> Node 3 always)
workflow.add_edge("node_2", "node_3")

# Add Terminal Edge (Node 3 -> End)
workflow.add_edge("node_3", END)

# Compile
app = workflow.compile()

# --- 5. TEST RUNS ---

print("\n>>> TEST CASE 1: (True, False) -> Should go 1 -> 2 -> 3")
app.invoke({"bool1": True, "bool2": False})

print("\n>>> TEST CASE 2: (True, True) -> Should go 1 -> 3")
app.invoke({"bool1": True, "bool2": True})