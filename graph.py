from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from clarification_agent import ClarificationAgent
from research_agent import ResearchAgent
import os
from dotenv import load_dotenv


load_dotenv()


def route_agent(state: AgentState) -> Literal["clarification", "research", "end"]:
    """Route to the appropriate agent based on the current phase"""
    current_phase = state.get("current_phase", "clarification")
    
    if current_phase == "clarification":
        if state.get("clarification_complete", False):
            return "research"
        return "clarification"
    elif current_phase == "research":
        return "research"
    else:
        return "end"


def create_research_graph():
    """Create the main research agent graph"""
    
    graph = StateGraph(AgentState)
    
    clarification_agent = ClarificationAgent()
    research_agent = ResearchAgent()
    
    graph.add_node("clarification", clarification_agent)
    graph.add_node("research", research_agent)
    
    graph.set_entry_point("clarification")
    
    graph.add_conditional_edges(
        "clarification",
        route_agent,
        {
            "clarification": "clarification",
            "research": "research",
            "end": END
        }
    )
    
    graph.add_edge("research", END)
    
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    
    return app


def run_research_agent():
    """Run the research agent in terminal mode"""
    app = create_research_graph()
    
    print("=" * 50)
    print("Research Agent Started")
    print("=" * 50)
    print()
    
    config = {"configurable": {"thread_id": "research_session_1"}}
    
    state = {"messages": [], "current_phase": "clarification"}
    
    while True:
        result = app.invoke(state, config)
        
        last_message = result["messages"][-1] if result.get("messages") else None
        
        if last_message and last_message.get("role") == "assistant":
            print(f"\nAssistant: {last_message['content']}\n")
        
        if result.get("current_phase") == "complete":
            print("\n" + "=" * 50)
            print("Research Complete!")
            print("=" * 50)
            break
        
        if result.get("current_phase") == "research" and not result.get("clarification_complete"):
            state = result
            continue
            
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nExiting research agent. Goodbye!")
            break
        
        state = result
        state["messages"].append({"role": "user", "content": user_input})


if __name__ == "__main__":
    run_research_agent()