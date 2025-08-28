from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import ResearchState
from clarification_agent import ClarificationAgent
from research_agent import ResearchAgent
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def route_next_step(state: ResearchState) -> Literal["clarification", "research", "end"]:
    """Determine the next step in the workflow"""
    
    current_step = state.get("current_step", "clarification")
    clarification_complete = state.get("clarification_complete", False)
    
    if current_step == "completed":
        return "end"
    elif current_step == "error":
        return "end"
    elif clarification_complete and current_step != "completed":
        return "research"
    else:
        return "clarification"


def create_research_graph():
    """Create the main research agent graph"""
    
    # Initialize the agents
    clarification_agent = ClarificationAgent()
    research_agent = ResearchAgent()
    
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_agent)
    workflow.add_node("research", research_agent)
    
    # Set the entry point
    workflow.set_entry_point("clarification")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "clarification",
        route_next_step,
        {
            "clarification": "clarification",
            "research": "research",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "research",
        route_next_step,
        {
            "research": "research",
            "end": END
        }
    )
    
    # Compile the graph with memory for conversation persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


def run_research_agent():
    """Interactive function to run the research agent"""
    
    print("=" * 60)
    print("Research Agent - Interactive Session")
    print("=" * 60)
    print("\nI'll help you conduct thorough research on any topic.")
    print("Type 'quit' to exit the session.\n")
    
    # Create the graph
    app = create_research_graph()
    
    # Initialize state
    config = {"configurable": {"thread_id": "research_session_1"}}
    
    # Initial empty state to trigger the greeting
    state = app.invoke(
        {"messages": [], "iteration_count": 0},
        config
    )
    
    # Print the initial AI message
    if state.get("messages"):
        last_message = state["messages"][-1]
        print(f"Assistant: {last_message.content}\n")
    
    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThank you for using the Research Agent. Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to state
        user_message = HumanMessage(content=user_input)
        
        # Invoke the graph with the user message
        state = app.invoke(
            {"messages": [user_message]},
            config
        )
        
        # Print the assistant's response
        if state.get("messages"):
            last_message = state["messages"][-1]
            print(f"\nAssistant: {last_message.content}\n")
        
        # Check if research is complete
        if state.get("current_step") == "completed":
            print("\n" + "=" * 60)
            print("RESEARCH COMPLETE")
            print("=" * 60)
            if state.get("research_report"):
                print("\n" + state["research_report"])
            print("\n" + "=" * 60)
            
            # Ask if user wants to continue with a new research
            continue_research = input("\nWould you like to research another topic? (yes/no): ").strip().lower()
            if continue_research != 'yes':
                print("\nThank you for using the Research Agent. Goodbye!")
                break
            else:
                # Reset for new research
                config = {"configurable": {"thread_id": f"research_session_{os.urandom(4).hex()}"}}
                state = app.invoke(
                    {"messages": [], "iteration_count": 0},
                    config
                )
                if state.get("messages"):
                    last_message = state["messages"][-1]
                    print(f"\nAssistant: {last_message.content}\n")


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please copy .env.example to .env and fill in your API keys.")
    else:
        run_research_agent()