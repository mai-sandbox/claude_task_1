"""Example usage of the research agent"""

from graph import create_research_graph
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def example_research_session():
    """Demonstrate a complete research session"""
    
    print("=" * 60)
    print("Example Research Session")
    print("=" * 60)
    
    # Create the graph
    app = create_research_graph()
    
    # Configuration for the session
    config = {"configurable": {"thread_id": "example_session_1"}}
    
    # Simulate a research conversation
    conversation_flow = [
        {
            "user": None,  # Initial state
            "expected": "greeting"
        },
        {
            "user": "I want to research the latest developments in quantum computing",
            "expected": "clarification"
        },
        {
            "user": "I'm particularly interested in quantum supremacy achievements, recent breakthroughs from major tech companies, and practical applications that might emerge in the next 5 years",
            "expected": "clarification"
        },
        {
            "user": "Focus on developments from 2023-2024, and include both hardware and software advances. A structured report with sections would be great.",
            "expected": "brief_confirmation"
        },
        {
            "user": "Yes, that looks perfect! Please proceed with the research.",
            "expected": "research"
        }
    ]
    
    # Run through the conversation
    for step in conversation_flow:
        if step["user"] is None:
            # Initial invocation
            print("\n--- Starting conversation ---")
            state = app.invoke(
                {"messages": [], "iteration_count": 0},
                config
            )
        else:
            # User input
            print(f"\nUser: {step['user']}")
            state = app.invoke(
                {"messages": [HumanMessage(content=step['user'])]},
                config
            )
        
        # Display assistant response
        if state.get("messages"):
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"\nAssistant: {last_message.content[:500]}...")
                if len(last_message.content) > 500:
                    print("(Response truncated for display)")
        
        # Check state
        print(f"\nCurrent step: {state.get('current_step', 'unknown')}")
        print(f"Clarification complete: {state.get('clarification_complete', False)}")
        
        if state.get("current_step") == "research":
            print("\n--- Research phase started ---")
            print("The agent is now conducting research using Tavily search...")
            print("(In a real scenario, this would continue until research is complete)")
            break
    
    print("\n" + "=" * 60)
    print("Example session complete!")
    print("In a real deployment, the research agent would continue")
    print("searching and generating a comprehensive report.")
    print("=" * 60)


def example_direct_research():
    """Example of directly triggering research with a pre-defined brief"""
    
    print("\n" + "=" * 60)
    print("Direct Research Example")
    print("=" * 60)
    
    # Create the graph
    app = create_research_graph()
    
    # Configuration
    config = {"configurable": {"thread_id": "direct_research_1"}}
    
    # Directly set up a research brief
    research_brief = """
    Research Brief:
    
    Topic: Latest developments in quantum computing
    
    Scope:
    1. Recent quantum supremacy achievements (2023-2024)
    2. Breakthroughs from major tech companies (Google, IBM, Microsoft, etc.)
    3. Practical applications emerging in the next 5 years
    4. Hardware advances in quantum processors
    5. Software and algorithm developments
    
    Requirements:
    - Focus on developments from 2023-2024
    - Include both technical details and business implications
    - Provide citations for all major claims
    - Structure the report with clear sections
    """
    
    # Invoke with pre-set state
    state = app.invoke(
        {
            "messages": [],
            "clarification_complete": True,
            "research_brief": research_brief,
            "current_step": "research",
            "iteration_count": 0
        },
        config
    )
    
    print(f"\nResearch initiated with pre-defined brief")
    print(f"Current step: {state.get('current_step', 'unknown')}")
    
    if state.get("messages"):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            print(f"\nAgent response: {last_message.content[:300]}...")


if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please copy .env.example to .env and fill in your API keys.")
    else:
        # Run examples
        example_research_session()
        print("\n" + "=" * 60 + "\n")
        example_direct_research()