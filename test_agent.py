#!/usr/bin/env python3
"""
Test script for the research agent.
This script simulates a user interaction to test the complete flow.
"""

from graph import create_research_graph
from dotenv import load_dotenv
import os


def test_research_agent():
    """Test the research agent with a sample research topic"""
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in environment variables")
        print("Please create a .env file with your Tavily API key")
        return
    
    app = create_research_graph()
    config = {"configurable": {"thread_id": "test_session"}}
    
    print("Testing Research Agent")
    print("=" * 50)
    
    test_messages = [
        "I want to research the latest developments in quantum computing",
        "Focus on recent breakthroughs in 2024, practical applications, and major companies involved",
        "Yes, please also include information about quantum supremacy claims and error correction advances",
        "The report should be detailed with sections for technical advances, commercial applications, and future outlook"
    ]
    
    state = {"messages": [], "current_phase": "clarification"}
    
    for i, message in enumerate(test_messages):
        print(f"\nTest Input {i+1}: {message}")
        
        state["messages"].append({"role": "user", "content": message})
        result = app.invoke(state, config)
        
        last_message = result["messages"][-1] if result.get("messages") else None
        
        if last_message and last_message.get("role") == "assistant":
            print(f"Assistant Response: {last_message['content'][:200]}...")
        
        state = result
        
        if result.get("current_phase") == "research":
            print("\n" + "=" * 50)
            print("Clarification complete. Moving to research phase...")
            print("=" * 50)
            break
    
    if result.get("current_phase") == "research":
        print("\nExecuting research phase...")
        result = app.invoke(state, config)
        
        if result.get("research_report"):
            print("\n" + "=" * 50)
            print("Research Report Generated Successfully!")
            print("=" * 50)
            print(f"\nReport Preview:\n{result['research_report'][:500]}...")
        else:
            print("\nNo research report generated")
    
    print("\n" + "=" * 50)
    print("Test Complete")
    print("=" * 50)


if __name__ == "__main__":
    test_research_agent()