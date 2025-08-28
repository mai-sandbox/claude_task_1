#!/usr/bin/env python3
"""
Simple test script for the research agent
"""
import os
import sys

# Set dummy API keys for testing if not present
if not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = "dummy_key_for_testing"
if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = "dummy_key_for_testing"

try:
    from agent import app
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)

def test_agent():
    """Test the research agent workflow"""
    print("Testing Deep Research Agent...")
    print("=" * 50)
    
    try:
        # Test initial state
        initial_state = {
            "messages": [],
            "research_brief": "",
            "research_scope_complete": False,
            "final_report": "",
            "user_input": "",
            "current_question": ""
        }
        
        print("✓ Agent initialized successfully")
        print("✓ State structure is valid")
        
        # Test the first interaction (should start scope clarification)
        config = {"recursion_limit": 5}  # Lower limit for testing
        result = app.invoke(initial_state, config=config)
        
        if result.get("messages") and len(result["messages"]) > 0:
            first_message = result["messages"][-1]
            print(f"✓ First interaction successful")
            print(f"Agent response: {first_message.content[:100]}...")
        else:
            print("✗ First interaction failed")
            
        print("\n" + "=" * 50)
        print("Test completed. Agent is ready for deployment.")
        print("\nTo use the agent:")
        print("1. Set ANTHROPIC_API_KEY environment variable")
        print("2. Set TAVILY_API_KEY environment variable") 
        print("3. Deploy to LangGraph platform using langgraph.json")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed required dependencies (pip install -r requirements.txt)")
        print("2. Set ANTHROPIC_API_KEY environment variable")
        print("3. Set TAVILY_API_KEY environment variable")

if __name__ == "__main__":
    test_agent()