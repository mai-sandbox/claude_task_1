"""
Test script for the research agent
"""

import asyncio
from research_agent import create_research_graph
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()


async def test_research_agent():
    """Test the research agent with a sample query"""
    
    # Verify environment variables
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file based on .env.example and add your API keys.")
        return
    
    print("Starting Research Agent Test...")
    print("="*50)
    
    # Create the graph
    app = create_research_graph()
    
    # Test scenario 1: Simple research query
    print("\nTest 1: Simple Research Query")
    print("-"*30)
    
    state = {
        "messages": [
            HumanMessage(content="I want to research the latest developments in quantum computing."),
        ],
        "research_brief": None,
        "research_report": None,
        "phase": "clarification",
        "clarification_complete": False
    }
    
    # First clarification round
    result = await app.ainvoke(state)
    print(f"Clarification Response: {result['messages'][-1].content[:200]}...")
    
    # Confirm to proceed
    result["messages"].append(HumanMessage(content="Focus on breakthroughs in 2024, particularly in error correction and quantum advantage. Yes, proceed with the research."))
    
    # Run research
    result = await app.ainvoke(result)
    
    if result.get("research_brief"):
        print(f"\nResearch Brief Created:")
        print(f"Topic: {result['research_brief'].topic}")
        print(f"Scope: {result['research_brief'].scope}")
    
    # Continue until research is complete
    max_iterations = 10
    iteration = 0
    
    while result.get("phase") != "complete" and iteration < max_iterations:
        result = await app.ainvoke(result)
        iteration += 1
        print(f"Research iteration {iteration}...")
    
    if result.get("research_report"):
        print("\nResearch Completed Successfully!")
        print(f"Report Preview: {result['research_report'][:500]}...")
    else:
        print("\nResearch did not complete within maximum iterations")
    
    print("\n" + "="*50)
    print("Test Complete")


def test_deployment_config():
    """Test that deployment configuration is valid"""
    print("\nTesting Deployment Configuration...")
    print("-"*30)
    
    # Check if langgraph.json exists
    if os.path.exists("langgraph.json"):
        print("✓ langgraph.json found")
    else:
        print("✗ langgraph.json not found")
    
    # Check if the app can be imported
    try:
        from research_agent_deployment import app
        print("✓ Deployment module can be imported")
        print(f"✓ App type: {type(app)}")
    except ImportError as e:
        print(f"✗ Failed to import deployment module: {e}")
    
    print("\nDeployment configuration test complete")


if __name__ == "__main__":
    print("Research Agent Test Suite")
    print("="*50)
    
    # Test deployment configuration
    test_deployment_config()
    
    # Run async test
    print("\nRunning agent test (requires API keys)...")
    try:
        asyncio.run(test_research_agent())
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure you have set up your .env file with valid API keys")