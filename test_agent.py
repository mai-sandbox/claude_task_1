"""
Test script for the research agent
"""

import os
from dotenv import load_dotenv
from research_agent import create_research_graph
from langchain_core.messages import HumanMessage, AIMessage
import json

load_dotenv()


def test_clarification_phase():
    """Test the clarification phase of the agent"""
    print("Testing Clarification Phase...")
    print("=" * 50)
    
    app = create_research_graph()
    
    # Initial state with a research request
    state = {
        "messages": [HumanMessage(content="I want to research the impact of AI on healthcare")],
        "research_brief": None,
        "clarification_complete": False,
        "final_report": None,
        "current_phase": "clarification"
    }
    
    # Run clarification
    result = app.invoke(state)
    
    print("Agent Response:")
    if result["messages"]:
        print(result["messages"][-1].content)
    
    print("\nClarification Complete:", result.get("clarification_complete", False))
    
    return result


def test_full_flow():
    """Test the complete flow with mock interactions"""
    print("\nTesting Full Flow...")
    print("=" * 50)
    
    app = create_research_graph()
    
    # Simulate a conversation
    state = {
        "messages": [
            HumanMessage(content="I want to research the impact of AI on healthcare"),
            AIMessage(content="I'll help you research the impact of AI on healthcare. To provide you with the most relevant information, let me ask a few clarifying questions:\n\n1. Are you interested in specific areas of healthcare (e.g., diagnostics, treatment, administration, patient care)?\n2. What timeframe are you considering - current applications, recent developments, or future predictions?"),
            HumanMessage(content="I'm interested in diagnostics and treatment, focusing on current applications and recent developments from the last 2-3 years"),
            AIMessage(content="Great! A few more questions to refine the scope:\n\n1. Are you looking for global perspectives or focusing on specific regions/countries?\n2. Are there particular AI technologies you're most interested in (e.g., machine learning, computer vision, NLP)?\n3. Do you want to include information about challenges, ethical considerations, or regulatory aspects?"),
            HumanMessage(content="Global perspective, all AI technologies, and yes include challenges and ethical considerations. CLARIFICATION_COMPLETE")
        ],
        "research_brief": None,
        "clarification_complete": False,
        "final_report": None,
        "current_phase": "clarification"
    }
    
    # Process clarification
    result = app.invoke(state)
    
    if result.get("research_brief"):
        print("\nResearch Brief Created:")
        brief = result["research_brief"]
        print(f"Topic: {brief.topic}")
        print(f"Scope: {brief.scope}")
        print(f"Questions: {brief.specific_questions}")
        print(f"Key Areas: {brief.key_areas}")
    
    # Continue to research if clarification is complete
    if result.get("clarification_complete"):
        print("\nProceeding to Research Phase...")
        result = app.invoke(result)
        
        if result.get("final_report"):
            print("\nFinal Report Generated:")
            print("-" * 50)
            print(result["final_report"][:500] + "..." if len(result["final_report"]) > 500 else result["final_report"])
    
    return result


def test_deployment_handler():
    """Test the deployment handler function"""
    print("\nTesting Deployment Handler...")
    print("=" * 50)
    
    from deployment import handle_request
    
    request = {
        "messages": ["I want to research quantum computing applications in cryptography"],
        "clarification_complete": False,
        "current_phase": "clarification"
    }
    
    response = handle_request(request)
    
    print("Request:", request)
    print("\nResponse:")
    print(json.dumps(response, indent=2, default=str)[:500])
    
    return response


if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the required API keys.")
        print("See .env.example for the template.")
        exit(1)
    
    print("🚀 Starting Research Agent Tests\n")
    
    try:
        # Run tests
        test_clarification_phase()
        # Note: Full flow test will actually call Tavily API
        # Uncomment the next line if you want to test with real API calls
        # test_full_flow()
        test_deployment_handler()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()