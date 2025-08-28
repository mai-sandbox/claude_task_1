#!/usr/bin/env python3
"""
Local runner for the research agent.
Run this script to test the agent locally before deploying to LangGraph platform.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from research_agent.graph import graph


def main():
    load_dotenv()
    
    required_keys = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("❌ Missing required environment variables:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease set these in your .env file")
        return
    
    print("🚀 Starting Deep Research Agent")
    print("=" * 50)
    
    config = {"configurable": {"thread_id": "research-session-1"}}
    
    initial_query = input("What would you like me to research? ")
    
    initial_state = {
        "messages": [HumanMessage(content=initial_query)],
        "research_brief": "",
        "clarification_complete": False,
        "final_report": ""
    }
    
    try:
        result = graph.invoke(initial_state, config)
        
        print("\n" + "="*50)
        print("🎉 Research Process Complete!")
        print("="*50)
        
        if result.get("final_report"):
            print(f"\n📊 Final Report:")
            print("-" * 30)
            print(result["final_report"])
        else:
            print("No final report generated.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Research session ended by user")
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()