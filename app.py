"""Main application file for LangGraph deployment"""

from graph import create_research_graph
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create and export the graph for LangGraph platform
graph = create_research_graph()

# For local testing
if __name__ == "__main__":
    from graph import run_research_agent
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please copy .env.example to .env and fill in your API keys.")
    else:
        run_research_agent()