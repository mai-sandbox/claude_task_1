from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import List, Dict, Any
import os


def get_search_tool():
    """Create and return the Tavily search tool"""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    
    return TavilySearchResults(
        max_results=5,
        api_key=tavily_api_key,
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        search_depth="advanced"
    )


@tool
def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable string
    
    Args:
        results: List of search result dictionaries from Tavily
        
    Returns:
        Formatted string of search results
    """
    if not results:
        return "No search results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"Result {i}:")
        formatted.append(f"  Title: {result.get('title', 'No title')}")
        formatted.append(f"  URL: {result.get('url', 'No URL')}")
        formatted.append(f"  Content: {result.get('content', 'No content')[:500]}...")
        formatted.append("")
    
    return "\n".join(formatted)