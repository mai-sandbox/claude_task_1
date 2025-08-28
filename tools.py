from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing import List, Dict, Any
import os


def create_tavily_search_tool(max_results: int = 5) -> TavilySearchResults:
    """Create a Tavily search tool with specified max results"""
    return TavilySearchResults(
        max_results=max_results,
        api_key=os.getenv("TAVILY_API_KEY"),
        description="Search the web for information using Tavily API. Returns relevant search results with content."
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
        formatted.append(f"\n**Result {i}:**")
        formatted.append(f"Title: {result.get('title', 'N/A')}")
        formatted.append(f"URL: {result.get('url', 'N/A')}")
        formatted.append(f"Content: {result.get('content', 'N/A')[:500]}...")
        formatted.append("-" * 50)
    
    return "\n".join(formatted)