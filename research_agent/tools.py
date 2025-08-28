from typing import Dict, Any
from tavily import TavilyClient
from langchain_core.tools import tool
import os


class TavilySearchTool:
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    @tool
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the web using Tavily for comprehensive information on a given query.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (default: 5)
            
        Returns:
            Dictionary containing search results with titles, URLs, and content snippets
        """
        try:
            results = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_domains=None,
                exclude_domains=None
            )
            
            formatted_results = {
                "query": query,
                "answer": results.get("answer", ""),
                "results": []
            }
            
            for result in results.get("results", []):
                formatted_results["results"].append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0)
                })
            
            return formatted_results
            
        except Exception as e:
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": []
            }


tavily_tool = TavilySearchTool().search_web