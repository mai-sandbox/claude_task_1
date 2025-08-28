from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages


class ResearchState(TypedDict):
    """State for the research agent workflow"""
    messages: Annotated[List, add_messages]
    clarification_complete: bool
    research_brief: Optional[str]
    research_report: Optional[str]
    current_step: str
    search_results: List[dict]
    iteration_count: int