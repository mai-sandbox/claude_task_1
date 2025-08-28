from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: str
    clarification_complete: bool
    final_report: str