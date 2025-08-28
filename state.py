from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    topic: str = Field(description="Main research topic")
    specific_questions: List[str] = Field(description="Specific questions to answer")
    scope: str = Field(description="Scope and boundaries of research")
    output_requirements: str = Field(description="Requirements for the final output")


class AgentState(TypedDict):
    messages: List[dict]
    research_brief: Optional[ResearchBrief]
    current_phase: Literal["clarification", "research", "complete"]
    research_report: Optional[str]
    search_results: List[dict]
    clarification_complete: bool