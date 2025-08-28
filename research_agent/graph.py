from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from research_agent.state import ResearchState
from research_agent.clarification_agent import ClarificationAgent
from research_agent.react_agent import ReactAgent


def should_continue_to_research(state: ResearchState) -> str:
    """Determine if we should continue to research phase"""
    if state.get("clarification_complete", False):
        return "research"
    return "clarify"


def create_research_graph():
    """Create the main research workflow graph"""
    
    clarification_agent = ClarificationAgent()
    react_agent = ReactAgent()
    
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("clarify", clarification_agent.clarify_research_scope)
    workflow.add_node("research", react_agent.conduct_research)
    
    workflow.set_entry_point("clarify")
    
    workflow.add_conditional_edges(
        "clarify",
        should_continue_to_research,
        {
            "clarify": "clarify",
            "research": "research"
        }
    )
    
    workflow.add_edge("research", END)
    
    memory = MemorySaver()
    
    app = workflow.compile(checkpointer=memory)
    
    return app


graph = create_research_graph()