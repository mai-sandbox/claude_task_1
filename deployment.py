"""
LangGraph Cloud deployment configuration
"""

from research_agent import create_research_graph
from langchain_core.messages import HumanMessage
from typing import Dict, Any

# Create the agent graph for deployment
agent = create_research_graph()


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle incoming requests to the deployed agent
    
    Args:
        request: Dictionary containing the request data
            - messages: List of messages
            - research_brief: Optional existing brief
            - clarification_complete: Boolean flag
            
    Returns:
        Dictionary containing the response
    """
    
    # Extract input from request
    messages = request.get("messages", [])
    
    # Convert string messages to proper message objects if needed
    processed_messages = []
    for msg in messages:
        if isinstance(msg, str):
            processed_messages.append(HumanMessage(content=msg))
        else:
            processed_messages.append(msg)
    
    # Prepare initial state
    state = {
        "messages": processed_messages,
        "research_brief": request.get("research_brief"),
        "clarification_complete": request.get("clarification_complete", False),
        "final_report": None,
        "current_phase": request.get("current_phase", "clarification")
    }
    
    # Run the agent
    result = agent.invoke(state)
    
    # Format response
    response = {
        "messages": [msg.content if hasattr(msg, 'content') else str(msg) 
                    for msg in result.get("messages", [])],
        "research_brief": result.get("research_brief"),
        "clarification_complete": result.get("clarification_complete", False),
        "final_report": result.get("final_report"),
        "current_phase": result.get("current_phase")
    }
    
    return response


# Export for LangGraph Cloud
__all__ = ["agent", "handle_request"]