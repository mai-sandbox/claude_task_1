from typing import Dict, Any, List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.constants import Send
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

# State definition for the main workflow
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    research_brief: str
    research_scope_complete: bool
    final_report: str
    user_input: str
    current_question: str

# Pydantic model for structured research brief
class ResearchBrief(BaseModel):
    """Structured research brief generated from user interaction"""
    topic: str = Field(description="Main research topic")
    scope: str = Field(description="Research scope and boundaries")
    key_questions: List[str] = Field(description="Key questions to investigate")
    deliverables: str = Field(description="Expected deliverables and format")
    depth: str = Field(description="Required depth of research (surface, detailed, comprehensive)")

# Initialize the model
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize Tavily search tool (will be created when needed to handle API key)
def get_tavily_tool():
    """Create Tavily search tool with proper error handling"""
    try:
        return TavilySearchResults(
            max_results=5,
            search_depth="advanced" if hasattr(TavilySearchResults, 'search_depth') else None,
            include_answer=True if hasattr(TavilySearchResults, 'include_answer') else None,
            include_raw_content=True if hasattr(TavilySearchResults, 'include_raw_content') else None
        )
    except Exception as e:
        # Fallback with minimal parameters
        return TavilySearchResults(max_results=5)

def scope_clarification_node(state: ResearchState) -> Dict[str, Any]:
    """Interactive node to clarify research scope with the user"""
    messages = state.get("messages", [])
    
    # If this is the first interaction, start the conversation
    if not messages:
        initial_message = AIMessage(content="""Hello! I'm your deep research agent. I'll help you conduct comprehensive research on any topic.

To get started, please tell me:
1. What topic would you like me to research?
2. What specific aspects are you most interested in?
3. What's the purpose of this research? (academic, business, personal interest, etc.)

Please provide as much detail as possible about what you're looking for.""")
        
        return {
            "messages": [initial_message],
            "current_question": "initial_scope",
            "research_scope_complete": False
        }
    
    # Get the latest human message
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not human_messages:
        return {"messages": [AIMessage(content="I'm waiting for your input. Please tell me what you'd like me to research.")]}
    
    latest_input = human_messages[-1].content
    
    # Use structured output to determine if we have enough information
    class ScopeAssessment(BaseModel):
        has_clear_topic: bool = Field(description="Whether the user has provided a clear research topic")
        has_scope_details: bool = Field(description="Whether specific scope and boundaries are defined")
        has_purpose: bool = Field(description="Whether the research purpose is clear")
        needs_clarification: bool = Field(description="Whether more clarification is needed")
        follow_up_question: str = Field(description="Next question to ask if clarification needed")
        research_brief: ResearchBrief = Field(description="Complete research brief if ready")
    
    assessment_prompt = f"""
    Based on the conversation so far, assess whether we have enough information to proceed with research.
    
    Conversation history:
    {chr(10).join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages[-5:]])}
    
    Latest user input: {latest_input}
    
    Determine if we have:
    1. A clear research topic
    2. Specific scope and boundaries  
    3. Clear purpose/use case
    4. Enough detail to create a comprehensive research brief
    
    If not ready, provide a specific follow-up question to get the missing information.
    If ready, create the complete research brief.
    """
    
    structured_model = model.with_structured_output(ScopeAssessment)
    assessment = structured_model.invoke([
        SystemMessage(content="You are assessing research scope clarity and generating follow-up questions or research briefs."),
        HumanMessage(content=assessment_prompt)
    ])
    
    if assessment.needs_clarification:
        # Use interrupt to pause for user input
        user_response = interrupt(assessment.follow_up_question)
        response = AIMessage(content=assessment.follow_up_question)
        return {
            "messages": [response],
            "current_question": "clarification",
            "research_scope_complete": False
        }
    else:
        # We have enough information - create the research brief
        brief_text = f"""
RESEARCH BRIEF:

Topic: {assessment.research_brief.topic}

Scope: {assessment.research_brief.scope}

Key Research Questions:
{chr(10).join([f"- {q}" for q in assessment.research_brief.key_questions])}

Deliverables: {assessment.research_brief.deliverables}

Research Depth: {assessment.research_brief.depth}
"""
        
        confirmation_message = AIMessage(content=f"""Perfect! I now have a clear understanding of your research needs. Here's what I'll investigate:

{brief_text}

I'll now proceed to conduct comprehensive research using advanced search tools and generate a detailed report for you. This may take a few moments as I gather and analyze information from multiple sources.

Starting research process...""")
        
        return {
            "messages": [confirmation_message],
            "research_brief": brief_text,
            "research_scope_complete": True,
            "current_question": "complete"
        }

def research_agent_node(state: ResearchState) -> Dict[str, Any]:
    """ReAct agent that conducts the actual research"""
    research_brief = state.get("research_brief", "")
    
    if not research_brief:
        return {"messages": [AIMessage(content="Error: No research brief available. Please complete scope clarification first.")]}
    
    # Create a ReAct agent with Tavily search
    research_agent = create_react_agent(
        model=model,
        tools=[get_tavily_tool()],
        prompt=f"""You are an expert research agent. Your task is to conduct comprehensive research based on the following brief:

{research_brief}

Use the search tool to gather information from multiple sources. Be thorough and analytical. Your research should:

1. Address all key questions mentioned in the brief
2. Gather information from diverse, credible sources
3. Analyze and synthesize findings
4. Identify patterns, trends, and insights
5. Note any conflicting information or gaps

After gathering information, compile a detailed research report that is well-structured and comprehensive.

Start by searching for information related to the main topic and key questions."""
    )
    
    # Run the research agent
    result = research_agent.invoke({
        "messages": [HumanMessage(content=f"Please conduct research based on this brief: {research_brief}")]
    })
    
    # Extract the final message content
    if result.get("messages"):
        final_message = result["messages"][-1]
        report_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        # Format the final report
        formatted_report = f"""
# COMPREHENSIVE RESEARCH REPORT

## Research Brief
{research_brief}

## Detailed Findings
{report_content}

---
*Research completed using advanced search and analysis tools*
"""
        
        return {
            "messages": [AIMessage(content=formatted_report)],
            "final_report": formatted_report
        }
    else:
        return {"messages": [AIMessage(content="Error: Research agent did not return results.")]}

def should_continue_clarification(state: ResearchState) -> str:
    """Conditional edge to determine if scope clarification is complete"""
    if state.get("research_scope_complete", False):
        return "research"
    else:
        return "clarification"

# Build the workflow
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("clarification", scope_clarification_node)
workflow.add_node("research", research_agent_node)

# Add edges
workflow.add_edge(START, "clarification")
workflow.add_conditional_edges(
    "clarification",
    should_continue_clarification,
    {
        "clarification": "clarification",  # Continue clarifying
        "research": "research"             # Move to research
    }
)
workflow.add_edge("research", END)

# Compile the graph
graph = workflow.compile()
app = graph