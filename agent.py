"""
LangGraph Deep Research Agent

This agent implements a two-phase research workflow:
1. Clarification Phase: Interactive back-and-forth with user to clarify research scope
2. Research Phase: ReAct agent with Tavily search to generate detailed report
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


# State definitions
class ResearchState(TypedDict):
    """Main state for the research workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: Optional[str]
    phase: str  # "clarification" or "research"
    final_report: Optional[str]


class ResearchBrief(BaseModel):
    """Structured research brief generated from clarification phase"""
    research_topic: str = Field(description="The main research topic")
    research_questions: List[str] = Field(description="Specific questions to answer")
    scope: str = Field(description="Scope and boundaries of the research")
    depth: str = Field(description="Required depth level (overview, detailed, comprehensive)")
    target_audience: str = Field(description="Intended audience for the report")
    key_areas: List[str] = Field(description="Key areas or subtopics to explore")


# Initialize models and tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize search tool with fallback for missing API key
try:
    search_tool = TavilySearchResults(max_results=5)
except Exception as e:
    # Fallback for testing without API key
    print(f"Warning: Could not initialize Tavily search tool: {e}")
    search_tool = None


def clarification_node(state: ResearchState) -> Dict[str, Any]:
    """
    Interactive clarification agent that asks follow-up questions
    to understand the research scope and requirements
    """
    messages = state["messages"]
    phase = state.get("phase", "clarification")
    
    if phase != "clarification":
        return {"messages": []}
    
    # System prompt for clarification agent
    clarification_prompt = """You are a research clarification specialist. Your job is to understand exactly what the user wants to research through interactive dialogue.

Ask targeted follow-up questions to clarify:
1. The specific research topic and its boundaries
2. What questions they want answered
3. The depth and scope of research needed
4. The intended audience for the final report
5. Any specific areas or angles they want explored

Be conversational and helpful. Ask 1-2 focused questions at a time. When you have enough information to create a comprehensive research brief, say "I have enough information to proceed with the research."

Current conversation context: This is an ongoing clarification dialogue to define the research scope."""
    
    # Add system message if this is the start of clarification
    if len(messages) == 1 or not any(isinstance(msg, SystemMessage) for msg in messages):
        clarification_messages = [SystemMessage(content=clarification_prompt)] + messages
    else:
        clarification_messages = messages
    
    # Check if we should move to research phase
    if messages and "I have enough information to proceed" in messages[-1].content:
        # Generate research brief
        brief_prompt = """Based on the conversation above, create a comprehensive research brief. Extract:
- The main research topic
- Specific research questions to answer  
- Scope and boundaries
- Required depth level
- Target audience
- Key areas to explore

Be thorough and specific."""
        
        brief_messages = clarification_messages + [HumanMessage(content=brief_prompt)]
        
        # Use structured output to generate research brief
        structured_model = model.with_structured_output(ResearchBrief)
        brief_response = structured_model.invoke(brief_messages)
        
        # Convert brief to string format
        brief_text = f"""Research Brief:

Topic: {brief_response.research_topic}

Research Questions:
{chr(10).join(f"- {q}" for q in brief_response.research_questions)}

Scope: {brief_response.scope}

Depth Level: {brief_response.depth}

Target Audience: {brief_response.target_audience}

Key Areas to Explore:
{chr(10).join(f"- {area}" for area in brief_response.key_areas)}"""
        
        return {
            "research_brief": brief_text,
            "phase": "research",
            "messages": [AIMessage(content="Perfect! I've created a comprehensive research brief and will now begin the detailed research phase.")]
        }
    
    # Continue clarification dialogue
    response = model.invoke(clarification_messages)
    
    # Interrupt for user input if we need more clarification
    if "I have enough information to proceed" not in response.content:
        interrupt("Please provide your response to continue the research clarification.")
    
    return {"messages": [response]}


def research_coordinator_node(state: ResearchState) -> Dict[str, Any]:
    """
    Coordinates the research phase by setting up the ReAct agent
    """
    if state.get("phase") != "research" or not state.get("research_brief"):
        return {"messages": []}
    
    research_brief = state["research_brief"]
    
    # Create research instruction for ReAct agent
    research_instruction = f"""You are a professional research analyst. Use the search tool to conduct comprehensive research based on this brief:

{research_brief}

Instructions:
1. Break down the research into logical search queries
2. Search for current, credible information on each aspect
3. Synthesize findings into a well-structured, detailed report
4. Include specific facts, statistics, examples, and sources
5. Address all research questions comprehensively

Begin your research now."""
    
    return {
        "messages": [HumanMessage(content=research_instruction)],
        "phase": "research_active"
    }


def should_continue_research(state: ResearchState) -> str:
    """Routing function to determine workflow path"""
    phase = state.get("phase", "clarification")
    
    if phase == "clarification":
        return "clarification"
    elif phase == "research":
        return "research_coordinator" 
    elif phase == "research_active":
        return "react_researcher"
    else:
        return END


# Create ReAct research agent
tools = [search_tool] if search_tool else []
react_researcher = create_react_agent(
    model=model,
    tools=tools,
    prompt="""You are a professional research analyst with access to web search. 

When you receive research instructions:
1. Plan your search strategy based on the research brief
2. Conduct systematic searches to gather comprehensive information
3. Analyze and synthesize the information
4. Create a detailed, well-structured report

Always:
- Use multiple search queries to get comprehensive coverage
- Fact-check important claims with additional searches
- Organize information logically with clear sections
- Include specific examples and data points
- Cite sources and provide context
- Address all aspects of the research brief

Your final response should be a complete research report ready for the intended audience."""
)


# Build the research workflow
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("clarification", clarification_node)
workflow.add_node("research_coordinator", research_coordinator_node)
workflow.add_node("react_researcher", react_researcher)

# Add edges
workflow.add_edge(START, "clarification")
workflow.add_conditional_edges(
    "clarification",
    should_continue_research,
    {
        "clarification": "clarification",
        "research_coordinator": "research_coordinator",
        END: END
    }
)
workflow.add_edge("research_coordinator", "react_researcher")
workflow.add_edge("react_researcher", END)

# Compile and export
graph = workflow.compile()
app = graph