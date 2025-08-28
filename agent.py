"""
Deep Research Agent with Interactive Scope Clarification and ReAct Search

This agent has two main components:
1. Interactive Scope Clarification Agent - Collects research requirements from user
2. ReAct Agent with Tavily Search - Conducts research and generates detailed reports
"""

from typing import Dict, Any, List, Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.graph.message import add_messages


# State definitions
class ResearchBrief(BaseModel):
    """Structured research brief generated from user interaction"""
    topic: str = Field(description="Main research topic")
    scope: str = Field(description="Research scope and boundaries") 
    key_questions: List[str] = Field(description="Key questions to investigate")
    depth_level: str = Field(description="Required depth: surface, moderate, or deep")
    target_audience: str = Field(description="Target audience for the report")
    deliverable_format: str = Field(description="Expected format of deliverable")


class ScopeState(TypedDict):
    """State for scope clarification phase"""
    messages: Annotated[List[BaseMessage], add_messages]
    clarification_complete: bool
    research_brief: Dict[str, Any]


class ResearchState(TypedDict):
    """State for research phase"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: Dict[str, Any]
    final_report: str


# Initialize model and tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Tavily search tool
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True
)


@tool
def create_research_brief(
    topic: str,
    scope: str, 
    key_questions: List[str],
    depth_level: str,
    target_audience: str,
    deliverable_format: str
) -> str:
    """Create a structured research brief from clarified requirements"""
    brief = ResearchBrief(
        topic=topic,
        scope=scope,
        key_questions=key_questions,
        depth_level=depth_level,
        target_audience=target_audience,
        deliverable_format=deliverable_format
    )
    return brief.model_dump_json()


# Scope Clarification Agent nodes
def scope_clarifier_node(state: ScopeState) -> Dict[str, Any]:
    """Interactive node that clarifies research scope with the user"""
    messages = state["messages"]
    
    if not messages or len(messages) == 1:
        # First interaction - ask initial clarifying questions
        system_prompt = """You are a research scope clarification specialist. Your job is to understand what the user wants to research through interactive dialogue.

Ask targeted questions to understand:
- The specific topic or subject area
- The scope and boundaries of research  
- Key questions they want answered
- Required depth level (surface overview, moderate analysis, or deep investigation)
- Target audience for the final report
- Preferred format for deliverables

Be conversational and ask 2-3 focused questions at a time. Once you have sufficient clarity on all aspects, use the create_research_brief tool to formalize the requirements.

Important: Use the interrupt() function to pause and get user input for each clarification round."""
        
        prompt_msg = SystemMessage(content=system_prompt)
        initial_msg = AIMessage(content="Hi! I'm here to help clarify your research requirements. To get started, could you tell me:\n\n1. What specific topic or subject area would you like me to research?\n2. What's your main goal with this research - are you looking for a general overview or investigating specific aspects?\n3. Who is the intended audience for this research?")
        
        from langgraph.constants import interrupt
        interrupt("Please provide your research topic and initial requirements.")
        
        return {
            "messages": [initial_msg],
            "clarification_complete": False
        }
    
    # Continue clarification dialogue
    response = model.invoke([
        SystemMessage(content="""Continue the clarification dialogue. Ask follow-up questions to get complete requirements for:
- Topic specificity and boundaries
- Key research questions 
- Depth level needed
- Target audience
- Deliverable format preferences

When you have enough information, use the create_research_brief tool to finalize the requirements.""")
    ] + messages)
    
    # Check if we have a tool call to create research brief
    if hasattr(response, 'tool_calls') and response.tool_calls:
        return {
            "messages": [response],
            "clarification_complete": True
        }
    else:
        from langgraph.constants import interrupt
        interrupt("Please provide additional details based on my questions.")
        return {
            "messages": [response],
            "clarification_complete": False
        }


def extract_brief_node(state: ScopeState) -> Dict[str, Any]:
    """Extract the research brief from tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Execute the tool call to get the research brief
        tool_call = last_message.tool_calls[0]
        if tool_call['name'] == 'create_research_brief':
            brief_json = create_research_brief.invoke(tool_call['args'])
            brief_dict = json.loads(brief_json)
            return {
                "research_brief": brief_dict,
                "messages": [AIMessage(content=f"Research brief created! Moving to research phase...\n\nBrief: {brief_json}")]
            }
    
    return {"research_brief": {}}


# Research Agent with ReAct pattern
def create_research_agent():
    """Create the ReAct agent for conducting research"""
    system_prompt = """You are an expert researcher. Use the Tavily search tool to conduct thorough research based on the provided research brief.

Your research process should:
1. Break down the research brief into searchable components
2. Conduct multiple targeted searches to gather comprehensive information
3. Analyze and synthesize findings
4. Generate a detailed, well-structured report

For each search, think about what specific information you're looking for and how it relates to the research questions. Be thorough and use multiple search queries to cover all aspects."""
    
    research_agent = create_react_agent(
        model=model,
        tools=[search_tool],
        prompt=system_prompt
    )
    return research_agent


def research_conductor_node(state: ResearchState) -> Dict[str, Any]:
    """Node that conducts research using the ReAct agent"""
    research_brief = state["research_brief"]
    
    # Create research instruction based on brief
    research_instruction = f"""
Based on this research brief, conduct comprehensive research:

Topic: {research_brief.get('topic', 'Not specified')}
Scope: {research_brief.get('scope', 'Not specified')}
Key Questions: {', '.join(research_brief.get('key_questions', []))}
Depth Level: {research_brief.get('depth_level', 'moderate')}
Target Audience: {research_brief.get('target_audience', 'general')}
Deliverable Format: {research_brief.get('deliverable_format', 'report')}

Please conduct thorough research and provide a detailed report that addresses all key questions with appropriate depth for the target audience.
"""
    
    # Create and run the research agent
    research_agent = create_research_agent()
    
    result = research_agent.invoke({
        "messages": [HumanMessage(content=research_instruction)]
    })
    
    # Extract the final report from the agent's response
    if result.get("messages"):
        final_message = result["messages"][-1]
        final_report = final_message.content if hasattr(final_message, 'content') else str(final_message)
    else:
        final_report = "Research completed but no detailed report was generated."
    
    return {
        "messages": [AIMessage(content=f"Research completed! Here's your detailed report:\n\n{final_report}")],
        "final_report": final_report
    }


# Conditional logic
def should_continue_clarification(state: ScopeState) -> Literal["extract_brief", "clarify"]:
    """Determine if clarification is complete"""
    return "extract_brief" if state.get("clarification_complete", False) else "clarify"


def should_start_research(state: ScopeState) -> Literal["research", "clarify"]:
    """Check if we have a valid research brief to start research"""
    return "research" if state.get("research_brief") else "clarify"


# Build the main workflow
def build_research_workflow():
    """Build the complete research workflow"""
    
    # Scope clarification subgraph
    scope_graph = StateGraph(ScopeState)
    scope_graph.add_node("clarify", scope_clarifier_node)
    scope_graph.add_node("extract_brief", extract_brief_node)
    
    scope_graph.add_edge(START, "clarify")
    scope_graph.add_conditional_edges(
        "clarify",
        should_continue_clarification,
        {"extract_brief": "extract_brief", "clarify": "clarify"}
    )
    scope_graph.add_conditional_edges(
        "extract_brief", 
        should_start_research,
        {"research": END, "clarify": "clarify"}
    )
    
    scope_workflow = scope_graph.compile()
    
    # Main research workflow
    main_graph = StateGraph(ResearchState)
    main_graph.add_node("scope_clarification", lambda state: scope_workflow.invoke({
        "messages": state["messages"],
        "clarification_complete": False,
        "research_brief": {}
    }))
    main_graph.add_node("research", research_conductor_node)
    
    main_graph.add_edge(START, "scope_clarification")
    main_graph.add_edge("scope_clarification", "research")
    main_graph.add_edge("research", END)
    
    return main_graph.compile()


# Create and export the main application
graph = build_research_workflow()
app = graph