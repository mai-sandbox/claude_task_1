"""
LangGraph Deep Research Agent

This agent has two phases:
1. Clarification Agent: Interactive scope gathering via terminal
2. ReAct Research Agent: Performs deep research using Tavily search
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


class ResearchState(BaseModel):
    """State for the research workflow"""
    messages: List[BaseMessage] = Field(default_factory=list)
    research_topic: Optional[str] = None
    research_scope: Optional[str] = None
    research_depth: Optional[str] = None
    target_audience: Optional[str] = None
    is_scope_clarified: bool = False
    research_brief: Optional[str] = None
    final_report: Optional[str] = None


class ClarificationResponse(BaseModel):
    """Structured response from clarification agent"""
    needs_more_info: bool
    question: Optional[str] = None
    research_topic: Optional[str] = None
    research_scope: Optional[str] = None
    research_depth: Optional[str] = None
    target_audience: Optional[str] = None
    is_complete: bool = False


# Initialize models and tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
search_tool = TavilySearchResults(max_results=5)

clarification_llm = model.with_structured_output(ClarificationResponse)


def clarification_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent that clarifies research scope through interactive dialogue"""
    current_state = ResearchState(**state)
    
    if current_state.is_scope_clarified:
        return {"is_scope_clarified": True}
    
    system_prompt = """You are a research scope clarification agent. Your job is to gather detailed information about the user's research needs through interactive dialogue.

You need to collect:
1. Research Topic: What exactly should be researched?
2. Research Scope: How broad or narrow should the research be?
3. Research Depth: Surface-level overview or deep technical analysis?
4. Target Audience: Who is this research for?

Ask focused questions to clarify these aspects. Once you have enough information to create a comprehensive research brief, set is_complete=True.

Current information gathered:
- Topic: {topic}
- Scope: {scope}  
- Depth: {depth}
- Audience: {audience}

If you have sufficient information for all four aspects, set is_complete=True and provide a summary. Otherwise, ask the most important missing question.
""".format(
        topic=current_state.research_topic or "Not specified",
        scope=current_state.research_scope or "Not specified", 
        depth=current_state.research_depth or "Not specified",
        audience=current_state.target_audience or "Not specified"
    )
    
    messages = [SystemMessage(content=system_prompt)] + current_state.messages
    response = clarification_llm.invoke(messages)
    
    updates = {}
    
    if response.research_topic:
        updates["research_topic"] = response.research_topic
    if response.research_scope:
        updates["research_scope"] = response.research_scope
    if response.research_depth:
        updates["research_depth"] = response.research_depth
    if response.target_audience:
        updates["target_audience"] = response.target_audience
    
    if response.is_complete:
        # Generate research brief
        brief = f"""Research Brief:
Topic: {updates.get('research_topic', current_state.research_topic)}
Scope: {updates.get('research_scope', current_state.research_scope)}
Depth: {updates.get('research_depth', current_state.research_depth)}
Target Audience: {updates.get('target_audience', current_state.target_audience)}

Please conduct comprehensive research on this topic following these specifications."""
        
        updates["research_brief"] = brief
        updates["is_scope_clarified"] = True
        
        ai_message = AIMessage(content=f"Perfect! I have all the information needed. Here's your research brief:\n\n{brief}\n\nNow I'll pass this to our research agent to conduct the deep research.")
    else:
        ai_message = AIMessage(content=response.question)
        # Use interrupt to pause for user input
        interrupt(response.question)
    
    updates["messages"] = current_state.messages + [ai_message]
    return updates


def create_research_brief(state: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a comprehensive research brief"""
    current_state = ResearchState(**state)
    
    brief = f"""Research Brief:
Topic: {current_state.research_topic}
Scope: {current_state.research_scope}
Depth: {current_state.research_depth}
Target Audience: {current_state.target_audience}

Conduct comprehensive research following these specifications. Use multiple searches to gather diverse perspectives and current information."""
    
    return {"research_brief": brief}


# Create ReAct research agent
research_agent = create_react_agent(
    model=model,
    tools=[search_tool],
    prompt="""You are a expert research agent. Your job is to conduct thorough research based on the provided brief.

Use the search tool multiple times with different queries to:
1. Gather comprehensive information about the topic
2. Find different perspectives and viewpoints  
3. Look for recent developments and current trends
4. Collect relevant statistics, case studies, and examples

After gathering information, synthesize it into a detailed, well-structured research report that matches the specified scope, depth, and target audience.

Your final response should be a comprehensive research report, not just search results."""
)


def research_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that performs the research using ReAct agent"""
    current_state = ResearchState(**state)
    
    # Prepare messages for research agent
    research_messages = [
        HumanMessage(content=f"Please conduct research based on this brief:\n\n{current_state.research_brief}")
    ]
    
    # Run research agent
    result = research_agent.invoke({"messages": research_messages})
    
    # Extract final report from the last AI message
    if result.get("messages"):
        final_message = result["messages"][-1]
        final_report = final_message.content
        
        return {
            "final_report": final_report,
            "messages": current_state.messages + [AIMessage(content=f"Research completed! Here's your detailed report:\n\n{final_report}")]
        }
    
    return {"messages": current_state.messages + [AIMessage(content="Research completed but no report was generated.")]}


def should_continue_clarification(state: Dict[str, Any]) -> Literal["research", "clarify"]:
    """Routing function to determine next step"""
    current_state = ResearchState(**state)
    
    if current_state.is_scope_clarified:
        return "research"
    else:
        return "clarify"


# Build the main workflow graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("clarify", clarification_agent)
workflow.add_node("research", research_node)

# Add edges
workflow.add_edge(START, "clarify")
workflow.add_conditional_edges("clarify", should_continue_clarification)
workflow.add_edge("research", END)

# Compile the graph
graph = workflow.compile()
app = graph