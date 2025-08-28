"""
LangGraph Deep Research Agent

This agent has two main components:
1. A clarification agent that interacts with the user to understand research scope
2. A ReAct agent with Tavily search that conducts research and generates a detailed report
"""

import os
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# State management for the research workflow
class ResearchState(TypedDict):
    """State for the deep research agent workflow."""
    # User's original query
    original_query: str
    
    # Research brief generated from clarification
    research_brief: str
    
    # Clarification questions and answers
    clarification_complete: bool
    
    # Final research report
    final_report: str
    
    # Current stage of the workflow
    current_stage: Literal["clarification", "research", "completed"]


class ClarificationQuestion(BaseModel):
    """A clarification question to ask the user."""
    question: str = Field(description="The clarification question to ask")
    reasoning: str = Field(description="Why this question is important for the research")


class ResearchBrief(BaseModel):
    """A structured research brief based on user clarifications."""
    research_objectives: List[str] = Field(description="Clear research objectives")
    scope_boundaries: List[str] = Field(description="What to include/exclude")
    key_questions: List[str] = Field(description="Key questions to answer")
    target_depth: str = Field(description="Desired depth of research (surface/detailed/comprehensive)")
    specific_focus_areas: List[str] = Field(description="Specific areas to focus on")


# Initialize the LLM (Anthropic preferred as per guidelines)
model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.1
)

# Initialize Tavily search tool
def get_tavily_search():
    """Get Tavily search tool with API key handling."""
    try:
        return TavilySearchResults(
            max_results=5,
            search_depth="advanced"
        )
    except Exception:
        # For development/testing without API key, create a mock tool
        from langchain_core.tools import tool
        
        @tool
        def mock_tavily_search(query: str) -> str:
            """Mock search tool for testing without API key."""
            return f"Mock search results for: {query}. Please set TAVILY_API_KEY environment variable for real search functionality."
        
        return mock_tavily_search

tavily_search = get_tavily_search()


def clarification_agent(state: ResearchState) -> ResearchState:
    """
    Agent that clarifies the research scope with the user through terminal interaction.
    Uses interrupt() to pause for user input until research brief is complete.
    """
    
    # If we already have a complete research brief, skip clarification
    if state.get("clarification_complete", False):
        return state
    
    original_query = state["original_query"]
    
    # Generate clarification questions using structured output
    clarification_prompt = f"""
    You are a research assistant helping to clarify the scope of a research request.
    
    Original query: "{original_query}"
    
    Generate 2-3 focused clarification questions that will help you understand:
    1. The specific scope and boundaries of the research
    2. The depth and detail level required
    3. Any particular focus areas or constraints
    
    Make the questions specific and actionable.
    """
    
    # Get clarification questions from LLM
    questions_response = model.with_structured_output(List[ClarificationQuestion]).invoke([
        SystemMessage(content=clarification_prompt)
    ])
    
    # Collect user answers through interrupts
    clarification_answers = []
    
    for q in questions_response:
        user_answer = interrupt({
            "question": q.question,
            "reasoning": q.reasoning,
            "stage": "clarification"
        })
        
        clarification_answers.append({
            "question": q.question,
            "answer": user_answer
        })
    
    # Generate final research brief based on answers
    brief_prompt = f"""
    Based on the original query and user clarifications, create a comprehensive research brief.
    
    Original query: "{original_query}"
    
    User clarifications:
    {chr(10).join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in clarification_answers])}
    
    Create a structured research brief that will guide the research agent.
    """
    
    research_brief = model.with_structured_output(ResearchBrief).invoke([
        SystemMessage(content=brief_prompt)
    ])
    
    # Ask user for final confirmation
    confirmation = interrupt({
        "message": "Research brief created. Please review and confirm:",
        "research_brief": research_brief.model_dump(),
        "question": "Does this research brief accurately capture what you want? (yes/no or provide feedback)"
    })
    
    if confirmation.lower().startswith('y'):
        # Convert brief to string for the research agent
        brief_text = f"""
Research Objectives:
{chr(10).join([f"- {obj}" for obj in research_brief.research_objectives])}

Scope Boundaries:
{chr(10).join([f"- {scope}" for scope in research_brief.scope_boundaries])}

Key Questions to Answer:
{chr(10).join([f"- {q}" for q in research_brief.key_questions])}

Target Depth: {research_brief.target_depth}

Specific Focus Areas:
{chr(10).join([f"- {area}" for area in research_brief.specific_focus_areas])}
        """
        
        return {
            "research_brief": brief_text,
            "clarification_complete": True,
            "current_stage": "research"
        }
    else:
        # If user wants changes, restart clarification with feedback
        # In a full implementation, you'd incorporate the feedback
        return {
            "clarification_complete": False,
            "current_stage": "clarification"
        }


def research_agent_node(state: ResearchState) -> ResearchState:
    """
    Node that runs the ReAct agent with Tavily search to conduct research.
    """
    
    research_brief = state["research_brief"]
    original_query = state["original_query"]
    
    # Create ReAct agent with Tavily search tool
    react_agent = create_react_agent(
        model=model,
        tools=[tavily_search],
        prompt=f"""
You are a skilled research agent tasked with conducting comprehensive research.

Original Query: {original_query}

Research Brief:
{research_brief}

Your task:
1. Use the search tool to gather comprehensive information
2. Search multiple times with different angles and keywords
3. Synthesize findings into a detailed, well-structured report
4. Ensure you address all objectives and key questions from the brief
5. Provide sources and citations for your findings

Be thorough and analytical in your research approach.
        """
    )
    
    # Run the ReAct agent
    research_input = {
        "messages": [
            HumanMessage(content=f"Please conduct research based on this brief: {research_brief}")
        ]
    }
    
    result = react_agent.invoke(research_input)
    
    # Extract the final message content as the report
    if result.get("messages"):
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            report = final_message.content
        else:
            report = str(final_message)
    else:
        report = "Research completed but no detailed report was generated."
    
    return {
        "final_report": report,
        "current_stage": "completed"
    }


def should_continue(state: ResearchState) -> Literal["research_agent", "end"]:
    """Conditional edge to determine next step."""
    if state.get("clarification_complete", False):
        return "research_agent"
    return "end"


# Build the workflow graph
def create_research_workflow():
    """Create the deep research agent workflow."""
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarification_agent", clarification_agent)
    workflow.add_node("research_agent", research_agent_node)
    
    # Add edges
    workflow.add_edge(START, "clarification_agent")
    workflow.add_conditional_edges(
        "clarification_agent",
        should_continue,
        {
            "research_agent": "research_agent",
            "end": END
        }
    )
    workflow.add_edge("research_agent", END)
    
    return workflow


# Compile the graph for deployment
workflow = create_research_workflow()

# Note: Not adding checkpointer for deployment as per guidelines
# For local testing with human-in-the-loop, you would add:
# checkpointer = InMemorySaver()
# app = workflow.compile(checkpointer=checkpointer)

# For deployment without checkpointer
app = workflow.compile()


# Test function for local development
def test_agent():
    """Test function for local development with checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver
    
    checkpointer = InMemorySaver()
    test_graph = workflow.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread"}}
    
    # Test input
    initial_state = {
        "original_query": "I want to research the impact of artificial intelligence on job markets",
        "clarification_complete": False,
        "current_stage": "clarification"
    }
    
    print("Starting research agent test...")
    print("=" * 50)
    
    try:
        result = test_graph.invoke(initial_state, config=config)
        print("Result:", result)
        
        # Check for interrupts
        if "__interrupt__" in result:
            print("\nInterrupt detected:")
            for interrupt_obj in result["__interrupt__"]:
                print(f"Value: {interrupt_obj.value}")
                
            # For testing, simulate user response
            print("\n=== Simulating user response ===")
            test_response = test_graph.invoke(
                Command(resume="I want to focus on white-collar jobs and understand both positive and negative impacts over the next 5-10 years."),
                config=config
            )
            print("After first response:", test_response)
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agent()