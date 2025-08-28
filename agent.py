"""
LangGraph Deep Research Agent

A two-stage research agent that:
1. Interacts with users via terminal to clarify research scope  
2. Uses a ReAct agent with Tavily search to conduct research and generate reports
"""
import os
from typing import Dict, Any, List, Annotated
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command


class ResearchState(BaseModel):
    """State for the research agent workflow"""
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    research_topic: str = ""
    research_scope: str = ""
    research_depth: str = ""
    target_audience: str = ""
    specific_questions: List[str] = Field(default_factory=list)
    research_brief: str = ""
    final_report: str = ""
    user_approved: bool = False


def clarify_research_scope(state: ResearchState) -> Dict[str, Any]:
    """
    Interactive node that clarifies research scope with the user via terminal.
    Uses interrupts to pause execution and gather user input.
    """
    
    # Step 1: Get the research topic
    if not state.research_topic:
        topic_response = interrupt({
            "type": "topic_clarification",
            "question": "What topic would you like me to research?",
            "prompt": "Please provide the main topic or subject area for your research:"
        })
        research_topic = topic_response.strip()
    else:
        research_topic = state.research_topic
    
    # Step 2: Clarify the scope and depth
    scope_response = interrupt({
        "type": "scope_clarification", 
        "question": "What is the scope and depth of research needed?",
        "prompt": f"For the topic '{research_topic}', please specify:\n1. Scope (broad overview, specific aspect, comparison, etc.)\n2. Depth (surface-level, detailed analysis, academic depth)\n3. Any specific angles or perspectives to focus on:"
    })
    
    # Step 3: Identify target audience
    audience_response = interrupt({
        "type": "audience_clarification",
        "question": "Who is the target audience for this research?", 
        "prompt": "Please describe the target audience (e.g., general public, students, professionals, executives, researchers):"
    })
    
    # Step 4: Gather specific questions
    questions_response = interrupt({
        "type": "questions_clarification",
        "question": "What specific questions should the research address?",
        "prompt": "Please list any specific questions or areas of inquiry you'd like the research to address (separate multiple questions with semicolons):"
    })
    
    specific_questions = [q.strip() for q in questions_response.split(';') if q.strip()]
    
    # Step 5: Generate research brief
    research_brief = f"""
RESEARCH BRIEF

Topic: {research_topic}

Scope & Depth: {scope_response}

Target Audience: {audience_response}

Specific Questions to Address:
{chr(10).join(f"- {q}" for q in specific_questions)}

Research Approach: Comprehensive analysis using multiple sources, with focus on current information and diverse perspectives.
"""
    
    # Step 6: Get user approval for the brief
    approval_response = interrupt({
        "type": "brief_approval",
        "question": "Please review and approve the research brief",
        "research_brief": research_brief,
        "prompt": "Please review the research brief above. Type 'approve' to proceed with research, or provide modifications:"
    })
    
    user_approved = approval_response.lower().strip() == 'approve'
    
    if not user_approved:
        # If not approved, incorporate feedback and ask again
        revised_brief = f"""
RESEARCH BRIEF (Revised based on feedback)

Topic: {research_topic}

Scope & Depth: {scope_response}

Target Audience: {audience_response}

Specific Questions to Address:
{chr(10).join(f"- {q}" for q in specific_questions)}

User Feedback: {approval_response}

Research Approach: Comprehensive analysis using multiple sources, with focus on current information and diverse perspectives.
"""
        
        final_approval = interrupt({
            "type": "final_approval", 
            "question": "Please approve the revised research brief",
            "research_brief": revised_brief,
            "prompt": "Please review the revised brief. Type 'approve' to proceed:"
        })
        
        research_brief = revised_brief
        user_approved = final_approval.lower().strip() == 'approve'
    
    return {
        "research_topic": research_topic,
        "research_scope": scope_response,
        "target_audience": audience_response, 
        "specific_questions": specific_questions,
        "research_brief": research_brief,
        "user_approved": user_approved
    }


def conduct_research(state: ResearchState) -> Dict[str, Any]:
    """
    Node that uses a ReAct agent with Tavily search to conduct research
    based on the approved research brief.
    """
    
    # Set up the ReAct agent with Tavily search
    model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
    
    # Create Tavily search tool
    search_tool = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False
    )
    
    # Create ReAct agent with research-focused prompt
    research_prompt = f"""You are a professional research analyst conducting comprehensive research.

RESEARCH BRIEF:
{state.research_brief}

Your task is to:
1. Use the search tool to gather comprehensive information on the topic
2. Search for multiple perspectives and current information
3. Focus on addressing the specific questions in the brief
4. Gather information suitable for the target audience
5. Look for recent developments, statistics, expert opinions, and credible sources

For each search, be strategic about your queries to gather diverse and comprehensive information.
After gathering information, synthesize it into a detailed, well-structured research report.

Remember to:
- Use multiple search queries to cover different aspects
- Include current data and recent developments  
- Present balanced perspectives when relevant
- Cite key findings and insights
- Structure the information clearly for the target audience
"""
    
    # Create the ReAct agent
    react_agent = create_react_agent(
        model=model,
        tools=[search_tool],
        prompt=research_prompt
    )
    
    # Prepare the research request message
    research_request = f"""
Please conduct comprehensive research based on the following brief:

{state.research_brief}

Please be thorough in your research, using multiple search queries to gather comprehensive information. 
After completing your research, provide a detailed report that addresses all the points in the brief.
"""
    
    # Run the ReAct agent
    try:
        result = react_agent.invoke({
            "messages": [{"role": "user", "content": research_request}]
        })
        
        # Extract the final report from the agent's response
        if result and "messages" in result:
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                final_report = final_message.content
            else:
                final_report = str(final_message)
        else:
            final_report = "Research completed but no detailed report was generated."
            
    except Exception as e:
        final_report = f"Error during research: {str(e)}. Please check your API keys and try again."
    
    return {
        "final_report": final_report
    }


def should_proceed_with_research(state: ResearchState) -> str:
    """Conditional edge function to determine if research should proceed"""
    if state.user_approved:
        return "conduct_research"
    else:
        return END


# Create the main workflow graph
def create_research_workflow():
    """Create and return the compiled research workflow graph"""
    
    # Create the state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("clarify_scope", clarify_research_scope)
    workflow.add_node("conduct_research", conduct_research)
    
    # Add edges
    workflow.add_edge(START, "clarify_scope")
    workflow.add_conditional_edges(
        "clarify_scope",
        should_proceed_with_research,
        {
            "conduct_research": "conduct_research",
            END: END
        }
    )
    workflow.add_edge("conduct_research", END)
    
    # Compile with checkpointer (required for interrupts)
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Export the app for LangGraph platform deployment
app = create_research_workflow()


if __name__ == "__main__":
    # Example usage for local testing
    import uuid
    from langgraph.types import Command
    
    # Create a unique thread ID for this research session
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("🔬 Deep Research Agent")
    print("=" * 50)
    print("This agent will help you conduct comprehensive research.")
    print("It will first clarify your research needs, then conduct the research using advanced search tools.")
    print("=" * 50)
    
    try:
        # Start the workflow
        result = app.invoke({}, config=config)
        
        # Handle interrupts in a loop
        while "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0]
            interrupt_value = interrupt_data.value
            
            print(f"\n📝 {interrupt_value.get('question', 'Input required:')}")
            print("-" * 40)
            
            if interrupt_value.get("research_brief"):
                print(interrupt_value["research_brief"])
                print("-" * 40)
            
            user_input = input(f"{interrupt_value.get('prompt', 'Your input')}: ")
            
            # Resume with user input
            result = app.invoke(Command(resume=user_input), config=config)
        
        # Display final results
        if hasattr(result, 'final_report') and result.final_report:
            print("\n" + "=" * 50)
            print("📊 RESEARCH REPORT")
            print("=" * 50)
            print(result.final_report)
        else:
            print("\nResearch session completed.")
            
    except KeyboardInterrupt:
        print("\n\nResearch session interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")