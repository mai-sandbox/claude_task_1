"""
LangGraph Deep Research Agent with Clarification and ReAct capabilities
"""

import operator
from typing import Annotated, TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class ResearchBrief(BaseModel):
    """Research brief generated from clarification phase"""
    topic: str = Field(description="Main research topic")
    scope: str = Field(description="Scope and boundaries of the research")
    specific_questions: List[str] = Field(description="Specific questions to answer")
    key_areas: List[str] = Field(description="Key areas to focus on")
    constraints: Optional[str] = Field(description="Any constraints or limitations")


class AgentState(TypedDict):
    """State shared between agents"""
    messages: Annotated[List[BaseMessage], operator.add]
    research_brief: Optional[ResearchBrief]
    clarification_complete: bool
    final_report: Optional[str]
    current_phase: Literal["clarification", "research", "complete"]


class ClarificationAgent:
    """Agent responsible for clarifying research scope with user"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research clarification specialist. Your job is to:
1. Understand what the user wants to research
2. Ask clarifying questions to define the scope
3. Identify specific questions to answer
4. Understand any constraints or limitations
5. Create a clear research brief when ready

Guidelines:
- Be concise but thorough
- Ask one or two questions at a time
- Focus on understanding the depth and breadth needed
- Clarify any ambiguous terms
- Confirm understanding before proceeding

When you have enough information, respond with "CLARIFICATION_COMPLETE" followed by a summary."""),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    def __call__(self, state: AgentState) -> AgentState:
        """Process clarification step"""
        messages = state["messages"]
        
        response = self.llm.invoke(self.clarification_prompt.format_messages(messages=messages))
        
        state["messages"].append(response)
        
        # Check if clarification is complete
        if "CLARIFICATION_COMPLETE" in response.content:
            # Extract research brief
            brief_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""Based on the conversation, create a structured research brief.
Extract the key information and format it properly."""),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessage(content="Create a structured research brief in JSON format with: topic, scope, specific_questions (list), key_areas (list), and constraints.")
            ])
            
            brief_response = self.llm.with_structured_output(ResearchBrief).invoke(
                brief_prompt.format_messages(messages=messages)
            )
            
            state["research_brief"] = brief_response
            state["clarification_complete"] = True
            state["current_phase"] = "research"
        
        return state


class ResearchAgent:
    """ReAct agent that performs the actual research"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
        
        self.tools = [self.search_tool]
        
        self.research_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert research agent. Based on the research brief provided:
1. Search for relevant information using the available tools
2. Analyze and synthesize findings
3. Answer all specified questions
4. Provide comprehensive coverage of key areas
5. Respect any stated constraints

Create a detailed, well-structured report with:
- Executive summary
- Detailed findings for each question/area
- Supporting evidence and sources
- Conclusions and insights

Be thorough but concise. Cite sources where applicable."""),
            HumanMessage(content="Research Brief: {brief}"),
            HumanMessage(content="Please conduct comprehensive research and create a detailed report."),
        ])
    
    def __call__(self, state: AgentState) -> AgentState:
        """Perform research based on the brief"""
        if not state.get("research_brief"):
            return state
        
        brief = state["research_brief"]
        
        # Create a ReAct agent for research
        react_agent = create_react_agent(
            self.llm,
            tools=self.tools,
            state_modifier=self.research_prompt.format_messages(
                brief=f"""
Topic: {brief.topic}
Scope: {brief.scope}
Questions: {', '.join(brief.specific_questions)}
Key Areas: {', '.join(brief.key_areas)}
Constraints: {brief.constraints or 'None'}
"""
            )
        )
        
        # Execute research
        research_messages = [
            HumanMessage(content=f"Research the following: {brief.topic}")
        ]
        
        # Run the ReAct agent
        result = react_agent.invoke({"messages": research_messages})
        
        # Extract the final report
        if result["messages"]:
            final_message = result["messages"][-1]
            state["final_report"] = final_message.content
            state["current_phase"] = "complete"
            state["messages"].append(AIMessage(content=f"Research Complete:\n\n{final_message.content}"))
        
        return state


def create_research_graph():
    """Create the main LangGraph workflow"""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize agents
    clarification_agent = ClarificationAgent(llm)
    research_agent = ResearchAgent(llm)
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_agent)
    workflow.add_node("research", research_agent)
    
    # Define routing logic
    def route_after_clarification(state: AgentState) -> str:
        if state.get("clarification_complete"):
            return "research"
        return "clarification"
    
    def route_after_research(state: AgentState) -> str:
        if state.get("final_report"):
            return END
        return "research"
    
    # Add edges
    workflow.set_entry_point("clarification")
    workflow.add_conditional_edges(
        "clarification",
        route_after_clarification,
        {
            "clarification": "clarification",
            "research": "research"
        }
    )
    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {
            "research": "research",
            END: END
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    return app


def run_research_agent():
    """Interactive function to run the research agent"""
    
    print("🔍 Deep Research Agent initialized")
    print("=" * 50)
    print("Please describe what you would like to research.")
    print("I'll ask clarifying questions to better understand your needs.\n")
    
    # Initialize the graph
    app = create_research_graph()
    
    # Initialize state
    state = {
        "messages": [],
        "research_brief": None,
        "clarification_complete": False,
        "final_report": None,
        "current_phase": "clarification"
    }
    
    # Initial user input
    user_input = input("👤 You: ")
    state["messages"] = [HumanMessage(content=user_input)]
    
    # Run clarification phase
    while state["current_phase"] == "clarification":
        result = app.invoke(state)
        
        # Display agent response
        last_message = result["messages"][-1]
        print(f"\n🤖 Agent: {last_message.content}\n")
        
        if not result.get("clarification_complete"):
            # Get user response
            user_input = input("👤 You: ")
            result["messages"].append(HumanMessage(content=user_input))
            state = result
        else:
            print("\n📋 Research Brief Created!")
            print("Starting research phase...\n")
            state = result
            break
    
    # Run research phase
    if state["current_phase"] == "research":
        print("🔬 Conducting research (this may take a moment)...\n")
        result = app.invoke(state)
        
        if result.get("final_report"):
            print("=" * 50)
            print("📊 RESEARCH REPORT")
            print("=" * 50)
            print(result["final_report"])
            print("=" * 50)
    
    return result


if __name__ == "__main__":
    run_research_agent()