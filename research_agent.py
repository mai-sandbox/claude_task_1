from typing import Annotated, TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class ResearchBrief(BaseModel):
    """Research brief generated from clarification phase"""
    topic: str = Field(description="Main research topic")
    scope: str = Field(description="Scope and boundaries of the research")
    key_questions: List[str] = Field(description="Key questions to answer")
    context: str = Field(description="Additional context and requirements")


class AgentState(TypedDict):
    """Main state for the research agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    research_brief: Optional[ResearchBrief]
    research_report: Optional[str]
    phase: Literal["clarification", "research", "complete"]
    clarification_complete: bool


class ClarificationAgent:
    """Agent that clarifies research requirements with the user"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.system_prompt = """You are a research assistant helping to clarify the scope of a research project.
        Your goal is to have a brief conversation with the user to understand:
        1. The main topic they want to research
        2. The specific aspects or questions they want answered
        3. Any constraints or specific requirements
        4. The depth and breadth of research needed
        
        Ask clarifying questions to ensure you have a complete understanding.
        Once you have enough information, create a structured research brief.
        
        When the user confirms they're ready to proceed, set clarification_complete to True."""
    
    def __call__(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        
        # Add system message if this is the first interaction
        if len(messages) == 1:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        # Check if we should generate a brief
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            content = last_message.content.lower()
            if any(phrase in content for phrase in ["yes", "proceed", "looks good", "go ahead", "start research", "confirmed"]):
                # Generate research brief
                brief_prompt = """Based on our conversation, create a structured research brief with:
                - Main topic
                - Scope and boundaries
                - Key questions to answer
                - Additional context
                
                Return as JSON."""
                
                brief_messages = messages + [HumanMessage(content=brief_prompt)]
                response = self.llm.with_structured_output(ResearchBrief).invoke(brief_messages)
                
                return {
                    **state,
                    "research_brief": response,
                    "clarification_complete": True,
                    "messages": messages + [AIMessage(content=f"Research brief created:\n\nTopic: {response.topic}\n\nScope: {response.scope}\n\nKey Questions:\n" + "\n".join(f"- {q}" for q in response.key_questions) + f"\n\nContext: {response.context}\n\nStarting research phase...")]
                }
        
        # Continue clarification conversation
        response = self.llm.invoke(messages)
        
        return {
            **state,
            "messages": messages + [response],
            "clarification_complete": False
        }


class ResearchAgent:
    """ReAct agent that performs the actual research"""
    
    def __init__(self):
        # Initialize Tavily search tool
        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        # Create ReAct agent with search tool
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        self.agent = create_react_agent(
            self.llm,
            tools=[self.search_tool],
            state_modifier=self._create_system_message
        )
    
    def _create_system_message(self, state: AgentState) -> List[BaseMessage]:
        """Create system message with research brief"""
        brief = state.get("research_brief")
        if not brief:
            return state["messages"]
        
        system_message = f"""You are a research agent tasked with conducting detailed research based on the following brief:

Topic: {brief.topic}
Scope: {brief.scope}

Key Questions to Answer:
{chr(10).join(f"- {q}" for q in brief.key_questions)}

Context: {brief.context}

Use the search tool to gather comprehensive information. Be thorough and systematic in your research.
Once you have gathered sufficient information, synthesize it into a detailed report that addresses all the key questions.

Important: Your final message should be a comprehensive research report that answers all questions and provides valuable insights."""
        
        return [SystemMessage(content=system_message)]
    
    def __call__(self, state: AgentState) -> AgentState:
        # Run the ReAct agent
        result = self.agent.invoke(state)
        
        # Extract the research report from the last message
        if result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                # Check if this looks like a final report
                content = last_message.content
                if len(content) > 500 and any(word in content.lower() for word in ["conclusion", "summary", "findings", "report"]):
                    result["research_report"] = content
                    result["phase"] = "complete"
        
        return result


def create_research_graph():
    """Create the main research agent graph"""
    
    # Initialize components
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    clarification_agent = ClarificationAgent(llm)
    research_agent = ResearchAgent()
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("clarification", clarification_agent)
    workflow.add_node("research", research_agent)
    
    # Define routing logic
    def route_after_clarification(state: AgentState) -> str:
        if state.get("clarification_complete", False):
            return "research"
        return "clarification"
    
    def route_after_research(state: AgentState) -> str:
        if state.get("phase") == "complete":
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
    
    # Compile graph
    return workflow.compile()


def main():
    """Main function to run the research agent"""
    print("Research Agent initialized. Let's clarify your research needs.\n")
    
    # Create the graph
    app = create_research_graph()
    
    # Initialize state
    initial_state = {
        "messages": [],
        "research_brief": None,
        "research_report": None,
        "phase": "clarification",
        "clarification_complete": False
    }
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting research agent.")
            break
        
        # Add user message to state
        initial_state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph
        result = app.invoke(initial_state)
        initial_state = result
        
        # Display the last AI message
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                print(f"\nAgent: {message.content}")
                break
        
        # Check if research is complete
        if result.get("phase") == "complete":
            print("\n" + "="*50)
            print("RESEARCH COMPLETE")
            print("="*50)
            if result.get("research_report"):
                print("\nFinal Report:")
                print(result["research_report"])
            break


if __name__ == "__main__":
    main()