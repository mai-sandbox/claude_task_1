from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from agents import ClarificationAgent, ReactResearchAgent, ResearchBrief
import os
from dotenv import load_dotenv

load_dotenv()


class OverallState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: ResearchBrief | None
    is_clarified: bool
    clarification_count: int
    research_report: str | None
    current_phase: Literal["clarification", "research", "complete"]


class ResearchOrchestrator:
    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.clarification_agent = ClarificationAgent(llm)
        self.react_agent = ReactResearchAgent(llm)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(OverallState)
        
        workflow.add_node("clarification", self.clarification_node)
        workflow.add_node("check_clarification", self.check_clarification_node)
        workflow.add_node("research", self.research_node)
        workflow.add_node("generate_report", self.generate_report_node)
        
        workflow.set_entry_point("clarification")
        
        workflow.add_edge("clarification", "check_clarification")
        
        workflow.add_conditional_edges(
            "check_clarification",
            self.should_continue_clarification,
            {
                "continue": "clarification",
                "proceed": "research"
            }
        )
        
        workflow.add_edge("research", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def clarification_node(self, state: OverallState) -> OverallState:
        clarification_state = {
            "messages": state.get("messages", []),
            "research_brief": state.get("research_brief"),
            "is_clarified": state.get("is_clarified", False),
            "clarification_count": state.get("clarification_count", 0)
        }
        
        result = self.clarification_agent.ask_clarification(clarification_state)
        
        return {
            **state,
            "messages": result["messages"],
            "clarification_count": result["clarification_count"],
            "current_phase": "clarification"
        }
    
    def check_clarification_node(self, state: OverallState) -> OverallState:
        last_message = state["messages"][-1] if state["messages"] else None
        
        if isinstance(last_message, HumanMessage):
            clarification_state = {
                "messages": state["messages"],
                "research_brief": state.get("research_brief"),
                "is_clarified": state.get("is_clarified", False),
                "clarification_count": state.get("clarification_count", 0)
            }
            
            result = self.clarification_agent.check_if_clarified(clarification_state)
            
            return {
                **state,
                "messages": result["messages"],
                "research_brief": result.get("research_brief"),
                "is_clarified": result.get("is_clarified", False),
                "clarification_count": result.get("clarification_count", 0)
            }
        
        return state
    
    def should_continue_clarification(self, state: OverallState) -> Literal["continue", "proceed"]:
        if state.get("is_clarified", False):
            return "proceed"
        
        if state.get("clarification_count", 0) >= 5:
            return "proceed"
        
        return "continue"
    
    def research_node(self, state: OverallState) -> OverallState:
        research_state = {
            "messages": [],
            "research_brief": state["research_brief"],
            "research_report": None,
            "search_results": []
        }
        
        result = self.react_agent.research(research_state)
        
        return {
            **state,
            "messages": state["messages"] + result["messages"],
            "current_phase": "research"
        }
    
    def generate_report_node(self, state: OverallState) -> OverallState:
        research_state = {
            "messages": state["messages"],
            "research_brief": state["research_brief"],
            "research_report": None,
            "search_results": []
        }
        
        result = self.react_agent.generate_report(research_state)
        
        return {
            **state,
            "research_report": result["research_report"],
            "messages": state["messages"] + [AIMessage(content=result["research_report"])],
            "current_phase": "complete"
        }
    
    def run_interactive(self):
        config = {"configurable": {"thread_id": "research_session"}}
        
        print("Research Assistant: Hello! I'm here to help you with your research.")
        print("Research Assistant: Could you please tell me what topic you'd like me to research?")
        print("-" * 50)
        
        state = {
            "messages": [],
            "research_brief": None,
            "is_clarified": False,
            "clarification_count": 0,
            "research_report": None,
            "current_phase": "clarification"
        }
        
        while state.get("current_phase") != "complete":
            if state.get("current_phase") == "clarification":
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("Research Assistant: Goodbye!")
                    break
                
                state["messages"].append(HumanMessage(content=user_input))
                
                result = self.graph.invoke(state, config)
                state = result
                
                last_ai_message = None
                for msg in reversed(state["messages"]):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break
                
                if last_ai_message:
                    print(f"Research Assistant: {last_ai_message.content}")
                    print("-" * 50)
            
            elif state.get("current_phase") == "research":
                print("\nResearch Assistant: Now conducting deep research based on your brief...")
                print("This may take a few moments as I search for comprehensive information...")
                print("-" * 50)
                
                result = self.graph.invoke(state, config)
                state = result
        
        if state.get("research_report"):
            print("\n" + "=" * 50)
            print("RESEARCH REPORT")
            print("=" * 50)
            print(state["research_report"])
            print("=" * 50)
            
            with open("research_report.md", "w") as f:
                f.write(state["research_report"])
            print("\nReport saved to 'research_report.md'")
    
    def run_batch(self, topic: str, objectives: list, questions: list) -> str:
        config = {"configurable": {"thread_id": "batch_research"}}
        
        research_brief = ResearchBrief(
            topic=topic,
            objectives=objectives,
            scope="Comprehensive research within the defined objectives",
            key_questions=questions,
            constraints=[]
        )
        
        state = {
            "messages": [],
            "research_brief": research_brief,
            "is_clarified": True,
            "clarification_count": 0,
            "research_report": None,
            "current_phase": "research"
        }
        
        print(f"Starting research on: {topic}")
        print("Conducting deep research...")
        
        final_state = self.graph.invoke(state, config)
        
        return final_state.get("research_report", "No report generated")


def create_app():
    return ResearchOrchestrator().graph


if __name__ == "__main__":
    orchestrator = ResearchOrchestrator()
    orchestrator.run_interactive()