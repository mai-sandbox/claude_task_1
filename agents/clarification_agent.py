from typing import List, Dict, Any, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    topic: str = Field(description="Main research topic")
    objectives: List[str] = Field(description="Specific research objectives")
    scope: str = Field(description="Scope and boundaries of the research")
    key_questions: List[str] = Field(description="Key questions to answer")
    constraints: List[str] = Field(description="Any constraints or limitations")


class ClarificationState(TypedDict):
    messages: List[Any]
    research_brief: ResearchBrief | None
    is_clarified: bool
    clarification_count: int


class ClarificationAgent:
    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.max_clarifications = 5
        
    def get_initial_prompt(self) -> str:
        return """You are a research assistant helping to clarify the scope of a research project.
        Your goal is to understand:
        1. The main topic to research
        2. Specific objectives and goals
        3. The scope and boundaries
        4. Key questions to answer
        5. Any constraints or limitations
        
        Ask clarifying questions to gather this information. Be conversational but focused."""
    
    def ask_clarification(self, state: ClarificationState) -> ClarificationState:
        messages = state["messages"]
        clarification_count = state.get("clarification_count", 0)
        
        if not messages:
            messages = [SystemMessage(content=self.get_initial_prompt())]
        
        if clarification_count == 0:
            response = AIMessage(content="Hello! I'm here to help you with your research. Could you please tell me what topic you'd like me to research?")
        else:
            prompt = self._build_clarification_prompt(messages)
            response = self.llm.invoke(prompt)
        
        messages.append(response)
        
        return {
            **state,
            "messages": messages,
            "clarification_count": clarification_count + 1
        }
    
    def _build_clarification_prompt(self, messages: List[Any]) -> List[Any]:
        system_prompt = """Based on the conversation so far, ask a clarifying question to better understand the research requirements.
        Focus on gathering information about:
        - Specific aspects of the topic that need investigation
        - The depth of analysis required
        - Any specific sources or types of information preferred
        - Time constraints or urgency
        - Expected format or structure of the final report
        
        Keep questions concise and focused. If you have enough information, indicate that you're ready to proceed."""
        
        return [SystemMessage(content=system_prompt)] + messages
    
    def process_user_input(self, state: ClarificationState, user_input: str) -> ClarificationState:
        messages = state["messages"]
        messages.append(HumanMessage(content=user_input))
        
        return {
            **state,
            "messages": messages
        }
    
    def check_if_clarified(self, state: ClarificationState) -> ClarificationState:
        messages = state["messages"]
        clarification_count = state.get("clarification_count", 0)
        
        if clarification_count >= self.max_clarifications:
            return self._finalize_brief(state, force=True)
        
        check_prompt = [
            SystemMessage(content="""Analyze the conversation and determine if you have enough information to create a comprehensive research brief.
            You need:
            1. A clear topic
            2. At least 2-3 specific objectives
            3. Understanding of scope
            4. Some key questions to answer
            
            Respond with 'READY' if you have enough information, or 'NEED_MORE' if you need more clarification.""")
        ] + messages
        
        response = self.llm.invoke(check_prompt)
        
        if "READY" in response.content.upper():
            return self._finalize_brief(state)
        
        return state
    
    def _finalize_brief(self, state: ClarificationState, force: bool = False) -> ClarificationState:
        messages = state["messages"]
        
        brief_prompt = [
            SystemMessage(content="""Based on the conversation, create a comprehensive research brief.
            Extract and structure the following information:
            1. Main topic
            2. Specific objectives (list them)
            3. Scope and boundaries
            4. Key questions to answer
            5. Any constraints or limitations mentioned
            
            If some information is missing, make reasonable assumptions based on the context.""")
        ] + messages
        
        structured_llm = self.llm.with_structured_output(ResearchBrief)
        research_brief = structured_llm.invoke(brief_prompt)
        
        confirmation_msg = f"""Great! I've prepared the research brief:

**Topic:** {research_brief.topic}

**Objectives:**
{chr(10).join(f"- {obj}" for obj in research_brief.objectives)}

**Scope:** {research_brief.scope}

**Key Questions:**
{chr(10).join(f"- {q}" for q in research_brief.key_questions)}

**Constraints:**
{chr(10).join(f"- {c}" for c in research_brief.constraints) if research_brief.constraints else "None specified"}

I'll now proceed with the research based on this brief."""
        
        messages.append(AIMessage(content=confirmation_msg))
        
        return {
            **state,
            "messages": messages,
            "research_brief": research_brief,
            "is_clarified": True
        }