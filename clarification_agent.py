from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from state import ResearchState


class ClarificationAgent:
    """Agent responsible for clarifying research scope with the user"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.system_prompt = """You are a research assistant helping to clarify the scope of a research task.
        Your goal is to gather enough information to create a clear research brief.
        
        Ask clarifying questions about:
        1. The main topic or question to research
        2. Specific aspects or subtopics to focus on
        3. Any constraints or requirements (e.g., time period, geography, industry)
        4. Desired depth and breadth of research
        5. Format or structure preferences for the final report
        
        Once you have enough information, summarize the research brief and ask for confirmation.
        When the user confirms, set clarification_complete to True."""
    
    def __call__(self, state: ResearchState) -> Dict[str, Any]:
        """Process the current state and generate clarifying questions or finalize the brief"""
        
        messages = state.get("messages", [])
        clarification_complete = state.get("clarification_complete", False)
        
        if not messages:
            initial_message = AIMessage(
                content="Hello! I'm here to help you with your research. Let's start by clarifying what you'd like me to research. What is the main topic or question you'd like me to investigate?"
            )
            return {
                "messages": [initial_message],
                "current_step": "clarification",
                "clarification_complete": False
            }
        
        # Check if the last user message confirms the brief
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        
        if last_user_message and any(
            keyword in last_user_message.content.lower() 
            for keyword in ["yes", "confirm", "correct", "proceed", "looks good", "perfect"]
        ):
            # Check if we already presented a brief
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and "Research Brief:" in msg.content:
                    # Extract the brief from the message
                    brief_start = msg.content.find("Research Brief:")
                    research_brief = msg.content[brief_start:]
                    
                    return {
                        "clarification_complete": True,
                        "research_brief": research_brief,
                        "current_step": "research",
                        "messages": [AIMessage(content="Great! I'll now proceed with the research based on this brief.")]
                    }
        
        # Continue clarification process
        system_message = SystemMessage(content=self.system_prompt)
        
        # Prepare messages for the LLM
        llm_messages = [system_message] + messages
        
        response = self.llm.invoke(llm_messages)
        
        # Check if the response contains a research brief
        if "Research Brief:" in response.content:
            return {
                "messages": [response],
                "current_step": "clarification",
                "clarification_complete": False
            }
        else:
            return {
                "messages": [response],
                "current_step": "clarification",
                "clarification_complete": False
            }