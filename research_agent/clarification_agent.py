from typing import Dict, Any
import sys
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from research_agent.state import ResearchState


class ClarificationAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7
        )
    
    def clarify_research_scope(self, state: ResearchState) -> Dict[str, Any]:
        print("\n🔍 Research Scope Clarification Agent")
        print("=" * 50)
        
        if not state.get("clarification_complete", False):
            user_input = self._get_initial_request(state)
            
            while True:
                clarifying_response = self._generate_clarifying_questions(user_input)
                print(f"\nAgent: {clarifying_response}")
                
                user_response = input("\nYou: ")
                
                if user_response.lower() in ["done", "complete", "finished", "ready"]:
                    break
                
                user_input += f"\nUser clarification: {user_response}"
            
            research_brief = self._generate_research_brief(user_input)
            
            print(f"\n📋 Research Brief Generated:")
            print("-" * 30)
            print(research_brief)
            print("-" * 30)
            
            confirmation = input("\nIs this research brief accurate? (yes/no): ")
            
            if confirmation.lower() in ["yes", "y"]:
                return {
                    "research_brief": research_brief,
                    "clarification_complete": True,
                    "messages": [AIMessage(content=f"Research brief created: {research_brief}")]
                }
            else:
                print("\nLet's refine the research scope...")
                return self.clarify_research_scope(state)
        
        return {"clarification_complete": True}
    
    def _get_initial_request(self, state: ResearchState) -> str:
        if state.get("messages"):
            latest_message = state["messages"][-1]
            if hasattr(latest_message, 'content'):
                return latest_message.content
        
        print("Welcome! I'm here to help you conduct thorough research.")
        user_input = input("What would you like me to research for you? ")
        return user_input
    
    def _generate_clarifying_questions(self, context: str) -> str:
        prompt = f"""Based on this research request: "{context}"

Generate 2-3 specific clarifying questions to better understand:
1. The scope and depth of research needed
2. The intended audience or use case
3. Any specific aspects or angles to focus on
4. Timeline or urgency considerations

Keep questions concise and focused. If the user's request is already very specific, acknowledge that and ask if there are any additional considerations."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_research_brief(self, full_context: str) -> str:
        prompt = f"""Based on this conversation: "{full_context}"

Create a comprehensive research brief that includes:
1. Research objective (what exactly needs to be researched)
2. Key questions to answer
3. Target audience/use case
4. Scope and boundaries
5. Expected deliverables

Format this as a clear, structured brief that a research agent can use to conduct thorough research."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content