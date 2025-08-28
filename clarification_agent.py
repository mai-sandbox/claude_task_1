from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState, ResearchBrief
import json


class ClarificationAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant helping to clarify the scope of a research project.
            Your goal is to have a back-and-forth conversation with the user to understand:
            1. The main research topic
            2. Specific questions they want answered
            3. The scope and boundaries of the research
            4. Any specific requirements for the output
            
            Ask clarifying questions one at a time. Be conversational and helpful.
            When you have enough information to create a comprehensive research brief, 
            respond with "CLARIFICATION_COMPLETE" followed by a JSON research brief.
            
            Current conversation:
            {conversation_history}"""),
            ("human", "{user_input}")
        ])
        
        self.brief_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract a research brief from the conversation.
            Create a JSON object with:
            - topic: main research topic
            - specific_questions: list of specific questions to answer
            - scope: scope and boundaries of research
            - output_requirements: requirements for the final output
            
            Conversation:
            {conversation_history}
            
            Return ONLY the JSON object."""),
            ("human", "Create the research brief")
        ])
    
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        
        if not messages:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "Hello! I'm here to help you with your research. What topic would you like me to research for you?"
                }],
                "current_phase": "clarification"
            }
        
        last_message = messages[-1]
        
        if last_message.get("role") == "user":
            conversation_history = "\n".join([
                f"{msg['role']}: {msg['content']}" for msg in messages
            ])
            
            response = self.llm.invoke(
                self.clarification_prompt.format(
                    conversation_history=conversation_history,
                    user_input=last_message["content"]
                )
            )
            
            response_content = response.content
            
            if "CLARIFICATION_COMPLETE" in response_content:
                brief_json = self.llm.invoke(
                    self.brief_extraction_prompt.format(
                        conversation_history=conversation_history
                    )
                )
                
                try:
                    brief_data = json.loads(brief_json.content)
                    research_brief = ResearchBrief(**brief_data)
                    
                    return {
                        "messages": messages + [{
                            "role": "assistant",
                            "content": f"Great! I have all the information I need. Here's the research brief:\n\n"
                                     f"**Topic:** {research_brief.topic}\n"
                                     f"**Questions:** {', '.join(research_brief.specific_questions)}\n"
                                     f"**Scope:** {research_brief.scope}\n"
                                     f"**Output Requirements:** {research_brief.output_requirements}\n\n"
                                     f"Starting the research now..."
                        }],
                        "research_brief": research_brief,
                        "current_phase": "research",
                        "clarification_complete": True
                    }
                except Exception as e:
                    print(f"Error parsing brief: {e}")
            
            return {
                "messages": messages + [{
                    "role": "assistant",
                    "content": response_content
                }],
                "current_phase": "clarification"
            }
        
        return state