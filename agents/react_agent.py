from typing import List, Dict, Any, TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from agents.clarification_agent import ResearchBrief
import operator


class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: ResearchBrief
    research_report: str | None
    search_results: List[Dict[str, Any]]


class ReactResearchAgent:
    def __init__(self, llm: ChatOpenAI = None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.2)
        self.tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        self.tools = [self.tavily_tool]
        
    def create_agent(self):
        return create_react_agent(
            self.llm,
            self.tools,
            state_modifier=self._create_system_prompt
        )
    
    def _create_system_prompt(self, state: ResearchState) -> List[BaseMessage]:
        research_brief = state.get("research_brief")
        
        if not research_brief:
            return state["messages"]
        
        system_prompt = f"""You are an expert research agent conducting deep research based on the following brief:

**Topic:** {research_brief.topic}

**Objectives:**
{chr(10).join(f"- {obj}" for obj in research_brief.objectives)}

**Scope:** {research_brief.scope}

**Key Questions to Answer:**
{chr(10).join(f"- {q}" for q in research_brief.key_questions)}

**Constraints:**
{chr(10).join(f"- {c}" for c in research_brief.constraints) if research_brief.constraints else "None"}

Your task is to:
1. Use the Tavily search tool to gather comprehensive information
2. Search for multiple perspectives and sources
3. Verify facts across multiple sources when possible
4. Focus on answering the key questions thoroughly
5. Stay within the defined scope and constraints

Conduct multiple searches as needed to gather comprehensive information. Think step by step about what information you need and search strategically."""
        
        return [SystemMessage(content=system_prompt)] + state["messages"]
    
    def research(self, state: ResearchState) -> ResearchState:
        research_brief = state["research_brief"]
        
        initial_message = HumanMessage(
            content=f"Please conduct comprehensive research on: {research_brief.topic}. "
            f"Focus on these objectives: {', '.join(research_brief.objectives[:3])}. "
            f"Answer these key questions: {', '.join(research_brief.key_questions[:3])}"
        )
        
        agent = self.create_agent()
        
        result = agent.invoke({
            "messages": [initial_message],
            "research_brief": research_brief
        })
        
        return {
            **state,
            "messages": result["messages"],
            "search_results": self._extract_search_results(result["messages"])
        }
    
    def _extract_search_results(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        search_results = []
        for msg in messages:
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    if tool_call.get('name') == 'tavily_search_results_json':
                        search_results.extend(tool_call.get('results', []))
        return search_results
    
    def generate_report(self, state: ResearchState) -> ResearchState:
        research_brief = state["research_brief"]
        messages = state["messages"]
        
        report_prompt = f"""Based on all the research conducted, generate a comprehensive and detailed report.

The report should:
1. Begin with an executive summary
2. Address each objective from the research brief
3. Answer all key questions thoroughly
4. Include relevant findings and insights
5. Cite sources where appropriate
6. End with conclusions and recommendations

Research Brief Reminder:
- Topic: {research_brief.topic}
- Objectives: {', '.join(research_brief.objectives)}
- Key Questions: {', '.join(research_brief.key_questions)}

Format the report with clear headings and subheadings using markdown."""
        
        report_messages = messages + [HumanMessage(content=report_prompt)]
        
        response = self.llm.invoke(report_messages)
        
        return {
            **state,
            "research_report": response.content,
            "messages": messages + [HumanMessage(content=report_prompt), response]
        }