from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from state import AgentState, ResearchBrief
from tools import get_search_tool
import json


class ResearchAgent:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        self.react_prompt = PromptTemplate.from_template("""You are a research agent with access to web search.
        
Research Brief:
Topic: {topic}
Questions to Answer: {questions}
Scope: {scope}
Output Requirements: {output_requirements}

You have access to the following tool:
{tools}

Use the following format:

Thought: Think about what information you need to gather
Action: the action to take (must be one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to write the report
Final Answer: [Comprehensive research report]

Begin your research:

{agent_scratchpad}""")
        
        self.search_tool = get_search_tool()
        self.tools = [self.search_tool]
        
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.react_prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
        
        self.report_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research report writer. Based on the research brief and search results,
            create a comprehensive, well-structured research report.
            
            The report should:
            1. Have a clear executive summary
            2. Address all specific questions from the brief
            3. Include relevant findings from the search results
            4. Be well-organized with sections and subsections
            5. Include citations to sources
            6. Meet the output requirements specified in the brief
            
            Research Brief:
            {research_brief}
            
            Search Results:
            {search_results}"""),
            ("human", "Create a comprehensive research report based on the above information.")
        ])
    
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        research_brief = state.get("research_brief")
        
        if not research_brief:
            return {
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": "No research brief found. Please complete the clarification phase first."
                }],
                "current_phase": "clarification"
            }
        
        try:
            questions_str = "\n".join([f"- {q}" for q in research_brief.specific_questions])
            
            result = self.agent_executor.invoke({
                "topic": research_brief.topic,
                "questions": questions_str,
                "scope": research_brief.scope,
                "output_requirements": research_brief.output_requirements
            })
            
            search_results = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) > 1 and isinstance(step[1], (list, str)):
                        search_results.append(str(step[1]))
            
            research_report = result.get("output", "")
            
            if not research_report or len(research_report) < 100:
                research_report = self.llm.invoke(
                    self.report_prompt.format(
                        research_brief=json.dumps({
                            "topic": research_brief.topic,
                            "questions": research_brief.specific_questions,
                            "scope": research_brief.scope,
                            "output_requirements": research_brief.output_requirements
                        }, indent=2),
                        search_results="\n\n".join(search_results) if search_results else "No search results available"
                    )
                ).content
            
            return {
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": f"Research completed! Here's the detailed report:\n\n{research_report}"
                }],
                "research_report": research_report,
                "search_results": search_results,
                "current_phase": "complete"
            }
            
        except Exception as e:
            return {
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": f"Error during research: {str(e)}"
                }],
                "current_phase": "complete"
            }