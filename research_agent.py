from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from tools import create_tavily_search_tool, format_search_results
from state import ResearchState
import json


class ResearchAgent:
    """ReAct agent responsible for conducting research using tools"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.search_tool = create_tavily_search_tool(max_results=5)
        self.tools = [self.search_tool, format_search_results]
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.system_prompt = """You are an expert research agent tasked with conducting thorough research based on the provided brief.
        
        Use the Tavily search tool to gather information from the web. Be systematic and comprehensive in your research:
        1. Break down the research brief into key questions or topics
        2. Search for each topic systematically
        3. Gather diverse perspectives and sources
        4. Verify important facts by finding multiple sources
        5. Synthesize the information into a comprehensive report
        
        Your final report should be:
        - Well-structured with clear sections
        - Based on factual information from your searches
        - Include citations (URLs) for key claims
        - Provide balanced and objective analysis
        - Address all aspects mentioned in the research brief
        
        Research Brief:
        {research_brief}
        
        Conduct thorough research and create a detailed report. You may perform multiple searches to gather comprehensive information."""
    
    def __call__(self, state: ResearchState) -> Dict[str, Any]:
        """Execute research based on the brief"""
        
        research_brief = state.get("research_brief", "")
        iteration_count = state.get("iteration_count", 0)
        search_results = state.get("search_results", [])
        
        if not research_brief:
            return {
                "messages": [AIMessage(content="No research brief provided. Please complete the clarification step first.")],
                "current_step": "error"
            }
        
        # Limit iterations to prevent infinite loops
        if iteration_count >= 10:
            # Generate final report
            report = self._generate_final_report(research_brief, search_results)
            return {
                "research_report": report,
                "messages": [AIMessage(content=report)],
                "current_step": "completed"
            }
        
        # Prepare the prompt
        system_message = SystemMessage(
            content=self.system_prompt.format(research_brief=research_brief)
        )
        
        # Get existing messages
        messages = state.get("messages", [])
        
        # If this is the first iteration, start fresh
        if iteration_count == 0:
            messages = [system_message]
        
        # Invoke the LLM with tools
        response = self.llm_with_tools.invoke(messages)
        
        # Check if the agent wants to use a tool
        if response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call["name"] == "tavily_search_results":
                    # Execute the search
                    search_query = tool_call["args"].get("query", "")
                    try:
                        results = self.search_tool.invoke({"query": search_query})
                        search_results.extend(results)
                        tool_messages.append(
                            ToolMessage(
                                content=json.dumps(results),
                                tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error performing search: {str(e)}",
                                tool_call_id=tool_call["id"]
                            )
                        )
            
            # Continue the conversation with tool results
            return {
                "messages": [response] + tool_messages,
                "iteration_count": iteration_count + 1,
                "search_results": search_results,
                "current_step": "research"
            }
        else:
            # Agent is done with research, check if report is in the response
            if "## Research Report" in response.content or "# Research Report" in response.content:
                return {
                    "research_report": response.content,
                    "messages": [response],
                    "current_step": "completed"
                }
            else:
                # Continue researching
                return {
                    "messages": [response],
                    "iteration_count": iteration_count + 1,
                    "search_results": search_results,
                    "current_step": "research"
                }
    
    def _generate_final_report(self, brief: str, search_results: List[dict]) -> str:
        """Generate a final research report based on collected information"""
        
        # Summarize search results
        results_summary = []
        for result in search_results[:20]:  # Limit to avoid token overflow
            results_summary.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "key_points": result.get("content", "")[:300]
            })
        
        prompt = f"""Based on the following research brief and search results, create a comprehensive research report.
        
        Research Brief:
        {brief}
        
        Search Results Summary:
        {json.dumps(results_summary, indent=2)}
        
        Create a detailed, well-structured report that addresses all aspects of the research brief.
        Include citations and organize the information logically."""
        
        response = self.llm.invoke([SystemMessage(content=prompt)])
        return response.content