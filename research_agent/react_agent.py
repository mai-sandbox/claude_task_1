from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from research_agent.tools import tavily_tool
from research_agent.state import ResearchState
import json


class ReactAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1
        ).bind_tools([tavily_tool])
        self.max_iterations = 10
    
    def conduct_research(self, state: ResearchState) -> Dict[str, Any]:
        print("\n🔬 ReAct Research Agent Starting")
        print("=" * 50)
        
        if not state.get("research_brief"):
            return {"final_report": "No research brief provided"}
        
        research_brief = state["research_brief"]
        print(f"Research Brief: {research_brief}")
        
        messages = [
            HumanMessage(content=f"""You are a thorough research agent. Your task is to conduct comprehensive research based on this brief:

{research_brief}

Use the search tool to gather information. Think step by step:
1. Identify key search queries needed to address the research brief
2. Search for information using the available tools
3. Analyze and synthesize the findings
4. Generate a comprehensive report

Start by planning your research approach and then execute the searches.""")
        ]
        
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Research Iteration {iteration} ---")
            
            response = self.llm.invoke(messages)
            messages.append(response)
            
            print(f"Agent Reasoning: {response.content}")
            
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    print(f"🔍 Executing search: {tool_call['args']['query']}")
                    
                    try:
                        tool_result = tavily_tool.invoke(tool_call["args"])
                        tool_message = ToolMessage(
                            content=json.dumps(tool_result, indent=2),
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)
                        
                        print(f"✅ Search completed. Found {len(tool_result.get('results', []))} results")
                        
                    except Exception as e:
                        error_message = ToolMessage(
                            content=f"Error executing search: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(error_message)
                        print(f"❌ Search failed: {str(e)}")
            
            else:
                print("🏁 Research complete - generating final report")
                break
        
        final_response = self.llm.invoke(messages + [
            HumanMessage(content="""Based on all the research you've conducted, please generate a comprehensive final report that addresses the research brief. The report should be well-structured, include key findings, insights, and conclusions. Make it detailed and informative.""")
        ])
        
        final_report = final_response.content
        
        print(f"\n📊 Final Research Report Generated")
        print("=" * 50)
        print(final_report)
        print("=" * 50)
        
        return {
            "final_report": final_report,
            "messages": [AIMessage(content=f"Research completed. Final report: {final_report}")]
        }