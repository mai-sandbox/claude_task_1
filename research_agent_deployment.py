"""
Deployment-ready version of the research agent for LangGraph platform
"""

from typing import Annotated, TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import os

# The app instance that will be deployed
from research_agent import create_research_graph

# Create the app instance for deployment
app = create_research_graph()

# This is the entry point for LangGraph platform
__all__ = ["app"]