# LangGraph Research Agent Deployment Guide

## Local Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# - OPENAI_API_KEY
# - TAVILY_API_KEY
# - LANGCHAIN_API_KEY (for LangGraph Cloud)
```

3. **Run locally:**
```bash
# Interactive terminal mode
python graph.py

# Test mode
python test_agent.py
```

## LangGraph Platform Deployment

### Prerequisites
- LangGraph Cloud account
- LangGraph CLI installed (`pip install -U langgraph-cli`)

### Deployment Steps

1. **Login to LangGraph:**
```bash
langgraph auth login
```

2. **Deploy the agent:**
```bash
langgraph deploy
```

3. **Test the deployment:**
```bash
langgraph test research_agent
```

### Configuration Files

- `langgraph.json`: Defines the graph configuration for deployment
- `app.py`: Entry point for the LangGraph platform
- `pyproject.toml`: Python project configuration with dependencies

### API Usage

Once deployed, you can interact with the agent via API:

```python
import requests

# Start a conversation
response = requests.post(
    "https://api.langgraph.com/v1/threads",
    headers={"Authorization": f"Bearer {YOUR_API_KEY}"},
    json={
        "graph": "research_agent",
        "input": {
            "messages": [{"role": "user", "content": "Research AI trends"}]
        }
    }
)
```

## Architecture

The agent consists of:

1. **Clarification Agent**: Conducts initial conversation to understand research requirements
2. **Research Agent**: Uses Tavily search to gather information and generate reports
3. **State Management**: Maintains conversation state and research progress
4. **Graph Orchestration**: Routes between agents based on conversation phase

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: For LLM operations
- `TAVILY_API_KEY`: For web search functionality
- `LANGCHAIN_API_KEY`: For LangGraph Cloud deployment
- `LANGCHAIN_TRACING_V2`: Set to "true" for debugging
- `LANGCHAIN_PROJECT`: Project name for tracing