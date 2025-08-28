# LangGraph Deep Research Agent

A sophisticated research agent built with LangGraph that conducts interactive clarification sessions before performing deep research using Tavily search.

## Architecture

The agent consists of two main components:

1. **Clarification Agent**: Engages in back-and-forth conversation to understand research requirements
2. **ReAct Research Agent**: Uses Tavily search to conduct research and generate comprehensive reports

## Features

- Interactive clarification phase to define research scope
- Structured research brief generation
- ReAct-based research with Tavily search integration
- Comprehensive report generation
- Deployable on LangGraph Cloud Platform

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Required API keys:
- `OPENAI_API_KEY`: OpenAI API key for LLM
- `TAVILY_API_KEY`: Tavily API key for search
- `LANGCHAIN_API_KEY`: LangChain API key for tracing (optional)

### 3. Test Locally

Run the interactive agent:
```bash
python research_agent.py
```

Run tests:
```bash
python test_agent.py
```

## Deployment to LangGraph Cloud

### Prerequisites

1. Install LangGraph CLI:
```bash
pip install langgraph-cli
```

2. Login to LangGraph Cloud:
```bash
langgraph auth login
```

### Deploy

1. Update `langgraph.json` with your environment variables

2. Deploy to LangGraph Cloud:
```bash
langgraph deploy
```

3. The deployment will provide you with an endpoint URL

### Using the Deployed Agent

Send POST requests to your deployment endpoint:

```python
import requests

response = requests.post(
    "https://your-deployment-url/invoke",
    headers={"Authorization": f"Bearer {YOUR_API_KEY}"},
    json={
        "messages": ["I want to research AI in healthcare"],
        "current_phase": "clarification"
    }
)
```

## Usage Example

```python
from research_agent import create_research_graph
from langchain_core.messages import HumanMessage

# Create the agent
app = create_research_graph()

# Initialize conversation
state = {
    "messages": [HumanMessage(content="Research renewable energy trends")],
    "research_brief": None,
    "clarification_complete": False,
    "final_report": None,
    "current_phase": "clarification"
}

# Run the agent
result = app.invoke(state)
```

## State Schema

The agent maintains the following state:

- `messages`: Conversation history
- `research_brief`: Structured brief from clarification
- `clarification_complete`: Boolean flag for phase transition
- `final_report`: Generated research report
- `current_phase`: Current execution phase ("clarification", "research", "complete")

## Monitoring

Enable LangChain tracing by setting:
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=research-agent
```

View traces at: https://smith.langchain.com

## API Endpoints (When Deployed)

- `/invoke`: Synchronous execution
- `/stream`: Streaming execution
- `/invoke_batch`: Batch processing

## Customization

### Modify Research Scope

Edit the system prompts in `research_agent.py`:
- `ClarificationAgent`: Adjust clarification strategy
- `ResearchAgent`: Modify research approach

### Add Additional Tools

Add tools to the ReAct agent in `ResearchAgent.__init__()`:
```python
self.tools = [self.search_tool, your_custom_tool]
```

## Troubleshooting

1. **API Key Issues**: Ensure all required API keys are set in `.env`
2. **Deployment Fails**: Check Docker is installed and running
3. **Search Not Working**: Verify Tavily API key and quota

## License

MIT