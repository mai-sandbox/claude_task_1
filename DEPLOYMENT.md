# LangGraph Research Agent - Deployment Guide

## Overview

This is a sophisticated research agent built with LangGraph that conducts deep research through a two-phase process:

1. **Clarification Phase**: Interactive conversation with the user to understand research requirements
2. **Research Phase**: ReAct agent with Tavily search tool that conducts comprehensive research

## Architecture

```
User Input
    ↓
Clarification Agent (Interactive Q&A)
    ↓
Research Brief Generation
    ↓
ReAct Research Agent (with Tavily Search)
    ↓
Detailed Research Report
```

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

Then add your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key for web search
- `LANGSMITH_API_KEY`: (Optional) For tracing and monitoring

### 3. Test Locally

Run the test script to verify everything works:

```bash
python test_agent.py
```

Or run the agent interactively:

```bash
python research_agent.py
```

## Deployment to LangGraph Platform

### Prerequisites

1. Install LangGraph CLI:
```bash
pip install langgraph-cli
```

2. Have a LangGraph Cloud account and API key

### Deployment Steps

1. **Build the agent**:
```bash
langgraph build
```

2. **Test locally with LangGraph server**:
```bash
langgraph dev
```
This will start a local server at http://localhost:8000

3. **Deploy to LangGraph Cloud**:
```bash
langgraph deploy
```

4. **Test the deployment**:
```bash
langgraph test
```

### Using the Deployed Agent

Once deployed, you can interact with the agent through:

1. **LangGraph Studio**: Visual interface for testing and debugging
2. **API Endpoint**: Direct API calls to your deployed agent
3. **LangServe Integration**: Integrate with existing LangChain applications

#### API Example

```python
import requests

# Deployed agent endpoint
url = "https://your-deployment.langgraph.app/invoke"

# Start conversation
response = requests.post(
    url,
    json={
        "input": {
            "messages": [
                {"role": "human", "content": "I want to research AI safety"}
            ]
        }
    },
    headers={"X-API-Key": "your-api-key"}
)
```

## Configuration

### Modifying the Agent

- **Change LLM Model**: Edit `ChatOpenAI(model="gpt-4o")` in `research_agent.py`
- **Adjust Search Parameters**: Modify `TavilySearchResults` settings
- **Customize Prompts**: Update system prompts in `ClarificationAgent` and `ResearchAgent`

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| OPENAI_API_KEY | OpenAI API key for LLM | Yes |
| TAVILY_API_KEY | Tavily API key for search | Yes |
| LANGSMITH_API_KEY | LangSmith for monitoring | No |
| LANGCHAIN_TRACING_V2 | Enable tracing | No |
| LANGCHAIN_PROJECT | Project name for tracing | No |

## Features

### Clarification Agent
- Interactive conversation to understand research scope
- Generates structured research brief
- Ensures complete understanding before research begins

### Research Agent (ReAct)
- Uses Tavily for advanced web search
- Systematic information gathering
- Synthesizes findings into comprehensive report
- Addresses all key questions from the brief

### State Management
- Maintains conversation history
- Tracks research progress
- Preserves research brief throughout process

## Monitoring and Debugging

### LangSmith Integration

When `LANGSMITH_API_KEY` is configured, you can:
- View execution traces
- Monitor token usage
- Debug agent decisions
- Analyze performance

### Local Debugging

Use the test script with verbose output:

```bash
python test_agent.py --verbose
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all required API keys are set in `.env`
2. **Import Errors**: Run `pip install -r requirements.txt`
3. **Deployment Fails**: Check `langgraph.json` configuration
4. **Search Not Working**: Verify Tavily API key and quota

### Support

For issues or questions:
1. Check LangGraph documentation
2. Review agent logs in LangSmith
3. Test locally before deploying

## Security Notes

- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- Rotate API keys regularly
- Monitor usage through provider dashboards