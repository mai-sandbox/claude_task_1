# Deep Research Agent - Deployment Guide

This LangGraph agent provides an interactive research workflow with terminal-based scope clarification and automated ReAct-based research using Tavily search.

## Architecture

1. **Clarification Agent**: Interactive terminal session to refine research scope
2. **ReAct Agent**: Autonomous research agent using Tavily search tools
3. **State Management**: Persistent state across workflow phases
4. **LangGraph Orchestration**: Seamless transitions between agents

## Prerequisites

- Python 3.11+
- API Keys:
  - Anthropic API key (Claude)
  - Tavily API key (search)

## Local Setup

1. **Install dependencies**:
   ```bash
   pip install poetry
   poetry install
   ```

2. **Set environment variables**:
   ```bash
   cp .env .env.local
   # Edit .env.local with your API keys
   ```

3. **Run locally**:
   ```bash
   python run_agent.py
   ```

## LangGraph Platform Deployment

### Option 1: Cloud SaaS

1. **Prepare repository**:
   ```bash
   git add .
   git commit -m "Initial research agent implementation"
   git push origin main
   ```

2. **Deploy to LangGraph Platform**:
   - Go to [LangGraph Platform](https://langgraph.com)
   - Connect your GitHub repository
   - Set environment variables in platform UI
   - Deploy

### Option 2: Self-Hosted

1. **Using Docker**:
   ```bash
   docker build -t research-agent .
   docker run -p 8000:8000 --env-file .env research-agent
   ```

2. **Using LangGraph CLI**:
   ```bash
   pip install langgraph-cli
   langgraph up
   ```

## Configuration

The agent is configured through:
- `langgraph.json`: Main configuration
- `pyproject.toml`: Dependencies
- `.env`: Environment variables

## API Usage

Once deployed, interact with the agent via:

```python
from langgraph_sdk import get_client

client = get_client(url="YOUR_DEPLOYMENT_URL")

# Start research session
thread = client.threads.create()

# Begin research process
run = client.runs.create(
    thread["thread_id"],
    assistant_id="research_agent"
)
```

## Features

- **Interactive Scope Clarification**: Terminal-based conversation to refine research goals
- **Autonomous Research**: ReAct agent with web search capabilities
- **Comprehensive Reporting**: Detailed final reports with structured findings
- **Persistent State**: Memory across agent transitions
- **Platform Ready**: Deployable on LangGraph Cloud or self-hosted

## Troubleshooting

- Ensure all API keys are properly configured
- Check Python version compatibility (3.11+)
- Verify network connectivity for Tavily searches
- Review LangGraph platform logs for deployment issues