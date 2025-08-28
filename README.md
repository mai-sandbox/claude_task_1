# LangGraph Deep Research Agent

A sophisticated research agent built with LangGraph that conducts comprehensive research through two phases:
1. **Clarification Phase**: Interactive dialogue to understand research scope and requirements
2. **Research Phase**: Deep research using ReAct pattern with Tavily search integration

## Features

- **Interactive Clarification**: Conversational interface to refine research requirements
- **ReAct Research Agent**: Strategic search and analysis using Tavily API
- **Comprehensive Reports**: Structured markdown reports with sources
- **LangGraph Deployment**: Ready for LangGraph Cloud deployment
- **Memory Persistence**: Conversation history and state management

## Architecture

```
┌─────────────────┐    ┌────────────────┐    ┌─────────────────┐
│  Clarification  │────│  Check Ready   │────│   Research      │
│     Agent       │    │   to Proceed   │    │     Agent       │
└─────────────────┘    └────────────────┘    └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │ Generate Report │
                                              └─────────────────┘
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # - OPENAI_API_KEY
   # - TAVILY_API_KEY
   # - LANGCHAIN_API_KEY (optional, for tracing)
   ```

3. **Get API Keys**:
   - OpenAI: https://platform.openai.com/api-keys
   - Tavily: https://tavily.com/
   - LangChain (optional): https://smith.langchain.com/

## Usage

### Interactive Mode

```bash
python main_graph.py
```

This starts an interactive session where the agent will:
1. Ask clarifying questions about your research topic
2. Build a comprehensive research brief
3. Conduct deep research using multiple searches
4. Generate and save a detailed report

### Programmatic Usage

```python
from main_graph import ResearchOrchestrator

orchestrator = ResearchOrchestrator()

# Batch mode with predefined parameters
report = orchestrator.run_batch(
    topic="Impact of AI on remote work",
    objectives=[
        "Understand current AI tools for remote work",
        "Analyze productivity impacts",
        "Identify future trends"
    ],
    questions=[
        "What AI tools are most popular for remote work?",
        "How has AI improved remote work productivity?",
        "What challenges remain?"
    ]
)

print(report)
```

## LangGraph Cloud Deployment

1. **Install LangGraph CLI**:
   ```bash
   pip install langgraph-cli
   ```

2. **Deploy to LangGraph Cloud**:
   ```bash
   langgraph deploy
   ```

The `langgraph.json` file configures the deployment with the main graph at `./main_graph.py:create_app`.

## Project Structure

```
├── agents/
│   ├── __init__.py
│   ├── clarification_agent.py    # Interactive clarification logic
│   └── react_agent.py           # ReAct research agent with Tavily
├── main_graph.py                # Main orchestration graph
├── langgraph.json              # LangGraph deployment config
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## How It Works

### Phase 1: Clarification
- Agent asks targeted questions to understand research scope
- Builds a structured `ResearchBrief` with:
  - Main topic
  - Specific objectives
  - Scope and boundaries
  - Key questions to answer
  - Constraints and limitations

### Phase 2: Research
- ReAct agent uses Tavily search strategically
- Conducts multiple searches based on research brief
- Gathers information from diverse sources
- Synthesizes findings into comprehensive report

### Output
- Detailed markdown report with:
  - Executive summary
  - Objective-based sections
  - Answered key questions
  - Sources and citations
  - Conclusions and recommendations

## Customization

### Adding New Tools
Add tools to `ReactResearchAgent.tools` in `agents/react_agent.py`:

```python
from langchain_community.tools import WikipediaQueryRun

self.tools = [
    self.tavily_tool,
    WikipediaQueryRun(),
    # Add more tools...
]
```

### Customizing LLM
Configure different models in the orchestrator:

```python
llm = ChatOpenAI(model="gpt-4", temperature=0.1)
orchestrator = ResearchOrchestrator(llm=llm)
```

### Extending Clarification Logic
Modify `ClarificationAgent` to add custom clarification flows or validation rules.

## Example Output

The agent generates comprehensive reports like:

```markdown
# Research Report: Impact of AI on Remote Work

## Executive Summary
This report examines the transformative impact of artificial intelligence on remote work practices...

## Objective 1: Understanding Current AI Tools
Based on research findings, the most prevalent AI tools for remote work include...

## Key Questions Answered
### What AI tools are most popular for remote work?
Research indicates that productivity suites like...

## Conclusions and Recommendations
1. Organizations should prioritize AI tool integration...
2. Training programs for remote workers should...
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.