# LangGraph Deep Research Agent

A sophisticated research agent built with LangGraph that conducts thorough research through a two-phase process: scope clarification and systematic research execution.

## Features

- **Interactive Clarification Phase**: Engages with users to understand and refine research requirements
- **ReAct Research Agent**: Systematically conducts research using Tavily search API
- **Comprehensive Reports**: Generates detailed, well-structured research reports with citations
- **LangGraph Platform Ready**: Fully deployable on LangGraph platform with included configuration

## Architecture

The agent consists of two main components:

1. **Clarification Agent**: Conducts interactive dialogue to gather research requirements and create a clear research brief
2. **Research Agent**: ReAct-based agent that uses Tavily search to gather information and synthesize comprehensive reports

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- Tavily API key
- (Optional) LangChain API key for tracing

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claude_task_1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Interactive Terminal Mode

Run the agent in interactive mode:

```bash
python app.py
```

The agent will:
1. Start by asking clarifying questions about your research topic
2. Build a comprehensive research brief based on your responses
3. Execute systematic research using web search
4. Generate a detailed report with citations

### Programmatic Usage

```python
from graph import create_research_graph
from langchain_core.messages import HumanMessage

# Create the agent
app = create_research_graph()

# Configure session
config = {"configurable": {"thread_id": "session_1"}}

# Start interaction
state = app.invoke(
    {"messages": [HumanMessage(content="I want to research AI safety")]},
    config
)
```

### Example Usage

Run the example demonstrations:

```bash
python example_usage.py
```

## Deployment on LangGraph Platform

### Local Deployment with LangGraph Studio

1. Install LangGraph Studio
2. Open the project directory
3. The agent will be automatically detected via `langgraph.json`

### Cloud Deployment

1. Build the Docker image:
```bash
docker build -t research-agent .
```

2. Deploy to LangGraph Cloud:
```bash
langgraph deploy --name research-agent
```

### Configuration Files

- `langgraph.json`: LangGraph platform configuration
- `Dockerfile`: Container configuration for deployment
- `.env`: Environment variables (API keys)

## Project Structure

```
claude_task_1/
├── app.py                 # Main application entry point
├── graph.py              # Graph orchestration and workflow
├── state.py              # State definitions
├── clarification_agent.py # Clarification phase agent
├── research_agent.py     # ReAct research agent
├── tools.py              # Tavily search tool integration
├── example_usage.py      # Usage examples
├── requirements.txt      # Python dependencies
├── langgraph.json       # LangGraph deployment config
├── Dockerfile           # Container configuration
├── .env.example         # Environment variables template
└── README.md           # This file
```

## API Keys

You'll need the following API keys:

1. **OpenAI API Key**: For LLM interactions
   - Get it from: https://platform.openai.com/api-keys

2. **Tavily API Key**: For web search functionality
   - Get it from: https://tavily.com/

3. **LangChain API Key** (Optional): For tracing and monitoring
   - Get it from: https://smith.langchain.com/

## Features in Detail

### Clarification Phase
- Interactive dialogue to understand research scope
- Asks about topics, constraints, depth, and format preferences
- Generates a structured research brief
- Confirms brief with user before proceeding

### Research Phase
- Breaks down research brief into key questions
- Performs systematic web searches
- Gathers diverse perspectives and sources
- Verifies facts with multiple sources
- Synthesizes findings into comprehensive report

### Report Generation
- Well-structured with clear sections
- Based on factual information from searches
- Includes citations and source URLs
- Provides balanced and objective analysis
- Addresses all aspects from research brief

## Troubleshooting

### Missing API Keys
If you see "Missing required environment variables", ensure you've:
1. Copied `.env.example` to `.env`
2. Added your API keys to `.env`
3. Restarted the application

### Search Errors
If Tavily searches fail:
1. Verify your Tavily API key is valid
2. Check your Tavily account has available credits
3. Ensure you have internet connectivity

### Memory Issues
For long research sessions:
1. The agent limits iterations to prevent infinite loops
2. Search results are capped to prevent token overflow
3. Consider breaking complex research into multiple sessions

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- Include appropriate error handling
- Update documentation as needed
- Test with various research scenarios

## License

[Your License Here]