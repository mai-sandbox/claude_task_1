# Quick Start Guide - LangGraph Research Agent

## 🚀 Get Started in 5 Minutes

### 1. Clone and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Add Your API Keys
Edit `.env` file:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Get keys from:
- OpenAI: https://platform.openai.com/api-keys
- Tavily: https://tavily.com/

### 3. Run the Agent

**Interactive Mode:**
```bash
python research_agent.py
```

**Test Mode:**
```bash
python test_agent.py
```

## 💬 Example Conversation

```
You: I want to research the impact of AI on healthcare

Agent: I'll help you research the impact of AI on healthcare. To ensure I cover exactly what you need, could you clarify:
1. Are you interested in specific areas (diagnosis, treatment, administration)?
2. Any particular timeframe (recent developments, historical overview)?
3. Geographic focus (global, specific countries)?

You: Focus on diagnostic AI in radiology, particularly recent breakthroughs in 2024. Yes, proceed.

Agent: Research brief created. Starting research phase...
[Agent conducts research and returns comprehensive report]
```

## 🚢 Deploy to LangGraph Cloud

```bash
# Install CLI
pip install langgraph-cli

# Test locally
langgraph dev

# Deploy
langgraph deploy
```

## 📁 Project Structure

```
├── research_agent.py           # Main agent implementation
├── research_agent_deployment.py # Deployment wrapper
├── langgraph.json              # LangGraph configuration
├── test_agent.py               # Test suite
├── requirements.txt            # Dependencies
└── .env                       # API keys (create from .env.example)
```

## 🔧 Customize

- **Change AI Model**: Edit line 156 in `research_agent.py`
- **Adjust Search Depth**: Modify `TavilySearchResults` parameters in line 102
- **Customize Prompts**: Update system prompts in agent classes

## 📊 Monitor Performance

With LangSmith configured, view:
- Execution traces
- Token usage
- Decision paths
- Performance metrics

Visit: https://smith.langchain.com/

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| OpenAI API Error | Check OPENAI_API_KEY in .env |
| Tavily Search Fails | Verify TAVILY_API_KEY |
| Import Errors | Run `pip install -r requirements.txt` |
| Deployment Issues | Check langgraph.json configuration |

## 📚 Learn More

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Tavily API](https://docs.tavily.com/)
- [Deployment Guide](./DEPLOYMENT.md)