#!/usr/bin/env python3

import os
from main_graph import ResearchOrchestrator
from dotenv import load_dotenv

load_dotenv()


def main():
    print("LangGraph Deep Research Agent Example")
    print("=" * 40)
    
    if not all([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("TAVILY_API_KEY")
    ]):
        print("ERROR: Please set OPENAI_API_KEY and TAVILY_API_KEY in your .env file")
        print("Copy .env.example to .env and add your API keys")
        return
    
    print("\n1. Interactive Mode (recommended)")
    print("2. Batch Mode Example")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    orchestrator = ResearchOrchestrator()
    
    if choice == "1":
        print("\nStarting interactive mode...")
        print("You'll have a conversation with the agent to clarify your research needs.")
        print("Type 'quit' or 'exit' anytime to stop.\n")
        orchestrator.run_interactive()
        
    elif choice == "2":
        print("\nRunning batch mode example...")
        print("Topic: 'Future of Quantum Computing'\n")
        
        report = orchestrator.run_batch(
            topic="Future of Quantum Computing",
            objectives=[
                "Understand current quantum computing capabilities",
                "Identify key challenges and limitations",
                "Analyze potential breakthrough applications",
                "Assess timeline for practical adoption"
            ],
            questions=[
                "What are the current state-of-the-art quantum computers?",
                "What are the main technical barriers to quantum computing?",
                "Which industries will benefit most from quantum computing?",
                "When might quantum computers become commercially viable?"
            ]
        )
        
        print("\n" + "=" * 50)
        print("RESEARCH REPORT")
        print("=" * 50)
        print(report)
        print("=" * 50)
        
        with open("example_report.md", "w") as f:
            f.write(report)
        print("\nReport saved to 'example_report.md'")
        
    elif choice == "3":
        print("Goodbye!")
        
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()