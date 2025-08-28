#!/usr/bin/env python3
"""
Test script for the Deep Research Agent
Validates graph structure and basic functionality without requiring API keys
"""

import sys
import os
sys.path.append('.')

# Mock the API dependencies for testing
class MockChatAnthropic:
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'mock-model')
        self.temperature = kwargs.get('temperature', 0.1)

class MockTavilySearch:
    def __init__(self, **kwargs):
        self.max_results = kwargs.get('max_results', 5)
    
    def invoke(self, query):
        return {"results": [{"title": "Mock Result", "content": "Mock content for testing"}]}

class MockReactAgent:
    def __init__(self, **kwargs):
        pass
    
    def invoke(self, inputs):
        return {
            "messages": [
                type('MockMessage', (), {
                    'content': 'Mock research report: This is a comprehensive analysis of the requested topic based on multiple sources and current information.'
                })()
            ]
        }

# Apply mocks
import agent
agent.ChatAnthropic = MockChatAnthropic
agent.TavilySearch = MockTavilySearch
agent.create_react_agent = lambda **kwargs: MockReactAgent(**kwargs)

def test_graph_structure():
    """Test that the graph structure is valid"""
    print("🧪 Testing graph structure...")
    
    try:
        # Import and create the workflow
        from agent import create_research_workflow
        
        workflow = create_research_workflow()
        
        # Check that the graph was created
        assert workflow is not None, "Workflow should not be None"
        
        # Get graph structure
        graph_dict = workflow.get_graph().to_json()
        
        # Verify nodes exist
        expected_nodes = {'clarify_scope', 'conduct_research'}
        actual_nodes = set(node['id'] for node in graph_dict['nodes'] if node['id'] not in ['__start__', '__end__'])
        
        assert expected_nodes.issubset(actual_nodes), f"Missing nodes. Expected: {expected_nodes}, Got: {actual_nodes}"
        
        print("✅ Graph structure is valid")
        print(f"   Nodes: {list(actual_nodes)}")
        print(f"   Edges: {len(graph_dict['edges'])} connections")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph structure test failed: {e}")
        return False

def test_state_model():
    """Test that the state model is properly defined"""
    print("\n🧪 Testing state model...")
    
    try:
        from agent import ResearchState
        
        # Create a sample state
        state = ResearchState(
            research_topic="Test Topic",
            research_scope="Test Scope",
            target_audience="Test Audience"
        )
        
        # Verify required fields exist
        assert hasattr(state, 'research_topic'), "State should have research_topic field"
        assert hasattr(state, 'research_brief'), "State should have research_brief field"
        assert hasattr(state, 'final_report'), "State should have final_report field"
        assert hasattr(state, 'user_approved'), "State should have user_approved field"
        
        print("✅ State model is valid")
        print(f"   Fields: {list(state.model_fields.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ State model test failed: {e}")
        return False

def test_node_functions():
    """Test that node functions are properly defined"""
    print("\n🧪 Testing node functions...")
    
    try:
        from agent import clarify_research_scope, conduct_research, should_proceed_with_research, ResearchState
        
        # Test function signatures
        import inspect
        
        # Check clarify_research_scope
        sig = inspect.signature(clarify_research_scope)
        assert 'state' in sig.parameters, "clarify_research_scope should accept state parameter"
        
        # Check conduct_research  
        sig = inspect.signature(conduct_research)
        assert 'state' in sig.parameters, "conduct_research should accept state parameter"
        
        # Check conditional function
        sig = inspect.signature(should_proceed_with_research)
        assert 'state' in sig.parameters, "should_proceed_with_research should accept state parameter"
        
        # Test conditional logic
        approved_state = ResearchState(user_approved=True)
        rejected_state = ResearchState(user_approved=False)
        
        assert should_proceed_with_research(approved_state) == "conduct_research", "Should proceed when approved"
        assert should_proceed_with_research(rejected_state) == "__end__", "Should end when not approved"
        
        print("✅ Node functions are valid")
        print("   ✓ clarify_research_scope function")
        print("   ✓ conduct_research function") 
        print("   ✓ should_proceed_with_research function")
        print("   ✓ Conditional logic working")
        
        return True
        
    except Exception as e:
        print(f"❌ Node functions test failed: {e}")
        return False

def test_deployment_config():
    """Test that deployment configuration is valid"""
    print("\n🧪 Testing deployment configuration...")
    
    try:
        import json
        
        # Check langgraph.json
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
        
        assert 'dependencies' in config, "langgraph.json should have dependencies"
        assert 'graphs' in config, "langgraph.json should have graphs"
        assert 'research_agent' in config['graphs'], "Should define research_agent graph"
        assert config['graphs']['research_agent'] == './agent.py:app', "Should point to correct app export"
        
        # Check requirements.txt exists
        assert os.path.exists('requirements.txt'), "requirements.txt should exist"
        
        # Check app export
        from agent import app
        assert app is not None, "app should be exported for deployment"
        
        print("✅ Deployment configuration is valid")
        print("   ✓ langgraph.json structure")
        print("   ✓ requirements.txt exists")
        print("   ✓ app exported correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Deep Research Agent")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        test_graph_structure,
        test_state_model,
        test_node_functions,
        test_deployment_config
    ]
    
    for test in tests:
        if not test():
            all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! The Deep Research Agent is ready for deployment.")
        print("\nNext steps:")
        print("1. Set up your API keys in .env file")
        print("2. Test locally with: python agent.py")
        print("3. Deploy with: langgraph dev")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()