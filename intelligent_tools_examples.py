#!/usr/bin/env python3
"""
Intelligent Tools Usage Examples - Demonstrates how to use the LLM-enhanced tool system.
"""

import os
from intelligent_tool_base import IntelligentToolConfig
from intelligent_tools_manager import IntelligentToolsManager

def create_config():
    """Create configuration for intelligent tools."""
    return IntelligentToolConfig(
        openai_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4o-mini",
        temperature=0.1,
        enable_parameter_inference=True,
        enable_context_awareness=True,
        enable_adaptive_responses=True
    )

def example_github_operations():
    """Example of intelligent GitHub operations."""
    print("\n=== GitHub Intelligent Operations Examples ===")
    
    config = create_config()
    manager = IntelligentToolsManager(config)
    
    # Initialize GitHub client
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        github_client = manager.initialize_github(github_token, "owner", "repo")
        
        # Example 1: Natural language issue retrieval
        result1 = manager.intelligent_execute(
            "Show me details about issue number 123",
            preferred_tool="github"
        )
        print(f"Issue details: {result1.get('user_message', 'No message')}")
        
        # Example 2: Smart issue creation
        result2 = manager.intelligent_execute(
            "Create a high priority bug report for login not working on mobile devices",
            preferred_tool="github"
        )
        print(f"Issue creation: {result2.get('user_message', 'No message')}")
        
        # Example 3: Intelligent search
        result3 = manager.intelligent_execute(
            "Find all open bugs related to authentication",
            preferred_tool="github"
        )
        print(f"Search results: {result3.get('user_message', 'No message')}")
    
    print("GitHub examples completed!")

def example_jira_operations():
    """Example of intelligent Jira operations."""
    print("\n=== Jira Intelligent Operations Examples ===")
    
    config = create_config()
    manager = IntelligentToolsManager(config)
    
    # Initialize Jira client
    jira_url = os.getenv('JIRA_BASE_URL')
    jira_email = os.getenv('JIRA_EMAIL')
    jira_token = os.getenv('JIRA_TOKEN')
    
    if all([jira_url, jira_email, jira_token]):
        jira_client = manager.initialize_jira(jira_url, jira_email, jira_token, "PROJECT")
        
        # Example 1: Natural language issue retrieval
        result1 = manager.intelligent_execute(
            "Get details for issue PROJECT-123",
            preferred_tool="jira"
        )
        print(f"Issue details: {result1.get('user_message', 'No message')}")
        
        # Example 2: Status transition with natural language
        result2 = manager.intelligent_execute(
            "Move PROJECT-123 to done status",
            preferred_tool="jira"
        )
        print(f"Status update: {result2.get('user_message', 'No message')}")
        
        # Example 3: Smart issue creation
        result3 = manager.intelligent_execute(
            "Create an urgent task to fix the database connection issue",
            preferred_tool="jira"
        )
        print(f"Issue creation: {result3.get('user_message', 'No message')}")
        
        # Example 4: Natural language search
        result4 = manager.intelligent_execute(
            "Find all open issues assigned to me",
            preferred_tool="jira"
        )
        print(f"Search results: {result4.get('user_message', 'No message')}")
    
    print("Jira examples completed!")

def example_notion_operations():
    """Example of intelligent Notion operations."""
    print("\n=== Notion Intelligent Operations Examples ===")
    
    config = create_config()
    manager = IntelligentToolsManager(config)
    
    # Initialize Notion client
    notion_token = os.getenv('NOTION_TOKEN')
    database_id = os.getenv('NOTION_DATABASE_ID')
    
    if notion_token:
        notion_client = manager.initialize_notion(notion_token, database_id)
        
        # Example 1: Smart page creation
        result1 = manager.intelligent_execute(
            "Create a new project page called 'Mobile App Redesign' with status as In Progress",
            preferred_tool="notion"
        )
        print(f"Page creation: {result1.get('user_message', 'No message')}")
        
        # Example 2: Intelligent text appending
        result2 = manager.intelligent_execute(
            "Add text 'Updated requirements based on user feedback' to the Mobile App Redesign page",
            preferred_tool="notion"
        )
        print(f"Text appending: {result2.get('user_message', 'No message')}")
        
        # Example 3: Natural language status update
        result3 = manager.intelligent_execute(
            "Change the status of Mobile App Redesign page to completed",
            preferred_tool="notion"
        )
        print(f"Status update: {result3.get('user_message', 'No message')}")
        
        # Example 4: Smart database queries
        result4 = manager.intelligent_execute(
            "Show me all pages with status In Progress",
            preferred_tool="notion"
        )
        print(f"Database query: {result4.get('user_message', 'No message')}")
        
        # Example 5: Intelligent search
        result5 = manager.intelligent_execute(
            "Search for any pages related to mobile development",
            preferred_tool="notion"
        )
        print(f"Search results: {result5.get('user_message', 'No message')}")
    
    print("Notion examples completed!")

def example_cross_tool_intelligence():
    """Example of cross-tool intelligent operations."""
    print("\n=== Cross-Tool Intelligence Examples ===")
    
    config = create_config()
    manager = IntelligentToolsManager(config)
    
    # Initialize all tools
    github_token = os.getenv('GITHUB_TOKEN')
    jira_url = os.getenv('JIRA_BASE_URL')
    jira_email = os.getenv('JIRA_EMAIL')
    jira_token = os.getenv('JIRA_TOKEN')
    notion_token = os.getenv('NOTION_TOKEN')
    
    if github_token:
        manager.initialize_github(github_token, "owner", "repo")
    
    if all([jira_url, jira_email, jira_token]):
        manager.initialize_jira(jira_url, jira_email, jira_token, "PROJECT")
    
    if notion_token:
        manager.initialize_notion(notion_token)
    
    # Example 1: Let the system choose the best tool
    result1 = manager.intelligent_execute(
        "Create a new issue about the login bug in the mobile app"
    )
    print(f"Auto-tool selection: {result1.get('user_message', 'No message')}")
    print(f"Tools considered: {result1.get('tools_considered', [])}")
    
    # Example 2: Cross-tool suggestions
    result2 = manager.intelligent_execute(
        "Create a GitHub issue for fixing the payment gateway"
    )
    if 'cross_tool_suggestions' in result2:
        print(f"Cross-tool suggestions: {result2['cross_tool_suggestions']}")
    
    print("Cross-tool intelligence examples completed!")

def example_system_statistics():
    """Example of getting system intelligence statistics."""
    print("\n=== System Intelligence Statistics ===")
    
    config = create_config()
    manager = IntelligentToolsManager(config)
    
    # Initialize some tools and perform operations
    if os.getenv('GITHUB_TOKEN'):
        manager.initialize_github(os.getenv('GITHUB_TOKEN'))
        
        # Perform some intelligent operations to generate stats
        manager.intelligent_execute("Show me issue #1", preferred_tool="github")
        manager.intelligent_execute("Create a bug report", preferred_tool="github")
    
    # Get comprehensive system stats
    stats = manager.get_system_intelligence_stats()
    
    print("System Statistics:")
    print(f"- Initialized tools: {stats['initialized_tools']}")
    print(f"- Total operations: {stats['system_metrics']['total_intelligent_operations']}")
    print(f"- Average success rate: {stats['system_metrics']['average_success_rate']:.2%}")
    print(f"- Active tools: {stats['system_metrics']['tools_active']}")
    
    # Tool-specific stats
    for tool_name, tool_stats in stats['tool_stats'].items():
        print(f"\n{tool_name.title()} Tool Stats:")
        print(f"  - Operations: {tool_stats.get('total_intelligent_operations', 0)}")
        print(f"  - Success rate: {tool_stats.get('average_success_rate', 0):.2%}")

def example_performance_optimization():
    """Example of system performance optimization."""
    print("\n=== Performance Optimization ===")
    
    config = create_config()
    manager = IntelligentToolsManager(config)
    
    # Initialize tools and perform operations
    if os.getenv('GITHUB_TOKEN'):
        manager.initialize_github(os.getenv('GITHUB_TOKEN'))
    
    # Simulate some operations over time
    import time
    for i in range(5):
        manager.intelligent_execute(f"Test operation {i}", preferred_tool="github")
        time.sleep(0.1)  # Simulate time passing
    
    # Run optimization
    optimization_result = manager.optimize_system_performance()
    
    print("Optimization Results:")
    print(f"- Cache cleared: {optimization_result['cache_cleared']}")
    print(f"- Context pruned: {optimization_result['context_pruned']}")
    
    if optimization_result['recommendations']:
        print("Recommendations:")
        for rec in optimization_result['recommendations']:
            print(f"  - {rec}")

def main():
    """Run all intelligent tools examples."""
    print("🚀 Intelligent Tools System Examples")
    print("====================================")
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️  Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables to run the examples.")
        return
    
    try:
        # Run examples
        example_github_operations()
        example_jira_operations()
        example_notion_operations()
        example_cross_tool_intelligence()
        example_system_statistics()
        example_performance_optimization()
        
        print("\n✅ All examples completed successfully!")
        print("\n📊 Key Benefits of the Intelligent Tool System:")
        print("- Natural language understanding for all operations")
        print("- Smart parameter inference and context awareness")
        print("- Cross-tool intelligence and workflow suggestions")
        print("- Adaptive responses based on user intent")
        print("- Learning from interaction patterns")
        print("- Performance monitoring and optimization")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()