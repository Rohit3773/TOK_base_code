#!/usr/bin/env python3
"""
Integration Guide for Intelligent Tools System
This shows how to integrate the new intelligent tools into existing applications.
"""

import os
import logging
from typing import Dict, Any, Optional
from intelligent_tool_base import IntelligentToolConfig
from intelligent_tools_manager import IntelligentToolsManager

logger = logging.getLogger(__name__)

class IntelligentToolsIntegration:
    """
    Integration wrapper that provides easy integration with existing applications.
    
    This class demonstrates how to integrate the intelligent tools system
    into applications like the existing Streamlit app without major changes.
    """
    
    def __init__(self):
        """Initialize the integration with intelligent tools."""
        self.manager = None
        self.config = None
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the intelligent tools system."""
        try:
            # Create configuration
            self.config = IntelligentToolConfig(
                openai_key=os.getenv('OPENAI_API_KEY'),
                model="gpt-4o-mini",
                temperature=0.1,
                enable_parameter_inference=True,
                enable_context_awareness=True,
                enable_adaptive_responses=True,
                cache_responses=True
            )
            
            # Create manager
            self.manager = IntelligentToolsManager(self.config)
            
            # Auto-initialize available tools
            self._auto_initialize_tools()
            
            logger.info("Intelligent tools system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize intelligent tools: {e}")
            self.manager = None
    
    def _auto_initialize_tools(self):
        """Automatically initialize tools based on available credentials."""
        
        # Initialize GitHub if token available
        github_token = os.getenv('GITHUB_TOKEN')
        if github_token:
            try:
                self.manager.initialize_github(github_token)
                logger.info("GitHub intelligent client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GitHub: {e}")
        
        # Initialize Jira if credentials available
        jira_url = os.getenv('JIRA_BASE_URL')
        jira_email = os.getenv('JIRA_EMAIL') 
        jira_token = os.getenv('JIRA_TOKEN')
        
        if all([jira_url, jira_email, jira_token]):
            try:
                project_key = os.getenv('JIRA_PROJECT_KEY', 'DEFAULT')
                self.manager.initialize_jira(jira_url, jira_email, jira_token, project_key)
                logger.info("Jira intelligent client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Jira: {e}")
        
        # Initialize Notion if token available
        notion_token = os.getenv('NOTION_TOKEN')
        if notion_token:
            try:
                database_id = os.getenv('NOTION_DATABASE_ID')
                self.manager.initialize_notion(notion_token, database_id)
                logger.info("Notion intelligent client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Notion: {e}")
    
    def execute_intelligent_action(self, user_input: str) -> Dict[str, Any]:
        """
        Execute an action using intelligent tools system.
        
        This method serves as a drop-in replacement for existing tool execution.
        It automatically determines the best tool and action based on user input.
        
        Args:
            user_input: Natural language input from user
            
        Returns:
            Enhanced result with intelligent interpretation
        """
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Intelligent tools system not available',
                'user_message': 'Falling back to basic tools'
            }
        
        try:
            # Let the intelligent system handle everything
            result = self.manager.intelligent_execute(user_input)
            
            # Add integration metadata
            result['processed_by_intelligent_system'] = True
            result['original_input'] = user_input
            
            return result
            
        except Exception as e:
            logger.error(f"Intelligent execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_message': f'Intelligent execution failed: {str(e)}',
                'fallback_needed': True
            }
    
    def execute_with_tool_preference(self, user_input: str, preferred_tool: str) -> Dict[str, Any]:
        """
        Execute with a preferred tool specified.
        
        Args:
            user_input: Natural language input
            preferred_tool: Tool to prefer ('github', 'jira', 'notion')
            
        Returns:
            Result from intelligent execution
        """
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Intelligent tools system not available'
            }
        
        return self.manager.intelligent_execute(
            user_input, 
            preferred_tool=preferred_tool
        )
    
    def execute_with_action_hints(self, user_input: str, action_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute with specific action hints to guide the intelligent system.
        
        Args:
            user_input: Natural language input
            action_hints: Hints about parameters or actions to use
            
        Returns:
            Result from intelligent execution
        """
        
        if not self.manager:
            return {
                'success': False,
                'error': 'Intelligent tools system not available'
            }
        
        return self.manager.intelligent_execute(
            user_input,
            action_hints=action_hints
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the intelligent tools system."""
        
        if not self.manager:
            return {
                'available': False,
                'error': 'System not initialized'
            }
        
        stats = self.manager.get_system_intelligence_stats()
        
        return {
            'available': True,
            'initialized_tools': stats['initialized_tools'],
            'system_metrics': stats['system_metrics'],
            'config': {
                'model': self.config.model,
                'temperature': self.config.temperature,
                'parameter_inference': self.config.enable_parameter_inference,
                'context_awareness': self.config.enable_context_awareness,
                'adaptive_responses': self.config.enable_adaptive_responses
            }
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance."""
        
        if not self.manager:
            return {'error': 'System not available'}
        
        return self.manager.optimize_system_performance()

def integrate_with_streamlit_example():
    """Example of how to integrate with existing Streamlit app."""
    
    print("\n=== Streamlit Integration Example ===")
    
    # Initialize integration
    integration = IntelligentToolsIntegration()
    
    # Check system status
    status = integration.get_system_status()
    print(f"System available: {status['available']}")
    
    if status['available']:
        print(f"Initialized tools: {status['initialized_tools']}")
        
        # Example user inputs that would come from Streamlit
        user_inputs = [
            "Show me issue #123 from GitHub",
            "Create a new Jira ticket for the login bug", 
            "Add text to the Testing Tools page in Notion",
            "Change status of PROJECT-456 to done",
            "Search for all open issues assigned to me"
        ]
        
        for user_input in user_inputs:
            print(f"\nProcessing: {user_input}")
            result = integration.execute_intelligent_action(user_input)
            
            if result.get('success'):
                print(f"✅ {result.get('user_message', 'Action completed')}")
                
                # Show cross-tool suggestions if available
                if 'cross_tool_suggestions' in result:
                    print(f"💡 Suggestions: {result['cross_tool_suggestions']}")
                    
            else:
                print(f"❌ {result.get('error', 'Unknown error')}")
    
    print("\nStreamlit integration example completed!")

def demonstrate_backward_compatibility():
    """Demonstrate how the system maintains backward compatibility."""
    
    print("\n=== Backward Compatibility Demo ===")
    
    integration = IntelligentToolsIntegration()
    
    # Example: existing code might call specific tools
    # But now it can benefit from intelligent enhancements
    
    # Old way: specific tool calls
    old_style_requests = [
        ("github", "get issue 123"),
        ("jira", "create task for database fix"),
        ("notion", "update page status to complete")
    ]
    
    for tool, request in old_style_requests:
        print(f"\nOld style: Tool={tool}, Request={request}")
        
        # New intelligent execution with tool preference
        result = integration.execute_with_tool_preference(request, tool)
        
        if result.get('success'):
            print(f"✅ Enhanced: {result.get('user_message')}")
        else:
            print(f"❌ Failed: {result.get('error')}")
    
    print("\nBackward compatibility demonstration completed!")

def demonstrate_advanced_features():
    """Demonstrate advanced intelligent features."""
    
    print("\n=== Advanced Features Demo ===")
    
    integration = IntelligentToolsIntegration()
    
    # Feature 1: Action hints for precise control
    print("\n1. Action Hints Example:")
    result = integration.execute_with_action_hints(
        "Create an issue",
        action_hints={
            "title": "Database Connection Timeout",
            "priority": "High",
            "labels": ["bug", "database"]
        }
    )
    print(f"Result: {result.get('user_message', 'No message')}")
    
    # Feature 2: System optimization
    print("\n2. System Optimization:")
    optimization = integration.optimize_performance()
    print(f"Optimization: {optimization}")
    
    # Feature 3: System statistics
    print("\n3. System Statistics:")
    status = integration.get_system_status()
    if 'system_metrics' in status:
        metrics = status['system_metrics']
        print(f"Total operations: {metrics.get('total_intelligent_operations', 0)}")
        print(f"Success rate: {metrics.get('average_success_rate', 0):.1%}")
        print(f"Active tools: {metrics.get('tools_active', 0)}")
    
    print("\nAdvanced features demonstration completed!")

def main():
    """Run integration examples."""
    print("🔧 Intelligent Tools Integration Guide")
    print("====================================")
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found. Please set this environment variable.")
        return
    
    try:
        # Run integration examples
        integrate_with_streamlit_example()
        demonstrate_backward_compatibility()
        demonstrate_advanced_features()
        
        print("\n✅ Integration examples completed!")
        
        print("\n📋 Integration Checklist:")
        print("1. ✅ Set OPENAI_API_KEY environment variable")
        print("2. ✅ Import IntelligentToolsIntegration class")
        print("3. ✅ Replace existing tool calls with execute_intelligent_action()")
        print("4. ✅ Handle enhanced responses with user_message field")
        print("5. ✅ Utilize cross-tool suggestions for better UX")
        print("6. ✅ Monitor system performance with get_system_status()")
        
        print("\n🚀 Benefits After Integration:")
        print("- Natural language understanding for all operations")
        print("- Automatic parameter inference and validation")
        print("- Smart tool selection based on user intent")
        print("- Enhanced error handling with suggestions")
        print("- Cross-tool workflow coordination")
        print("- Learning from user patterns for improvement")
        
    except Exception as e:
        print(f"❌ Integration examples failed: {e}")

if __name__ == "__main__":
    main()