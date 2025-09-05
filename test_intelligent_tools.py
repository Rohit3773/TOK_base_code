#!/usr/bin/env python3
"""
Test Suite for Intelligent Tools System
Tests all components of the LLM-enhanced tool architecture.
"""

import os
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligent_tool_base import IntelligentToolConfig, IntelligentToolBase
from intelligent_tools_manager import IntelligentToolsManager
from intelligent_github_client import IntelligentGitHubClient
from intelligent_jira_client import IntelligentJiraClient  
from intelligent_notion_client import IntelligentNotionClient

class MockOpenAIClient:
    """Mock OpenAI client for testing."""
    
    def __init__(self):
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock()
    
    def create_mock_response(self, content):
        """Create a mock response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = content
        return mock_response

class TestIntelligentToolConfig(unittest.TestCase):
    """Test IntelligentToolConfig class."""
    
    def test_config_creation(self):
        """Test creating configuration."""
        config = IntelligentToolConfig(
            openai_key="test-key",
            model="gpt-4o-mini",
            temperature=0.1
        )
        
        self.assertEqual(config.openai_key, "test-key")
        self.assertEqual(config.model, "gpt-4o-mini")
        self.assertEqual(config.temperature, 0.1)
        self.assertTrue(config.enable_parameter_inference)

class TestIntelligentToolBase(unittest.TestCase):
    """Test IntelligentToolBase abstract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntelligentToolConfig(openai_key="test-key")
        
        # Create a concrete implementation for testing
        class TestTool(IntelligentToolBase):
            def get_tool_name(self):
                return "TestTool"
            
            def get_available_actions(self):
                return [
                    {
                        "name": "test_action",
                        "description": "Test action",
                        "parameters": ["param1", "param2"]
                    }
                ]
            
            def execute_base_action(self, action, params):
                if action == "test_action":
                    return {"success": True, "data": f"Executed with {params}"}
                return {"success": False, "error": "Unknown action"}
        
        self.tool = TestTool(self.config)
    
    @patch('intelligent_tool_base.OpenAI')
    def test_tool_initialization(self, mock_openai):
        """Test tool initialization."""
        self.assertEqual(self.tool.get_tool_name(), "TestTool")
        self.assertEqual(len(self.tool.get_available_actions()), 1)
        mock_openai.assert_called_once_with(api_key="test-key")
    
    @patch('intelligent_tool_base.OpenAI')
    def test_base_action_execution(self, mock_openai):
        """Test base action execution."""
        result = self.tool.execute_base_action("test_action", {"param1": "value1"})
        
        self.assertTrue(result["success"])
        self.assertIn("Executed with", result["data"])
    
    @patch('intelligent_tool_base.OpenAI')
    def test_pattern_generation(self, mock_openai):
        """Test pattern key generation."""
        patterns = [
            ("Create a new issue", "create_pattern"),
            ("Update the status", "update_pattern"),
            ("Get issue details", "read_pattern"),
            ("Delete the project", "delete_pattern"),
            ("Random request", "general_pattern")
        ]
        
        for request, expected_pattern in patterns:
            pattern = self.tool._generate_pattern_key(request)
            self.assertEqual(pattern, expected_pattern)
    
    @patch('intelligent_tool_base.OpenAI')
    def test_json_extraction(self, mock_openai):
        """Test JSON extraction from LLM responses."""
        test_cases = [
            ('{"test": "value"}', {"test": "value"}),
            ('```json\n{"test": "value"}\n```', {"test": "value"}),
            ('Some text {"test": "value"} more text', {"test": "value"}),
            ('Invalid JSON', {})
        ]
        
        for input_text, expected in test_cases:
            result = self.tool._extract_json_from_response(input_text)
            self.assertEqual(result, expected)

class TestIntelligentGitHubClient(unittest.TestCase):
    """Test IntelligentGitHubClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntelligentToolConfig(openai_key="test-key")
        
        with patch('intelligent_github_client.GitHubClient'):
            self.github_client = IntelligentGitHubClient("test-token", self.config)
    
    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.github_client.get_tool_name(), "GitHub")
    
    def test_available_actions(self):
        """Test available actions."""
        actions = self.github_client.get_available_actions()
        action_names = [action["name"] for action in actions]
        
        expected_actions = ["get_issue", "create_issue", "search_issues", "create_pull_request", "list_issues"]
        for expected in expected_actions:
            self.assertIn(expected, action_names)
    
    def test_repo_extraction(self):
        """Test repository extraction from user requests."""
        test_cases = [
            ("Show issue from owner/repo", {"owner": "owner", "repo": "repo"}),
            ("github.com/user/project issue #123", {"owner": "user", "repo": "project"}),
            ("repository microsoft/vscode", {"owner": "microsoft", "repo": "vscode"}),
            ("Random text with no repo", None)
        ]
        
        for request, expected in test_cases:
            result = self.github_client._extract_repo_from_request(request)
            self.assertEqual(result, expected)
    
    def test_issue_number_extraction(self):
        """Test issue number extraction."""
        test_cases = [
            ("Show me issue #123", 123),
            ("Get issue 456", 456),  
            ("Issue number 789", 789),
            ("No issue number here", None)
        ]
        
        for request, expected in test_cases:
            result = self.github_client._extract_issue_number(request)
            self.assertEqual(result, expected)

class TestIntelligentJiraClient(unittest.TestCase):
    """Test IntelligentJiraClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntelligentToolConfig(openai_key="test-key")
        
        with patch('intelligent_jira_client.JiraClient'):
            self.jira_client = IntelligentJiraClient(
                "https://test.atlassian.net", 
                "test@example.com", 
                "test-token",
                self.config
            )
    
    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.jira_client.get_tool_name(), "Jira")
    
    def test_issue_key_extraction(self):
        """Test Jira issue key extraction."""
        test_cases = [
            ("Show PROJECT-123", "PROJECT-123"),
            ("Get details for abc-456", "ABC-456"),
            ("Update TEST_789", "TEST_789"),
            ("No issue key here", None)
        ]
        
        for request, expected in test_cases:
            result = self.jira_client._extract_issue_key(request)
            self.assertEqual(result, expected)
    
    def test_status_extraction(self):
        """Test status extraction from natural language."""
        test_cases = [
            ("Mark as done", "done"),
            ("Set to in progress", "in progress"),
            ("Change to todo", "todo"),
            ("Make it blocked", "blocked"),
            ("Random status", None)
        ]
        
        for request, expected in test_cases:
            result = self.jira_client._extract_target_status(request)
            self.assertEqual(result, expected)
    
    def test_jql_detection(self):
        """Test JQL detection."""
        test_cases = [
            ("project = TEST AND status = Open", True),
            ("assignee = currentUser()", True),
            ("Find all open bugs", False),
            ("Show me recent issues", False)
        ]
        
        for query, expected in test_cases:
            result = self.jira_client._looks_like_jql(query)
            self.assertEqual(result, expected)
    
    def test_issue_type_inference(self):
        """Test issue type inference."""
        test_cases = [
            ("Fix the login bug", "Bug"),
            ("Add new feature for users", "Task"),
            ("User story about registration", "Story"),
            ("Major epic for redesign", "Epic"),
            ("Generic request", "Task")  # Default
        ]
        
        for request, expected in test_cases:
            result = self.jira_client._infer_issue_type(request)
            self.assertEqual(result, expected)

class TestIntelligentNotionClient(unittest.TestCase):
    """Test IntelligentNotionClient."""
    
    def setUp(self):
        """Set up test fixtures.""" 
        self.config = IntelligentToolConfig(openai_key="test-key")
        
        with patch('intelligent_notion_client.NotionClient'):
            self.notion_client = IntelligentNotionClient("test-token", self.config)
    
    def test_tool_name(self):
        """Test tool name."""
        self.assertEqual(self.notion_client.get_tool_name(), "Notion")
    
    def test_page_name_extraction(self):
        """Test page name extraction."""
        test_cases = [
            ('Add text to page "Testing Tools"', "Testing Tools"),
            ("Update the Project Management page", "Project Management"),
            ('In "My Notes" page add text', "My Notes"),
            ("Random request", None)
        ]
        
        for request, expected in test_cases:
            result = self.notion_client._extract_page_name(request)
            self.assertEqual(result, expected)
    
    def test_status_extraction(self):
        """Test status extraction."""
        test_cases = [
            ("Set status to done", "Done"),
            ("Mark as in progress", "In Progress"),
            ("Change to todo", "To Do"),
            ("Make it blocked", "Blocked"),
            ("Set for review", "Review"),
            ("Random status", None)
        ]
        
        for request, expected in test_cases:
            result = self.notion_client._extract_status(request)
            self.assertEqual(result, expected)
    
    def test_title_extraction(self):
        """Test title extraction for new pages."""
        test_cases = [
            ('Create page "New Project"', "New Project"),
            ("New page called 'Task List'", "Task List"),
            ('Page titled "Meeting Notes"', "Meeting Notes"),
            ("Random request", None)
        ]
        
        for request, expected in test_cases:
            result = self.notion_client._extract_title_from_request(request)
            self.assertEqual(result, expected)

class TestIntelligentToolsManager(unittest.TestCase):
    """Test IntelligentToolsManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntelligentToolConfig(openai_key="test-key")
        self.manager = IntelligentToolsManager(self.config)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.config, self.config)
        self.assertEqual(len(self.manager.tools), 0)
        self.assertEqual(len(self.manager._initialized_tools), 0)
    
    @patch('intelligent_tools_manager.IntelligentGitHubClient')
    def test_github_initialization(self, mock_github):
        """Test GitHub client initialization."""
        mock_client = Mock()
        mock_github.return_value = mock_client
        
        result = self.manager.initialize_github("test-token", "owner", "repo")
        
        self.assertEqual(result, mock_client)
        self.assertIn("github", self.manager.tools)
        self.assertIn("github", self.manager._initialized_tools)
        mock_client.set_current_repo.assert_called_once_with("owner", "repo")
    
    @patch('intelligent_tools_manager.IntelligentJiraClient')
    def test_jira_initialization(self, mock_jira):
        """Test Jira client initialization."""
        mock_client = Mock()
        mock_jira.return_value = mock_client
        
        result = self.manager.initialize_jira("url", "email", "token", "PROJECT")
        
        self.assertEqual(result, mock_client)
        self.assertIn("jira", self.manager.tools)
        self.assertIn("jira", self.manager._initialized_tools)
        mock_client.set_current_project.assert_called_once_with("PROJECT")
    
    @patch('intelligent_tools_manager.IntelligentNotionClient')
    def test_notion_initialization(self, mock_notion):
        """Test Notion client initialization."""
        mock_client = Mock()
        mock_notion.return_value = mock_client
        
        result = self.manager.initialize_notion("token", "db-id")
        
        self.assertEqual(result, mock_client)
        self.assertIn("notion", self.manager.tools)
        self.assertIn("notion", self.manager._initialized_tools)
        mock_client.set_current_database.assert_called_once_with("db-id")
    
    def test_tool_selection(self):
        """Test tool selection logic."""
        self.manager._initialized_tools = {"github", "jira", "notion"}
        
        test_cases = [
            ("Create GitHub issue", ["github"]),
            ("Update Jira ticket", ["jira"]),
            ("Add Notion page", ["notion"]),
            ("Generic request", ["github", "jira", "notion"])  # All tools
        ]
        
        for request, expected in test_cases:
            result = self.manager._select_tools(request)
            if len(expected) == 1:
                self.assertEqual(result[0], expected[0])
            else:
                # For generic requests, should return all initialized tools
                self.assertEqual(set(result), set(expected))
    
    def test_cross_tool_context_building(self):
        """Test cross-tool context building."""
        # Mock some tools
        mock_github = Mock()
        mock_github.get_intelligence_stats.return_value = {"stats": "data"}
        
        mock_jira = Mock()
        mock_jira.get_intelligence_stats.return_value = {"stats": "data"}
        
        self.manager.tools = {"github": mock_github, "jira": mock_jira}
        
        context = self.manager._build_cross_tool_context("test request", "notion")
        
        self.assertEqual(context["primary_tool"], "notion")
        self.assertEqual(context["user_request"], "test request")
        self.assertIn("cross_tool_data", context)

class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = IntelligentToolConfig(openai_key="test-key")
        self.manager = IntelligentToolsManager(self.config)
    
    @patch('intelligent_tools_manager.IntelligentGitHubClient')
    @patch('intelligent_tools_manager.IntelligentJiraClient')
    @patch('intelligent_tools_manager.IntelligentNotionClient')
    def test_full_system_workflow(self, mock_notion, mock_jira, mock_github):
        """Test complete system workflow."""
        # Setup mocks
        mock_github_client = Mock()
        mock_jira_client = Mock()
        mock_notion_client = Mock()
        
        mock_github.return_value = mock_github_client
        mock_jira.return_value = mock_jira_client
        mock_notion.return_value = mock_notion_client
        
        # Initialize all tools
        self.manager.initialize_github("token")
        self.manager.initialize_jira("url", "email", "token")
        self.manager.initialize_notion("token")
        
        # Test system stats
        stats = self.manager.get_system_intelligence_stats()
        
        self.assertIn("initialized_tools", stats)
        self.assertIn("system_metrics", stats)
        self.assertEqual(len(stats["initialized_tools"]), 3)
    
    def test_performance_optimization(self):
        """Test performance optimization."""
        # Add some mock cross-tool context
        import time
        current_time = time.time()
        
        self.manager._cross_tool_context = {
            "recent_github_operations": [
                {"timestamp": current_time - 7200},  # 2 hours ago
                {"timestamp": current_time - 90000}  # 25 hours ago
            ]
        }
        
        result = self.manager.optimize_system_performance()
        
        self.assertIn("context_pruned", result)
        self.assertIn("recommendations", result)

def run_comprehensive_tests():
    """Run all tests with detailed reporting."""
    print("Running Intelligent Tools Test Suite")
    print("="*50)
    
    # Create test suite
    test_classes = [
        TestIntelligentToolConfig,
        TestIntelligentToolBase,
        TestIntelligentGitHubClient,
        TestIntelligentJiraClient,
        TestIntelligentNotionClient,  
        TestIntelligentToolsManager,
        TestSystemIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    if result.wasSuccessful():
        print("\nAll tests passed! The intelligent tools system is working correctly.")
    else:
        print(f"\nSome tests failed. Please review the failures and errors above.")
    
    return result.wasSuccessful()

def run_smoke_tests():
    """Run basic smoke tests to verify system functionality."""
    print("\nRunning Smoke Tests")
    print("-"*30)
    
    try:
        # Test 1: Configuration creation
        config = IntelligentToolConfig(openai_key="test-key")
        print("Configuration creation: PASS")
        
        # Test 2: Manager initialization
        manager = IntelligentToolsManager(config)
        print("Manager initialization: PASS")
        
        # Test 3: Tool selection logic
        manager._initialized_tools = {"github", "jira"}
        tools = manager._select_tools("Create GitHub issue")
        assert "github" in tools
        print("Tool selection logic: PASS")
        
        # Test 4: JSON extraction
        from intelligent_tool_base import IntelligentToolBase
        
        class DummyTool(IntelligentToolBase):
            def get_tool_name(self): return "Test"
            def get_available_actions(self): return []
            def execute_base_action(self, action, params): return {}
        
        dummy = DummyTool(config)
        result = dummy._extract_json_from_response('{"test": "value"}')
        assert result == {"test": "value"}
        print("JSON extraction: PASS")
        
        print("\nAll smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"\nSmoke test failed: {e}")
        return False

if __name__ == "__main__":
    print("Intelligent Tools System Test Runner")
    print("===================================")
    
    # Check for optional dependencies
    missing_deps = []
    try:
        import openai
    except ImportError:
        missing_deps.append("openai")
    
    if missing_deps:
        print(f"Missing optional dependencies: {', '.join(missing_deps)}")
        print("Some tests may be skipped or mocked.")
        print()
    
    # Run smoke tests first
    smoke_success = run_smoke_tests()
    
    if smoke_success:
        # Run comprehensive tests
        comprehensive_success = run_comprehensive_tests()
        
        if comprehensive_success:
            print("\nTest Summary: ALL SYSTEMS GO!")
            print("The intelligent tools system is ready for production use.")
        else:
            print("\nTest Summary: Some issues found.")
            print("Please address the failing tests before deployment.")
    else:
        print("\nBasic functionality tests failed. Please check the system setup.")