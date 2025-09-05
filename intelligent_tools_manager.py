#!/usr/bin/env python3
"""
Intelligent Tools Manager - Coordinates all intelligent tools and provides unified interface.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from intelligent_tool_base import IntelligentToolConfig
from intelligent_github_client import IntelligentGitHubClient
from intelligent_jira_client import IntelligentJiraClient
from intelligent_notion_client import IntelligentNotionClient

logger = logging.getLogger(__name__)

class IntelligentToolsManager:
    """
    Manages all intelligent tools and provides unified interface.
    
    Features:
    - Automatic tool selection based on user intent
    - Cross-tool context sharing and workflow coordination
    - Unified intelligent operations across all platforms
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: IntelligentToolConfig):
        self.config = config
        self.tools = {}
        self._initialized_tools = set()
        self._cross_tool_context = {}
    
    def initialize_github(self, github_token: str, owner: str = None, repo: str = None) -> IntelligentGitHubClient:
        """Initialize intelligent GitHub client."""
        try:
            github_client = IntelligentGitHubClient(github_token, self.config)
            if owner and repo:
                github_client.set_current_repo(owner, repo)
            
            self.tools['github'] = github_client
            self._initialized_tools.add('github')
            logger.info("Initialized intelligent GitHub client")
            
            return github_client
        except Exception as e:
            logger.error(f"Failed to initialize GitHub client: {e}")
            raise
    
    def initialize_jira(self, jira_base_url: str, jira_email: str, jira_token: str, project_key: str = None) -> IntelligentJiraClient:
        """Initialize intelligent Jira client."""
        try:
            jira_client = IntelligentJiraClient(jira_base_url, jira_email, jira_token, self.config)
            if project_key:
                jira_client.set_current_project(project_key)
            
            self.tools['jira'] = jira_client
            self._initialized_tools.add('jira')
            logger.info("Initialized intelligent Jira client")
            
            return jira_client
        except Exception as e:
            logger.error(f"Failed to initialize Jira client: {e}")
            raise
    
    def initialize_notion(self, notion_token: str, database_id: str = None) -> IntelligentNotionClient:
        """Initialize intelligent Notion client."""
        try:
            notion_client = IntelligentNotionClient(notion_token, self.config)
            if database_id:
                notion_client.set_current_database(database_id)
            
            self.tools['notion'] = notion_client
            self._initialized_tools.add('notion')
            logger.info("Initialized intelligent Notion client")
            
            return notion_client
        except Exception as e:
            logger.error(f"Failed to initialize Notion client: {e}")
            raise
    
    def intelligent_execute(
        self, 
        user_request: str, 
        preferred_tool: str = None,
        action_hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute user request intelligently across all available tools.
        
        Args:
            user_request: Natural language request from user
            preferred_tool: Preferred tool to use (optional)
            action_hints: Hints about what action/parameters to use
            
        Returns:
            Enhanced result with cross-tool intelligence
        """
        
        try:
            # Step 1: Determine which tool(s) to use
            selected_tools = self._select_tools(user_request, preferred_tool)
            
            if not selected_tools:
                return {
                    'success': False,
                    'error': 'No suitable tools available for this request',
                    'available_tools': list(self._initialized_tools)
                }
            
            # Step 2: Execute on primary tool with cross-tool context
            primary_tool = selected_tools[0]
            
            enhanced_context = self._build_cross_tool_context(user_request, primary_tool)
            
            if primary_tool == 'github' and 'github' in self.tools:
                result = self._execute_github_intelligently(user_request, enhanced_context, action_hints)
            elif primary_tool == 'jira' and 'jira' in self.tools:
                result = self._execute_jira_intelligently(user_request, enhanced_context, action_hints)
            elif primary_tool == 'notion' and 'notion' in self.tools:
                result = self._execute_notion_intelligently(user_request, enhanced_context, action_hints)
            else:
                return {
                    'success': False,
                    'error': f'Tool {primary_tool} not initialized or available'
                }
            
            # Step 3: Apply cross-tool enhancements
            enhanced_result = self._apply_cross_tool_enhancements(result, user_request, selected_tools)
            
            # Step 4: Update cross-tool context for future requests
            self._update_cross_tool_context(user_request, primary_tool, enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Intelligent execution failed: {e}")
            return {
                'success': False,
                'error': f'Intelligent execution failed: {str(e)}',
                'fallback_suggestion': 'Try specifying which tool to use (github/jira/notion)'
            }
    
    def _select_tools(self, user_request: str, preferred_tool: str = None) -> List[str]:
        """Select appropriate tools based on user request."""
        
        if preferred_tool and preferred_tool in self._initialized_tools:
            return [preferred_tool]
        
        # Analyze user request to determine tool preference
        request_lower = user_request.lower()
        
        tool_keywords = {
            'github': ['github', 'repository', 'repo', 'issue', 'pull request', 'pr', 'branch', 'commit'],
            'jira': ['jira', 'ticket', 'story', 'epic', 'sprint', 'project', 'task', 'bug'],
            'notion': ['notion', 'page', 'database', 'note', 'document', 'wiki']
        }
        
        tool_scores = {}
        
        for tool, keywords in tool_keywords.items():
            if tool in self._initialized_tools:
                score = sum(1 for keyword in keywords if keyword in request_lower)
                if score > 0:
                    tool_scores[tool] = score
        
        # Sort by score and return
        if tool_scores:
            sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
            return [tool for tool, _ in sorted_tools]
        
        # Fallback to all available tools
        return list(self._initialized_tools)
    
    def _build_cross_tool_context(self, user_request: str, primary_tool: str) -> Dict[str, Any]:
        """Build context that includes information from other tools."""
        
        context = {
            'primary_tool': primary_tool,
            'user_request': user_request,
            'available_tools': list(self._initialized_tools),
            'cross_tool_data': {}
        }
        
        # Add context from other tools
        for tool_name, tool_client in self.tools.items():
            if tool_name != primary_tool:
                try:
                    if hasattr(tool_client, 'get_intelligence_stats'):
                        stats = tool_client.get_intelligence_stats()
                        context['cross_tool_data'][tool_name] = {
                            'stats': stats,
                            'recent_patterns': stats.get('pattern_breakdown', {})
                        }
                except Exception as e:
                    logger.debug(f"Could not get stats from {tool_name}: {e}")
        
        return context
    
    def _execute_github_intelligently(
        self, 
        user_request: str, 
        context: Dict[str, Any], 
        action_hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute GitHub operations with intelligence."""
        
        github_client = self.tools['github']
        
        # Determine specific GitHub action based on request
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ['get', 'show', 'display']) and 'issue' in request_lower:
            return github_client.smart_get_issue(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['create', 'new']) and 'issue' in request_lower:
            return github_client.smart_create_issue(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['search', 'find']) and 'issue' in request_lower:
            return github_client.smart_search_issues(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['list', 'show', 'get']) and any(word in request_lower for word in ['repositories', 'repos', 'repository']):
            return github_client.smart_list_repositories(user_request, **(action_hints or {}))
        else:
            # Fallback to general intelligent execution
            return github_client.intelligent_execute(
                user_request, 
                context=context,
                suggested_params=action_hints
            )
    
    def _execute_jira_intelligently(
        self, 
        user_request: str, 
        context: Dict[str, Any], 
        action_hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute Jira operations with intelligence."""
        
        jira_client = self.tools['jira']
        
        # Determine specific Jira action based on request
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ['get', 'show', 'display']) and any(word in request_lower for word in ['issue', 'ticket']):
            return jira_client.smart_get_issue(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['create', 'new']) and any(word in request_lower for word in ['issue', 'ticket']):
            return jira_client.smart_create_issue(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['transition', 'move', 'change']) and 'status' in request_lower:
            return jira_client.smart_transition_issue(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['search', 'find']):
            return jira_client.smart_search(user_request, **(action_hints or {}))
        else:
            # Fallback to general intelligent execution
            return jira_client.intelligent_execute(
                user_request,
                context=context,
                suggested_params=action_hints
            )
    
    def _execute_notion_intelligently(
        self, 
        user_request: str, 
        context: Dict[str, Any], 
        action_hints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute Notion operations with intelligence."""
        
        notion_client = self.tools['notion']
        
        # Determine specific Notion action based on request
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ['create', 'new']) and 'page' in request_lower:
            return notion_client.smart_create_page(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['add', 'append']) and 'text' in request_lower:
            return notion_client.smart_append_text(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['update', 'change']) and 'status' in request_lower:
            return notion_client.smart_update_status(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['search', 'find']):
            return notion_client.smart_search(user_request, **(action_hints or {}))
        elif any(word in request_lower for word in ['query', 'list']) and 'database' in request_lower:
            return notion_client.smart_query_database(user_request, **(action_hints or {}))
        else:
            # Fallback to general intelligent execution
            return notion_client.intelligent_execute(
                user_request,
                context=context,
                suggested_params=action_hints
            )
    
    def _apply_cross_tool_enhancements(
        self, 
        result: Dict[str, Any], 
        user_request: str, 
        selected_tools: List[str]
    ) -> Dict[str, Any]:
        """Apply cross-tool enhancements to the result."""
        
        # Add suggestions for related actions in other tools
        if result.get('success') and len(selected_tools) > 1:
            suggestions = []
            
            # Example: If creating GitHub issue, suggest creating Jira ticket
            if 'github' in selected_tools and 'jira' in selected_tools:
                if 'issue' in user_request.lower() and 'create' in user_request.lower():
                    suggestions.append("Consider creating a related Jira ticket for project management")
            
            # Example: If creating Jira issue, suggest adding to Notion documentation
            if 'jira' in selected_tools and 'notion' in selected_tools:
                if 'issue' in user_request.lower():
                    suggestions.append("Consider documenting this in Notion for team visibility")
            
            if suggestions:
                result['cross_tool_suggestions'] = suggestions
        
        # Add cross-tool metadata
        result['intelligent_execution'] = True
        result['tools_considered'] = selected_tools
        result['cross_tool_enhancements'] = True
        
        return result
    
    def _update_cross_tool_context(
        self, 
        user_request: str, 
        primary_tool: str, 
        result: Dict[str, Any]
    ):
        """Update cross-tool context based on execution results."""
        
        context_key = f"recent_{primary_tool}_operations"
        
        if context_key not in self._cross_tool_context:
            self._cross_tool_context[context_key] = []
        
        self._cross_tool_context[context_key].append({
            'request': user_request,
            'success': result.get('success', False),
            'timestamp': __import__('time').time(),
            'result_summary': result.get('user_message', 'Operation completed')
        })
        
        # Keep only recent operations (last 50)
        if len(self._cross_tool_context[context_key]) > 50:
            self._cross_tool_context[context_key] = self._cross_tool_context[context_key][-50:]
    
    def get_system_intelligence_stats(self) -> Dict[str, Any]:
        """Get comprehensive intelligence statistics across all tools."""
        
        system_stats = {
            'initialized_tools': list(self._initialized_tools),
            'total_cross_tool_operations': len(self._cross_tool_context),
            'tool_stats': {}
        }
        
        # Get stats from each tool
        for tool_name, tool_client in self.tools.items():
            try:
                if hasattr(tool_client, 'get_intelligence_stats'):
                    system_stats['tool_stats'][tool_name] = tool_client.get_intelligence_stats()
            except Exception as e:
                logger.debug(f"Could not get stats from {tool_name}: {e}")
        
        # Calculate overall system intelligence metrics
        total_ops = sum(
            stats.get('total_intelligent_operations', 0) 
            for stats in system_stats['tool_stats'].values()
        )
        
        avg_success_rate = sum(
            stats.get('average_success_rate', 0) 
            for stats in system_stats['tool_stats'].values()
        ) / len(system_stats['tool_stats']) if system_stats['tool_stats'] else 0
        
        system_stats['system_metrics'] = {
            'total_intelligent_operations': total_ops,
            'average_success_rate': avg_success_rate,
            'tools_active': len(self._initialized_tools),
            'cross_tool_context_size': sum(len(context) for context in self._cross_tool_context.values())
        }
        
        return system_stats
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on usage patterns."""
        
        optimizations = {
            'cache_cleared': False,
            'context_pruned': False,
            'recommendations': []
        }
        
        # Clear old cross-tool context
        current_time = __import__('time').time()
        for context_key, operations in self._cross_tool_context.items():
            old_count = len(operations)
            # Keep only operations from last 24 hours
            self._cross_tool_context[context_key] = [
                op for op in operations 
                if current_time - op['timestamp'] < 86400  # 24 hours
            ]
            new_count = len(self._cross_tool_context[context_key])
            
            if old_count != new_count:
                optimizations['context_pruned'] = True
        
        # Generate performance recommendations
        stats = self.get_system_intelligence_stats()
        
        if stats['system_metrics']['average_success_rate'] < 0.8:
            optimizations['recommendations'].append(
                "Consider reviewing failed operations to improve success rate"
            )
        
        if stats['system_metrics']['total_intelligent_operations'] > 1000:
            optimizations['recommendations'].append(
                "High usage detected - consider caching frequently used patterns"
            )
        
        return optimizations