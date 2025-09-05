#!/usr/bin/env python3
"""
Intelligent Jira Client - Enhanced with LLM capabilities for better user experience.
"""

import logging
from typing import Any, Dict, List, Optional
from intelligent_tool_base import IntelligentToolBase, IntelligentToolConfig
from jira_client import JiraClient

logger = logging.getLogger(__name__)

class IntelligentJiraClient(IntelligentToolBase):
    """
    Jira client enhanced with LLM intelligence.
    
    Features:
    - Smart issue key detection and validation
    - Context-aware status transitions with natural language
    - Intelligent JQL query building
    - Adaptive project management suggestions
    """
    
    def __init__(self, jira_base_url: str, jira_email: str, jira_token: str, config: IntelligentToolConfig):
        super().__init__(config)
        self.jira_client = JiraClient(jira_base_url, jira_email, jira_token)
        self._current_project = None
        self._known_statuses = {}
        self._known_issue_types = {}
    
    def get_tool_name(self) -> str:
        return "Jira"
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Return available Jira actions with their descriptions."""
        return [
            {
                "name": "get_issue",
                "description": "Get details of a specific issue",
                "parameters": ["issue_key"],
                "intelligence_features": ["auto_detect_issue_key", "context_enrichment"]
            },
            {
                "name": "create_issue", 
                "description": "Create a new issue",
                "parameters": ["project_key", "summary", "description", "issuetype_name", "priority"],
                "intelligence_features": ["auto_detect_project", "smart_issue_typing", "priority_suggestions"]
            },
            {
                "name": "search",
                "description": "Search issues using JQL",
                "parameters": ["jql", "max_results"],
                "intelligence_features": ["jql_optimization", "natural_language_to_jql"]
            },
            {
                "name": "transition_issue",
                "description": "Change issue status",
                "parameters": ["issue_key", "transition_id", "target_status"],
                "intelligence_features": ["natural_language_status", "smart_transitions"]
            },
            {
                "name": "add_comment",
                "description": "Add comment to issue",
                "parameters": ["issue_key", "body"],
                "intelligence_features": ["auto_detect_issue_key", "comment_enhancement"]
            },
            {
                "name": "get_project_details",
                "description": "Get project information",
                "parameters": ["project_key"],
                "intelligence_features": ["auto_detect_project", "project_insights"]
            }
        ]
    
    def execute_base_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute base Jira action without intelligence enhancement."""
        try:
            # Map action to Jira client method
            method = getattr(self.jira_client, action, None)
            if not method:
                return {'success': False, 'error': f'Unknown Jira action: {action}'}
            
            result = method(**params)
            return {'success': True, 'data': result}
            
        except Exception as e:
            logger.error(f"Jira action {action} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def smart_get_issue(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently get issue details with enhanced context."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="get_issue", 
            suggested_params=kwargs,
            context={
                "current_project": self._current_project,
                "action_type": "read",
                "intelligence_features": {
                    "auto_detect_issue_key": True,
                    "context_enrichment": True,
                    "related_issues": True
                }
            }
        )
    
    def smart_create_issue(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently create issues with smart categorization."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="create_issue",
            suggested_params=kwargs,
            context={
                "current_project": self._current_project,
                "action_type": "create",
                "known_issue_types": self._known_issue_types,
                "intelligence_features": {
                    "auto_detect_project": True,
                    "smart_issue_typing": True,
                    "priority_suggestions": True,
                    "template_suggestions": True
                }
            }
        )
    
    def smart_transition_issue(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently transition issues using natural language status."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="transition_issue",
            suggested_params=kwargs,
            context={
                "current_project": self._current_project,
                "known_statuses": self._known_statuses,
                "action_type": "modify",
                "intelligence_features": {
                    "natural_language_status": True,
                    "smart_transitions": True,
                    "auto_detect_issue_key": True
                }
            }
        )
    
    def smart_search(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently search with natural language to JQL conversion."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="search",
            suggested_params=kwargs,
            context={
                "current_project": self._current_project,
                "action_type": "search",
                "intelligence_features": {
                    "natural_language_to_jql": True,
                    "jql_optimization": True,
                    "smart_filtering": True
                }
            }
        )
    
    def set_current_project(self, project_key: str):
        """Set current project context for intelligent operations."""
        self._current_project = project_key
        
        # Load project-specific context
        try:
            project_details = self.jira_client.get_project_details(project_key)
            
            # Cache issue types for this project  
            if 'issueTypes' in project_details:
                self._known_issue_types[project_key] = {
                    it['name'].lower(): it for it in project_details['issueTypes']
                }
            
            logger.info(f"Set Jira context to project {project_key}")
        except Exception as e:
            logger.warning(f"Could not load context for project {project_key}: {e}")
    
    def _understand_and_plan(
        self, 
        user_request: str,
        suggested_action: str = None,
        suggested_params: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced Jira-specific planning with issue key and status intelligence."""
        
        # First call parent method for general planning
        base_plan = super()._understand_and_plan(user_request, suggested_action, suggested_params, context)
        
        # Add Jira-specific intelligence
        enhanced_params = base_plan.get('parameters', {}).copy()
        
        # Auto-detect issue key if not provided
        if suggested_action in ['get_issue', 'transition_issue', 'add_comment'] and not enhanced_params.get('issue_key'):
            issue_key = self._extract_issue_key(user_request)
            if issue_key:
                enhanced_params['issue_key'] = issue_key
        
        # Auto-detect project key for create operations
        if suggested_action == 'create_issue' and not enhanced_params.get('project_key'):
            project_key = self._extract_project_key(user_request) or self._current_project
            if project_key:
                enhanced_params['project_key'] = project_key
        
        # Convert natural language status to transition
        if suggested_action == 'transition_issue' and not enhanced_params.get('target_status'):
            target_status = self._extract_target_status(user_request)
            if target_status:
                enhanced_params['target_status'] = target_status
        
        # Convert natural language to JQL for search
        if suggested_action == 'search' and enhanced_params.get('jql'):
            if not self._looks_like_jql(enhanced_params['jql']):
                enhanced_jql = self._convert_natural_language_to_jql(enhanced_params['jql'], user_request)
                enhanced_params['jql'] = enhanced_jql
        
        # Infer issue type and priority for create operations
        if suggested_action == 'create_issue':
            if not enhanced_params.get('issuetype_name'):
                issue_type = self._infer_issue_type(user_request)
                if issue_type:
                    enhanced_params['issuetype_name'] = issue_type
            
            if not enhanced_params.get('priority'):
                priority = self._infer_priority(user_request)
                if priority:
                    enhanced_params['priority'] = priority
        
        base_plan['parameters'] = enhanced_params
        base_plan['jira_intelligence'] = True
        
        return base_plan
    
    def _extract_issue_key(self, user_request: str) -> Optional[str]:
        """Extract Jira issue key from user request."""
        import re
        
        # Pattern for Jira issue keys (PROJECT-123)
        pattern = r'\b([A-Z]+[-_][0-9]+)\b'
        match = re.search(pattern, user_request, re.IGNORECASE)
        
        if match:
            return match.group(1).upper()
        
        return None
    
    def _extract_project_key(self, user_request: str) -> Optional[str]:
        """Extract project key from user request."""
        import re
        
        patterns = [
            r'project\s+([A-Z]+)',  # "project ABC"
            r'in\s+([A-Z]+)\s+project',  # "in ABC project"
            r'\bfor\s+([A-Z]+)\b',  # "for ABC"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _extract_target_status(self, user_request: str) -> Optional[str]:
        """Extract target status from natural language.""" 
        status_mapping = {
            'done': ['done', 'complete', 'completed', 'finished', 'resolved', 'closed'],
            'in progress': ['in progress', 'progress', 'working', 'started', 'doing'],
            'todo': ['todo', 'to do', 'open', 'new', 'backlog'],
            'blocked': ['blocked', 'blocked', 'stuck', 'waiting']
        }
        
        user_lower = user_request.lower()
        
        for status, keywords in status_mapping.items():
            if any(keyword in user_lower for keyword in keywords):
                return status
        
        return None
    
    def _looks_like_jql(self, query: str) -> bool:
        """Check if a query looks like JQL syntax."""
        jql_keywords = ['project', 'status', 'assignee', 'reporter', 'created', 'updated', 'AND', 'OR', '=', '!=', '~']
        return any(keyword in query for keyword in jql_keywords)
    
    def _convert_natural_language_to_jql(self, natural_query: str, full_request: str) -> str:
        """Convert natural language to JQL query."""
        
        jql_parts = []
        
        # Add project context if available
        if self._current_project:
            jql_parts.append(f'project = "{self._current_project}"')
        
        # Convert common natural language patterns
        lower_query = natural_query.lower()
        
        if 'open' in lower_query or 'active' in lower_query:
            jql_parts.append('status != Done')
        elif 'closed' in lower_query or 'done' in lower_query:
            jql_parts.append('status = Done')
        
        if 'assigned to me' in lower_query:
            jql_parts.append('assignee = currentUser()')
        elif 'unassigned' in lower_query:
            jql_parts.append('assignee is EMPTY')
        
        if 'bug' in lower_query:
            jql_parts.append('type = Bug')
        elif 'task' in lower_query:
            jql_parts.append('type = Task')
        
        # If we couldn't parse anything useful, use text search
        if not jql_parts and natural_query:
            if self._current_project:
                jql_parts = [f'project = "{self._current_project}"', f'text ~ "{natural_query}"']
            else:
                jql_parts = [f'text ~ "{natural_query}"']
        
        return ' AND '.join(jql_parts) if jql_parts else natural_query
    
    def _infer_issue_type(self, user_request: str) -> Optional[str]:
        """Infer issue type from user request."""
        
        type_keywords = {
            'Bug': ['bug', 'error', 'broken', 'fix', 'issue', 'problem'],
            'Task': ['task', 'work', 'do', 'implement', 'add'],  
            'Story': ['story', 'feature', 'user story', 'requirement'],
            'Epic': ['epic', 'large', 'major']
        }
        
        user_lower = user_request.lower()
        
        for issue_type, keywords in type_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return issue_type
        
        return 'Task'  # Default fallback
    
    def _infer_priority(self, user_request: str) -> Optional[str]:
        """Infer priority from user request."""
        
        priority_keywords = {
            'Highest': ['urgent', 'critical', 'emergency', 'asap', 'immediately'],
            'High': ['high', 'important', 'priority', 'soon'],
            'Medium': ['medium', 'normal', 'regular'],
            'Low': ['low', 'minor', 'whenever', 'eventually']
        }
        
        user_lower = user_request.lower()
        
        for priority, keywords in priority_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                return priority
        
        return None  # Let Jira use default

    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get statistics about intelligent Jira operations."""
        
        base_stats = self.get_learned_patterns()
        
        jira_stats = {
            'current_project_context': self._current_project,
            'known_issue_types': len(self._known_issue_types),
            'known_statuses': len(self._known_statuses),
            'total_intelligent_operations': sum(pattern['total_interactions'] for pattern in base_stats.values()),
            'average_success_rate': sum(pattern['success_rate'] for pattern in base_stats.values()) / len(base_stats) if base_stats else 0,
            'pattern_breakdown': base_stats
        }
        
        return jira_stats