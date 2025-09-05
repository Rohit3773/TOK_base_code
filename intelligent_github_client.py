#!/usr/bin/env python3
"""
Intelligent GitHub Client - Enhanced with LLM capabilities for better user experience.
"""

import logging
from typing import Any, Dict, List, Optional
from intelligent_tool_base import IntelligentToolBase, IntelligentToolConfig
from github_client import GitHubClient

logger = logging.getLogger(__name__)

class IntelligentGitHubClient(IntelligentToolBase):
    """
    GitHub client enhanced with LLM intelligence.
    
    Features:
    - Smart parameter inference (auto-detect repo, branch, etc.)
    - Context-aware issue/PR management  
    - Intelligent error handling and suggestions
    - Adaptive responses based on user intent
    """
    
    def __init__(self, github_token: str, config: IntelligentToolConfig):
        super().__init__(config)
        self.github_client = GitHubClient(github_token)
        self._current_repo = None
        self._current_owner = None
    
    def get_tool_name(self) -> str:
        return "GitHub"
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Return available GitHub actions with their descriptions."""
        return [
            {
                "name": "get_issue",
                "description": "Get details of a specific issue",
                "parameters": ["owner", "repo", "issue_number"],
                "intelligence_features": ["auto_detect_repo", "smart_issue_lookup"]
            },
            {
                "name": "create_issue", 
                "description": "Create a new issue",
                "parameters": ["owner", "repo", "title", "body", "labels", "assignees"],
                "intelligence_features": ["auto_detect_repo", "smart_labeling", "assignee_suggestions"]
            },
            {
                "name": "search_issues",
                "description": "Search for issues using queries",
                "parameters": ["query", "per_page"],
                "intelligence_features": ["query_optimization", "smart_filtering"]
            },
            {
                "name": "create_pull_request",
                "description": "Create a pull request",
                "parameters": ["owner", "repo", "title", "head", "base", "body"],
                "intelligence_features": ["auto_detect_branches", "smart_pr_templates"]
            },
            {
                "name": "list_issues",
                "description": "List repository issues",
                "parameters": ["owner", "repo", "state", "labels"],
                "intelligence_features": ["auto_detect_repo", "smart_filtering"]
            },
            {
                "name": "list_repositories", 
                "description": "List repositories for the authenticated user",
                "parameters": ["visibility", "affiliation", "type", "sort", "direction"],
                "intelligence_features": ["smart_filtering", "result_organization"]
            }
        ]
    
    def execute_base_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute base GitHub action without intelligence enhancement."""
        try:
            # Map action to GitHub client method
            method = getattr(self.github_client, action, None)
            if not method:
                return {'success': False, 'error': f'Unknown GitHub action: {action}'}
            
            result = method(**params)
            return {'success': True, 'data': result}
            
        except Exception as e:
            logger.error(f"GitHub action {action} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def smart_get_issue(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently get issue details with enhanced context understanding."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="get_issue", 
            suggested_params=kwargs,
            context={
                "current_repo": f"{self._current_owner}/{self._current_repo}" if self._current_owner and self._current_repo else None,
                "action_type": "read",
                "intelligence_features": {
                    "auto_detect_repo": True,
                    "smart_issue_lookup": True,
                    "extract_issue_number": True
                }
            }
        )
    
    def smart_create_issue(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently create issues with smart parameter inference."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="create_issue",
            suggested_params=kwargs,
            context={
                "current_repo": f"{self._current_owner}/{self._current_repo}" if self._current_owner and self._current_repo else None,
                "action_type": "create", 
                "intelligence_features": {
                    "auto_detect_repo": True,
                    "smart_labeling": True,
                    "assignee_suggestions": True,
                    "template_detection": True
                }
            }
        )
    
    def smart_search_issues(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently search issues with query optimization."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="search_issues",
            suggested_params=kwargs,
            context={
                "current_repo": f"{self._current_owner}/{self._current_repo}" if self._current_owner and self._current_repo else None,
                "action_type": "search",
                "intelligence_features": {
                    "query_optimization": True,
                    "smart_filtering": True,
                    "result_ranking": True
                }
            }
        )
    
    def smart_list_repositories(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently list repositories with smart filtering."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="list_repositories",
            suggested_params=kwargs,
            context={
                "action_type": "list",
                "intelligence_features": {
                    "smart_filtering": True,
                    "result_organization": True,
                    "visibility_optimization": True
                }
            }
        )
    
    def set_current_repo(self, owner: str, repo: str):
        """Set current repository context for intelligent operations."""
        self._current_owner = owner
        self._current_repo = repo
        logger.info(f"Set GitHub context to {owner}/{repo}")
    
    def _understand_and_plan(
        self, 
        user_request: str,
        suggested_action: str = None,
        suggested_params: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced GitHub-specific planning with repository intelligence."""
        
        # First call parent method for general planning
        base_plan = super()._understand_and_plan(user_request, suggested_action, suggested_params, context)
        
        # Add GitHub-specific intelligence
        enhanced_params = base_plan.get('parameters', {}).copy()
        
        # Auto-detect repository if not provided
        if suggested_action in ['get_issue', 'create_issue', 'list_issues'] and not (enhanced_params.get('owner') and enhanced_params.get('repo')):
            repo_info = self._extract_repo_from_request(user_request)
            if repo_info:
                enhanced_params.update(repo_info)
            elif self._current_owner and self._current_repo:
                enhanced_params['owner'] = self._current_owner
                enhanced_params['repo'] = self._current_repo
        
        # Auto-detect issue number for issue operations  
        if suggested_action == 'get_issue' and not enhanced_params.get('issue_number'):
            issue_number = self._extract_issue_number(user_request)
            if issue_number:
                enhanced_params['issue_number'] = issue_number
        
        # Enhance search queries
        if suggested_action == 'search_issues' and enhanced_params.get('query'):
            enhanced_query = self._optimize_search_query(enhanced_params['query'], user_request)
            enhanced_params['query'] = enhanced_query
        
        base_plan['parameters'] = enhanced_params
        base_plan['github_intelligence'] = True
        
        return base_plan
    
    def _extract_repo_from_request(self, user_request: str) -> Optional[Dict[str, str]]:
        """Extract owner/repo from user request using various patterns."""
        import re
        
        patterns = [
            r'(?:github\.com/|^|[\s])([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)(?:[/\s]|$)',  # owner/repo format
            r'repository\s+([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)',  # "repository owner/repo"
            r'repo\s+([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)',  # "repo owner/repo"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                return {'owner': match.group(1), 'repo': match.group(2)}
        
        return None
    
    def _extract_issue_number(self, user_request: str) -> Optional[int]:
        """Extract issue number from user request."""
        import re
        
        patterns = [
            r'#(\d+)',  # #123
            r'issue\s+#?(\d+)',  # issue 123 or issue #123
            r'number\s+(\d+)',  # number 123
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _optimize_search_query(self, original_query: str, user_request: str) -> str:
        """Optimize search query based on user intent and repository context."""
        
        # Add repository context if available
        if self._current_owner and self._current_repo and f"repo:{self._current_owner}/{self._current_repo}" not in original_query:
            original_query = f"repo:{self._current_owner}/{self._current_repo} {original_query}"
        
        # Add common search improvements based on user request
        user_lower = user_request.lower()
        
        if 'open' in user_lower and 'state:' not in original_query:
            original_query += ' state:open'
        elif 'closed' in user_lower and 'state:' not in original_query:
            original_query += ' state:closed'
        
        if 'bug' in user_lower and 'label:' not in original_query:
            original_query += ' label:bug'
        
        return original_query.strip()

    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get statistics about intelligent operations."""
        
        base_stats = self.get_learned_patterns()
        
        github_stats = {
            'current_repo_context': f"{self._current_owner}/{self._current_repo}" if self._current_owner and self._current_repo else None,
            'total_intelligent_operations': sum(pattern['total_interactions'] for pattern in base_stats.values()),
            'average_success_rate': sum(pattern['success_rate'] for pattern in base_stats.values()) / len(base_stats) if base_stats else 0,
            'pattern_breakdown': base_stats
        }
        
        return github_stats