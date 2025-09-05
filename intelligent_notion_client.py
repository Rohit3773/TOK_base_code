#!/usr/bin/env python3
"""
Intelligent Notion Client - Enhanced with LLM capabilities for better user experience.
"""

import logging
from typing import Any, Dict, List, Optional
from intelligent_tool_base import IntelligentToolBase, IntelligentToolConfig
from notion_client import NotionClient

logger = logging.getLogger(__name__)

class IntelligentNotionClient(IntelligentToolBase):
    """
    Notion client enhanced with LLM intelligence.
    
    Features:
    - Smart page and database detection
    - Context-aware content creation and updates
    - Intelligent property inference and management
    - Adaptive search and organization suggestions
    """
    
    def __init__(self, notion_token: str, config: IntelligentToolConfig):
        super().__init__(config)
        self.notion_client = NotionClient(notion_token)
        self._current_database = None
        self._known_databases = {}
        self._known_pages = {}
    
    def get_tool_name(self) -> str:
        return "Notion"
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Return available Notion actions with their descriptions."""
        return [
            {
                "name": "get_page",
                "description": "Get details of a specific page",
                "parameters": ["page_id"],
                "intelligence_features": ["auto_detect_page", "context_enrichment"]
            },
            {
                "name": "create_page", 
                "description": "Create a new page",
                "parameters": ["parent_database_id", "parent_page_id", "properties", "children"],
                "intelligence_features": ["auto_detect_parent", "smart_properties", "template_suggestions"]
            },
            {
                "name": "update_page",
                "description": "Update page properties",
                "parameters": ["page_id", "properties"],
                "intelligence_features": ["auto_detect_page", "smart_property_inference"]
            },
            {
                "name": "append_text_to_page",
                "description": "Append text content to a page",
                "parameters": ["page_name", "text"],
                "intelligence_features": ["auto_detect_page", "content_enhancement"]
            },
            {
                "name": "update_status_with_database",
                "description": "Update page status with database context",
                "parameters": ["page_name", "status", "database_id"],
                "intelligence_features": ["natural_language_status", "status_validation"]
            },
            {
                "name": "query_database",
                "description": "Query database for pages",
                "parameters": ["database_id", "filter_conditions", "sorts"],
                "intelligence_features": ["smart_filtering", "natural_language_queries"]
            },
            {
                "name": "search",
                "description": "Search across Notion workspace",
                "parameters": ["query"],
                "intelligence_features": ["query_optimization", "result_ranking"]
            }
        ]
    
    def execute_base_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute base Notion action without intelligence enhancement."""
        try:
            # Map action to Notion client method
            method = getattr(self.notion_client, action, None)
            if not method:
                return {'success': False, 'error': f'Unknown Notion action: {action}'}
            
            result = method(**params)
            return {'success': True, 'data': result}
            
        except Exception as e:
            logger.error(f"Notion action {action} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def smart_create_page(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently create pages with smart property inference."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="create_page",
            suggested_params=kwargs,
            context={
                "current_database": self._current_database,
                "known_databases": self._known_databases,
                "action_type": "create",
                "intelligence_features": {
                    "auto_detect_parent": True,
                    "smart_properties": True,
                    "template_suggestions": True,
                    "content_structuring": True
                }
            }
        )
    
    def smart_append_text(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently append text with content enhancement."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="append_text_to_page",
            suggested_params=kwargs,
            context={
                "current_database": self._current_database,
                "known_pages": self._known_pages,
                "action_type": "modify",
                "intelligence_features": {
                    "auto_detect_page": True,
                    "content_enhancement": True,
                    "formatting_suggestions": True
                }
            }
        )
    
    def smart_update_status(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently update page status with natural language."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="update_status_with_database",
            suggested_params=kwargs,
            context={
                "current_database": self._current_database,
                "action_type": "modify",
                "intelligence_features": {
                    "natural_language_status": True,
                    "status_validation": True,
                    "auto_detect_page": True
                }
            }
        )
    
    def smart_search(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently search with query optimization."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="search",
            suggested_params=kwargs,
            context={
                "current_database": self._current_database,
                "action_type": "search",
                "intelligence_features": {
                    "query_optimization": True,
                    "result_ranking": True,
                    "context_aware_search": True
                }
            }
        )
    
    def smart_query_database(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Intelligently query database with natural language filters."""
        
        return self.intelligent_execute(
            user_request=user_request,
            suggested_action="query_database",
            suggested_params=kwargs,
            context={
                "current_database": self._current_database,
                "action_type": "search",
                "intelligence_features": {
                    "smart_filtering": True,
                    "natural_language_queries": True,
                    "result_organization": True
                }
            }
        )
    
    def set_current_database(self, database_id: str):
        """Set current database context for intelligent operations."""
        self._current_database = database_id
        
        # Load database schema and properties
        try:
            db_details = self.notion_client.get_database(database_id)
            self._known_databases[database_id] = {
                'title': db_details.get('title', []),
                'properties': db_details.get('properties', {}),
                'url': db_details.get('url', ''),
                'last_edited': db_details.get('last_edited_time', '')
            }
            
            logger.info(f"Set Notion context to database {database_id}")
        except Exception as e:
            logger.warning(f"Could not load context for database {database_id}: {e}")
    
    def _understand_and_plan(
        self, 
        user_request: str,
        suggested_action: str = None,
        suggested_params: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Enhanced Notion-specific planning with page and database intelligence."""
        
        # First call parent method for general planning
        base_plan = super()._understand_and_plan(user_request, suggested_action, suggested_params, context)
        
        # Add Notion-specific intelligence
        enhanced_params = base_plan.get('parameters', {}).copy()
        
        # Auto-detect page name for page operations
        if suggested_action in ['append_text_to_page', 'update_status_with_database'] and not enhanced_params.get('page_name'):
            page_name = self._extract_page_name(user_request)
            if page_name:
                enhanced_params['page_name'] = page_name
        
        # Auto-detect database context
        if suggested_action in ['create_page', 'update_status_with_database'] and not enhanced_params.get('database_id') and not enhanced_params.get('parent_database_id'):
            if self._current_database:
                if suggested_action == 'create_page':
                    enhanced_params['parent_database_id'] = self._current_database
                else:
                    enhanced_params['database_id'] = self._current_database
        
        # Convert natural language status
        if suggested_action == 'update_status_with_database' and not enhanced_params.get('status'):
            status = self._extract_status(user_request)
            if status:
                enhanced_params['status'] = status
        
        # Enhance search queries
        if suggested_action == 'search' and enhanced_params.get('query'):
            enhanced_query = self._optimize_search_query(enhanced_params['query'], user_request)
            enhanced_params['query'] = enhanced_query
        
        # Infer page properties for creation
        if suggested_action == 'create_page' and not enhanced_params.get('properties'):
            properties = self._infer_page_properties(user_request)
            if properties:
                enhanced_params['properties'] = properties
        
        base_plan['parameters'] = enhanced_params
        base_plan['notion_intelligence'] = True
        
        return base_plan
    
    def _extract_page_name(self, user_request: str) -> Optional[str]:
        """Extract page name from user request using various patterns."""
        import re
        
        patterns = [
            r'(?:page|to)\s+["\']([^"\']+)["\']',  # page "Name" or to "Name"
            r'(?:page|to)\s+([A-Z][A-Za-z0-9\s]+?)(?:\s|$)',  # page Title Case
            r'in\s+([A-Z][A-Za-z0-9\s]+?)(?:\s(?:page|database)|$)',  # in PageName
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_status(self, user_request: str) -> Optional[str]:
        """Extract status value from natural language."""
        status_mapping = {
            'Done': ['done', 'complete', 'completed', 'finished', 'resolved'],
            'In Progress': ['in progress', 'progress', 'working', 'started', 'doing'],
            'To Do': ['todo', 'to do', 'pending', 'open', 'new'],
            'Blocked': ['blocked', 'stuck', 'waiting', 'paused'],
            'Review': ['review', 'reviewing', 'needs review'],
            'Testing': ['testing', 'test', 'qa']
        }
        
        user_lower = user_request.lower()
        
        for status, keywords in status_mapping.items():
            if any(keyword in user_lower for keyword in keywords):
                return status
        
        return None
    
    def _optimize_search_query(self, original_query: str, user_request: str) -> str:
        """Optimize search query based on user intent."""
        
        # Add context-specific improvements
        user_lower = user_request.lower()
        
        # Don't modify if already looks optimized
        if len(original_query.split()) > 3:
            return original_query
        
        # Add common search improvements
        optimized_terms = []
        
        if 'recent' in user_lower or 'latest' in user_lower:
            optimized_terms.append(original_query)  # Keep original for recency
        elif 'old' in user_lower or 'archived' in user_lower:
            optimized_terms.append(original_query + ' archived')
        else:
            optimized_terms.append(original_query)
        
        return ' '.join(optimized_terms)
    
    def _infer_page_properties(self, user_request: str) -> Optional[Dict[str, Any]]:
        """Infer page properties from user request and current database schema."""
        
        if not self._current_database or self._current_database not in self._known_databases:
            return None
        
        db_properties = self._known_databases[self._current_database].get('properties', {})
        inferred_properties = {}
        
        # Extract title if available
        title_property = None
        for prop_name, prop_config in db_properties.items():
            if prop_config.get('type') == 'title':
                title_property = prop_name
                break
        
        if title_property:
            # Try to extract title from request
            title = self._extract_title_from_request(user_request)
            if title:
                inferred_properties[title_property] = {
                    'title': [{'type': 'text', 'text': {'content': title}}]
                }
        
        # Infer status if available
        for prop_name, prop_config in db_properties.items():
            if prop_config.get('type') == 'select' and 'status' in prop_name.lower():
                status = self._extract_status(user_request)
                if status:
                    inferred_properties[prop_name] = {'select': {'name': status}}
                break
        
        return inferred_properties if inferred_properties else None
    
    def _extract_title_from_request(self, user_request: str) -> Optional[str]:
        """Extract title for new page from user request.""" 
        import re
        
        patterns = [
            r'create\s+(?:page\s+)?["\']([^"\']+)["\']',  # create page "Title"
            r'new\s+page\s+["\']([^"\']+)["\']',  # new page "Title"
            r'page\s+called\s+["\']([^"\']+)["\']',  # page called "Title"
            r'titled\s+["\']([^"\']+)["\']',  # titled "Title"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_request, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get statistics about intelligent Notion operations."""
        
        base_stats = self.get_learned_patterns()
        
        notion_stats = {
            'current_database_context': self._current_database,
            'known_databases': len(self._known_databases),
            'known_pages': len(self._known_pages),
            'total_intelligent_operations': sum(pattern['total_interactions'] for pattern in base_stats.values()),
            'average_success_rate': sum(pattern['success_rate'] for pattern in base_stats.values()) / len(base_stats) if base_stats else 0,
            'pattern_breakdown': base_stats
        }
        
        return notion_stats