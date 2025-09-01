# clients/jira_client.py
from __future__ import annotations
import json
import requests
import time
import logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class JiraError(RuntimeError):
    """Enhanced Jira error with more context."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


@dataclass
class JiraConfig:
    """Configuration for Jira client with performance tuning."""
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.3
    pool_connections: int = 10
    pool_maxsize: int = 20
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes


class JiraClient:
    """High-performance Jira client with connection pooling, caching, and retry logic."""

    def __init__(
            self,
            base_url: str,
            email: str,
            api_token: str,
            config: Optional[JiraConfig] = None
    ):
        if not all([base_url, email, api_token]):
            raise JiraError("Missing required Jira credentials")

        self.base = base_url.rstrip("/")
        self.email = email
        self.config = config or JiraConfig()

        # Setup authentication
        self.auth = HTTPBasicAuth(email, api_token)

        # Setup session with connection pooling and retry strategy
        self.session = self._create_optimized_session()

        # Cache for frequently accessed data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"Initialized Jira client for {self.base}")

    def _create_optimized_session(self) -> requests.Session:
        """Create session with optimized connection pooling and retry strategy."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )

        # Configure HTTP adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            max_retries=retry_strategy
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "OptimizedJiraClient/1.0"
        })

        session.auth = self.auth

        return session

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if valid and not expired."""
        if not self.config.enable_caching or cache_key not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self.config.cache_ttl:
            # Cache expired
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp."""
        if self.config.enable_caching:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = time.time()

    def _request(
            self,
            method: str,
            path: str,
            *,
            json_body: Optional[Dict] = None,
            params: Optional[Dict] = None,
            cache_key: Optional[str] = None,
            timeout: Optional[float] = None
    ) -> Any:
        """Enhanced request method with caching, error handling, and performance monitoring."""

        # Check cache for GET requests
        if method == "GET" and cache_key:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data

        url = f"{self.base}{path}"
        request_timeout = timeout or self.config.timeout

        start_time = time.time()

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_body,
                params=params,
                timeout=request_timeout
            )

            execution_time = time.time() - start_time
            logger.debug(f"{method} {path} completed in {execution_time:.3f}s")

            # Handle HTTP errors
            if response.status_code >= 400:
                error_detail = self._parse_error_response(response)
                raise JiraError(
                    f"{method} {path} failed [{response.status_code}]: {error_detail}",
                    status_code=response.status_code,
                    response_data=error_detail
                )

            # Parse response
            try:
                data = response.json()
            except ValueError:
                # Non-JSON response (e.g., 204 No Content)
                data = {"ok": True, "status": response.status_code}

            # Cache successful GET responses
            if method == "GET" and cache_key and data:
                self._set_cache(cache_key, data)

            return data

        except requests.exceptions.Timeout:
            raise JiraError(f"Request to {path} timed out after {request_timeout}s")
        except requests.exceptions.ConnectionError:
            raise JiraError(f"Connection error for {path}")
        except requests.exceptions.RequestException as e:
            raise JiraError(f"Request failed for {path}: {str(e)}")

    def _parse_error_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parse error response with fallback handling."""
        try:
            return response.json()
        except ValueError:
            return {
                "message": response.text or f"HTTP {response.status_code}",
                "status_code": response.status_code
            }

    # ---------------- Enhanced Basic Operations ----------------

    @lru_cache(maxsize=1)
    def whoami(self) -> Dict[str, Any]:
        """Get current user info with caching."""
        return self._request("GET", "/rest/api/3/myself", cache_key="whoami")

    def project_info(self, project_key: str) -> Dict[str, Any]:
        """Get project info with caching."""
        cache_key = f"project_info:{project_key}"
        return self._request("GET", f"/rest/api/3/project/{project_key}", cache_key=cache_key)

    # ---------------- Optimized Issue Type Resolution ----------------

    @lru_cache(maxsize=32)
    def _get_project_issue_types(self, project_key: str) -> List[Dict[str, Any]]:
        """Get project issue types with caching."""
        cache_key = f"issue_types:{project_key}"

        params = {
            "projectKeys": project_key,
            "expand": "projects.issuetypes.fields"
        }

        meta = self._request("GET", "/rest/api/3/issue/createmeta", params=params, cache_key=cache_key)
        projects = meta.get("projects", [])

        if not projects:
            raise JiraError(f"No create metadata found for project {project_key}")

        return projects[0].get("issuetypes", [])

    def _resolve_issue_type_id(self, project_key: str, prefer_name: Optional[str] = None) -> str:
        """Resolve issue type ID with caching and better error handling."""
        issue_types = self._get_project_issue_types(project_key)

        if not issue_types:
            raise JiraError(f"No issue types available for project {project_key}")

        # Look for preferred type name
        if prefer_name:
            prefer_name_lower = prefer_name.lower()
            for issue_type in issue_types:
                type_name = (issue_type.get("name") or "").lower()
                if type_name == prefer_name_lower:
                    return issue_type.get("id")

            logger.warning(f"Issue type '{prefer_name}' not found in {project_key}, using default")

        # Return first available type
        return issue_types[0].get("id")

    # ---------------- Enhanced ADF Content Handling ----------------

    @lru_cache(maxsize=128)
    def _create_adf_paragraph(self, text: str) -> Dict[str, Any]:
        """Create ADF paragraph with caching for repeated content."""
        if not text:
            return {"type": "doc", "version": 1, "content": []}

        return {
            "type": "doc",
            "version": 1,
            "content": [{
                "type": "paragraph",
                "content": [{"type": "text", "text": text}]
            }]
        }

    def _create_adf_content(self, content: Union[str, Dict, List]) -> Dict[str, Any]:
        """Create ADF content with support for various input types."""
        if isinstance(content, str):
            return self._create_adf_paragraph(content)
        elif isinstance(content, dict):
            return content  # Assume it's already ADF
        elif isinstance(content, list):
            # Convert list of strings to paragraphs
            paragraphs = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    paragraphs.append({
                        "type": "paragraph",
                        "content": [{"type": "text", "text": item}]
                    })
            return {"type": "doc", "version": 1, "content": paragraphs}
        else:
            return self._create_adf_paragraph(str(content))

    # ---------------- Enhanced Issue Operations ----------------

    def create_issue(
            self,
            project_key: str,
            summary: str,
            description: Union[str, Dict, List] = "",
            issuetype_id: Optional[str] = None,
            issuetype_name: str = "Task",
            additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create issue with enhanced field support and validation."""

        if not summary.strip():
            raise JiraError("Issue summary cannot be empty")

        # Resolve issue type
        type_id = issuetype_id or self._resolve_issue_type_id(project_key, issuetype_name)

        # Build base fields
        fields = {
            "project": {"key": project_key},
            "summary": summary.strip(),
            "issuetype": {"id": type_id}
        }

        # Add description if provided
        if description:
            fields["description"] = self._create_adf_content(description)

        # Add additional fields if provided
        if additional_fields:
            fields.update(additional_fields)

        payload = {"fields": fields}

        try:
            result = self._request("POST", "/rest/api/3/issue", json_body=payload)
            logger.info(f"Created issue {result.get('key')} in project {project_key}")
            return result
        except JiraError as e:
            logger.error(f"Failed to create issue in {project_key}: {e}")
            raise

    def add_comment(
            self,
            issue_key: str,
            body: Union[str, Dict],
            visibility: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Add comment with enhanced content support."""

        if not body:
            raise JiraError("Comment body cannot be empty")

        # Handle different body types
        if isinstance(body, str):
            comment_body = body
        elif isinstance(body, dict):
            comment_body = body
        else:
            comment_body = str(body)

        payload = {"body": comment_body}

        # Add visibility if specified
        if visibility:
            payload["visibility"] = visibility

        try:
            result = self._request("POST", f"/rest/api/3/issue/{issue_key}/comment", json_body=payload)
            logger.info(f"Added comment to issue {issue_key}")
            return result
        except JiraError as e:
            logger.error(f"Failed to add comment to {issue_key}: {e}")
            raise

    def list_transitions(self, issue_key: str) -> Dict[str, Any]:
        """List issue transitions with caching."""
        cache_key = f"transitions:{issue_key}"
        return self._request("GET", f"/rest/api/3/issue/{issue_key}/transitions", cache_key=cache_key)

    def transition_issue(
            self,
            issue_key: str,
            transition_id: str,
            comment: Optional[str] = None,
            fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Transition issue with optional comment and field updates."""

        payload = {"transition": {"id": str(transition_id)}}

        # Add comment if provided
        if comment:
            payload["update"] = {
                "comment": [{"add": {"body": comment}}]
            }

        # Add field updates if provided
        if fields:
            if "fields" not in payload:
                payload["fields"] = {}
            payload["fields"].update(fields)

        try:
            result = self._request("POST", f"/rest/api/3/issue/{issue_key}/transitions", json_body=payload)
            logger.info(f"Transitioned issue {issue_key} to {transition_id}")

            # Clear transitions cache for this issue
            cache_key = f"transitions:{issue_key}"
            if cache_key in self._cache:
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]

            return result
        except JiraError as e:
            logger.error(f"Failed to transition {issue_key}: {e}")
            raise

    # ---------------- Enhanced Search with Optimization ----------------

    def search(
            self,
            jql: str,
            max_results: int = 50,
            start_at: int = 0,
            fields: Optional[List[str]] = None,
            expand: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhanced search with pagination and field selection."""

        # Validate and clamp max_results
        max_results = max(1, min(max_results, 1000))

        # Default fields for better performance
        if fields is None:
            fields = ["summary", "status", "issuetype", "assignee", "priority", "created", "updated"]

        body = {
            "jql": jql,
            "maxResults": max_results,
            "startAt": start_at,
            "fields": fields
        }

        if expand:
            body["expand"] = expand

        try:
            data = self._request("POST", "/rest/api/3/search", json_body=body)

            # Process and normalize results
            issues = self._process_search_results(data.get("issues", []))

            return {
                "issues": issues,
                "total": data.get("total", 0),
                "startAt": data.get("startAt", 0),
                "maxResults": data.get("maxResults", max_results)
            }

        except JiraError as e:
            logger.error(f"Search failed for JQL '{jql}': {e}")
            raise

    def _process_search_results(self, raw_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize search results for consistent output."""
        processed_issues = []

        for issue in raw_issues:
            fields = issue.get("fields", {})

            processed_issue = {
                "key": issue.get("key"),
                "id": issue.get("id"),
                "summary": fields.get("summary"),
                "status": self._extract_name(fields.get("status")),
                "assignee": self._extract_display_name(fields.get("assignee")),
                "issuetype": self._extract_name(fields.get("issuetype")),
                "priority": self._extract_name(fields.get("priority")),
                "created": fields.get("created"),
                "updated": fields.get("updated"),
                "url": f"{self.base}/browse/{issue.get('key')}"
            }

            processed_issues.append(processed_issue)

        return processed_issues

    def _extract_name(self, obj: Optional[Dict[str, Any]]) -> Optional[str]:
        """Safely extract name from Jira objects."""
        return obj.get("name") if obj else None

    def _extract_display_name(self, obj: Optional[Dict[str, Any]]) -> Optional[str]:
        """Safely extract display name from user objects."""
        return obj.get("displayName") if obj else None

    # ---------------- Batch Operations for Better Performance ----------------

    def create_issues_batch(
            self,
            issues: List[Dict[str, Any]],
            parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Create multiple issues efficiently."""

        if not issues:
            return []

        if parallel and len(issues) > 1:
            return self._create_issues_parallel(issues)
        else:
            return self._create_issues_sequential(issues)

    def _create_issues_sequential(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create issues sequentially."""
        results = []

        for i, issue_data in enumerate(issues):
            try:
                result = self.create_issue(**issue_data)
                results.append({"success": True, "result": result, "index": i})
            except Exception as e:
                results.append({"success": False, "error": str(e), "index": i})

        return results

    def _create_issues_parallel(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create issues in parallel for better performance."""

        def create_single(issue_with_index):
            index, issue_data = issue_with_index
            try:
                result = self.create_issue(**issue_data)
                return {"success": True, "result": result, "index": index}
            except Exception as e:
                return {"success": False, "error": str(e), "index": index}

        indexed_issues = list(enumerate(issues))
        results = list(self._executor.map(create_single, indexed_issues))

        # Sort by original index
        return sorted(results, key=lambda x: x["index"])

    # ---------------- Pagination Helper ----------------

    def search_all(
            self,
            jql: str,
            batch_size: int = 50,
            max_total: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search with automatic pagination to get all results."""

        all_issues = []
        start_at = 0

        while len(all_issues) < max_total:
            remaining = max_total - len(all_issues)
            current_batch_size = min(batch_size, remaining)

            try:
                result = self.search(jql, max_results=current_batch_size, start_at=start_at)
                issues = result.get("issues", [])

                if not issues:
                    break  # No more results

                all_issues.extend(issues)

                # Check if we've got all available results
                if len(issues) < current_batch_size or result.get("total", 0) <= len(all_issues):
                    break

                start_at += len(issues)

            except JiraError as e:
                logger.error(f"Pagination failed at offset {start_at}: {e}")
                break

        return all_issues[:max_total]  # Ensure we don't exceed max_total

    # ---------------- Resource Management ----------------

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Jira client resources cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---------------- Performance and Monitoring ----------------

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._get_project_issue_types.cache_clear()
        self._create_adf_paragraph.cache_clear()
        logger.info("Jira client cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        try:
            start_time = time.time()
            user_info = self.whoami()
            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": response_time,
                "user": user_info.get("displayName"),
                "base_url": self.base
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "base_url": self.base
            }