# clients/jira_client.py
from __future__ import annotations
import base64
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
import urllib3

# Disable SSL warnings for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    verify_ssl: bool = True
    api_version: str = "3"  # Jira Cloud uses API v3


class JiraClient:
    """
    Enhanced Jira client with robust authentication, connection pooling,
    caching, and comprehensive error handling.
    """

    def __init__(
            self,
            base_url: str,
            email: str,
            api_token: str,
            config: Optional[JiraConfig] = None
    ):
        if not all([base_url, email, api_token]):
            raise JiraError("Missing required Jira credentials: base_url, email, and api_token are required")

        # Clean and normalize base URL
        self.base = self._normalize_base_url(base_url)
        self.email = email.strip()
        self.api_token = api_token.strip()
        self.config = config or JiraConfig()

        # Setup authentication with enhanced validation
        self.auth = HTTPBasicAuth(self.email, self.api_token)

        # Setup session with connection pooling and retry strategy
        self.session = self._create_optimized_session()

        # Cache for frequently accessed data
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Test connection immediately
        self._test_connection()

        logger.info(f"Successfully initialized Jira client for {self.base}")

    def _normalize_base_url(self, base_url: str) -> str:
        """Normalize and validate the base URL."""
        url = base_url.strip()

        # Add https:// if missing
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'

        # Remove trailing slash
        url = url.rstrip('/')

        # Validate URL format
        if not (url.startswith('https://') and '.atlassian.net' in url):
            logger.warning(f"URL format may be incorrect: {url}. Expected format: https://yourcompany.atlassian.net")

        return url

    def _create_optimized_session(self) -> requests.Session:
        """Create session with optimized connection pooling and retry strategy."""
        session = requests.Session()

        # Configure retry strategy with exponential backoff
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
            "User-Agent": "EnhancedJiraClient/2.0",
            "X-Atlassian-Token": "no-check"  # Helps with some corporate setups
        })

        # Set authentication
        session.auth = self.auth

        # Configure SSL verification
        session.verify = self.config.verify_ssl

        return session

    def _test_connection(self) -> None:
        """Test the connection and authentication."""
        try:
            # Use a simple endpoint to test connectivity
            test_url = f"{self.base}/rest/api/{self.config.api_version}/myself"

            logger.info(f"Testing Jira connection to: {test_url}")
            logger.info(f"Using email: {self.email}")
            logger.info(f"API token length: {len(self.api_token)} characters")

            response = self.session.get(
                test_url,
                timeout=self.config.timeout
            )

            logger.info(f"Test response status: {response.status_code}")
            logger.info(f"Test response headers: {dict(response.headers)}")

            if response.status_code == 401:
                error_detail = self._parse_error_response(response)
                raise JiraError(
                    f"Authentication failed. Please check your credentials:\n"
                    f"- Base URL: {self.base}\n"
                    f"- Email: {self.email}\n"
                    f"- API Token: {'*' * (len(self.api_token) - 4)}{self.api_token[-4:]}\n"
                    f"Error: {error_detail.get('message', 'Unknown authentication error')}",
                    status_code=401,
                    response_data=error_detail
                )
            elif response.status_code == 404:
                # Try API v2 as fallback for Jira Server
                logger.warning("API v3 not found, trying API v2 (Jira Server)")
                self.config.api_version = "2"
                test_url = f"{self.base}/rest/api/2/myself"
                response = self.session.get(test_url, timeout=self.config.timeout)

            if response.status_code >= 400:
                error_detail = self._parse_error_response(response)
                raise JiraError(
                    f"Connection test failed [{response.status_code}]: {error_detail}",
                    status_code=response.status_code,
                    response_data=error_detail
                )

            # Connection successful
            user_info = response.json()
            logger.info(f"Successfully connected to Jira as: {user_info.get('displayName', 'Unknown User')}")

        except requests.exceptions.SSLError as e:
            logger.warning(f"SSL Error: {e}. Trying with SSL verification disabled...")
            self.config.verify_ssl = False
            self.session.verify = False
            # Retry the test
            self._test_connection()
        except requests.exceptions.ConnectionError as e:
            raise JiraError(
                f"Cannot connect to Jira at {self.base}. Please check:\n"
                f"- URL is correct\n"
                f"- Network connectivity\n"
                f"- Firewall settings\n"
                f"Connection error: {str(e)}"
            )
        except requests.exceptions.Timeout:
            raise JiraError(f"Connection to {self.base} timed out. Please check your network connection.")

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

        # Build full URL
        if not path.startswith('/'):
            path = f'/rest/api/{self.config.api_version}/{path}'
        elif not path.startswith('/rest/'):
            path = f'/rest/api/{self.config.api_version}{path}'

        url = f"{self.base}{path}"
        request_timeout = timeout or self.config.timeout

        start_time = time.time()

        try:
            logger.debug(f"Making {method} request to: {url}")

            response = self.session.request(
                method=method,
                url=url,
                json=json_body,
                params=params,
                timeout=request_timeout
            )

            execution_time = time.time() - start_time
            logger.debug(f"{method} {path} completed in {execution_time:.3f}s")

            # Handle HTTP errors with detailed logging
            if response.status_code >= 400:
                logger.error(f"Request failed: {method} {url}")
                logger.error(f"Status: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")

                error_detail = self._parse_error_response(response)

                # Provide specific error guidance
                error_message = f"{method} {path} failed [{response.status_code}]"
                if response.status_code == 401:
                    error_message += " - Authentication failed. Check your email and API token."
                elif response.status_code == 403:
                    error_message += " - Access forbidden. Check your Jira permissions."
                elif response.status_code == 404:
                    error_message += " - Resource not found. Check the URL and resource existence."
                elif response.status_code == 422:
                    error_message += " - Invalid data. Check required fields and values."

                error_message += f": {error_detail}"

                raise JiraError(
                    error_message,
                    status_code=response.status_code,
                    response_data=error_detail
                )

            # Parse response
            try:
                data = response.json() if response.content else {}
            except ValueError:
                # Non-JSON response (e.g., 204 No Content)
                data = {"ok": True, "status": response.status_code}

            # Cache successful GET responses
            if method == "GET" and cache_key and data:
                self._set_cache(cache_key, data)

            return data

        except requests.exceptions.Timeout:
            raise JiraError(f"Request to {path} timed out after {request_timeout}s")
        except requests.exceptions.ConnectionError as e:
            raise JiraError(f"Connection error for {path}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise JiraError(f"Request failed for {path}: {str(e)}")

    def _parse_error_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parse error response with enhanced debugging and fallback handling."""
        try:
            error_data = response.json()
            logger.debug(f"Parsed error response: {error_data}")
            return error_data
        except ValueError:
            error_text = response.text or f"HTTP {response.status_code}"
            logger.debug(f"Non-JSON error response: {error_text}")
            return {
                "message": error_text,
                "status_code": response.status_code
            }

    # ============================================================================
    # ENHANCED BASIC OPERATIONS
    # ============================================================================

    def whoami(self) -> Dict[str, Any]:
        """Get current user info with caching and enhanced error handling."""
        try:
            return self._request("GET", "/myself", cache_key="whoami")
        except JiraError:
            # Fallback to API v2 if v3 fails
            if self.config.api_version == "3":
                logger.info("Falling back to API v2 for user info")
                self.config.api_version = "2"
                return self._request("GET", "/myself", cache_key="whoami")
            raise

    def project_info(self, project_key: str) -> Dict[str, Any]:
        """Get project info with enhanced validation and caching."""
        if not project_key or not project_key.strip():
            raise JiraError("Project key cannot be empty")

        project_key = project_key.strip().upper()
        cache_key = f"project_info:{project_key}"

        try:
            return self._request("GET", f"/project/{project_key}", cache_key=cache_key)
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Project '{project_key}' not found. Please check the project key.")
            raise

    # ============================================================================
    # ENHANCED ISSUE TYPE RESOLUTION
    # ============================================================================

    @lru_cache(maxsize=32)
    def _get_project_issue_types(self, project_key: str) -> List[Dict[str, Any]]:
        """Get project issue types with enhanced caching and error handling."""
        cache_key = f"issue_types:{project_key}"

        try:
            params = {
                "projectKeys": project_key,
                "expand": "projects.issuetypes.fields"
            }

            meta = self._request("GET", "/issue/createmeta", params=params, cache_key=cache_key)
            projects = meta.get("projects", [])

            if not projects:
                # Try alternative approach for older Jira versions
                logger.info(f"No create metadata found for {project_key}, trying alternative approach")
                project_data = self.project_info(project_key)
                return project_data.get("issueTypes", [])

            return projects[0].get("issuetypes", [])
        except JiraError as e:
            logger.warning(f"Failed to get issue types for {project_key}: {e}")
            return []

    def _resolve_issue_type_id(self, project_key: str, prefer_name: Optional[str] = None) -> str:
        """Resolve issue type ID with enhanced caching and better error handling."""
        issue_types = self._get_project_issue_types(project_key)

        if not issue_types:
            logger.warning(f"No issue types found for project {project_key}, using default 'Task'")
            # Return a default task type ID (commonly "10001" for many Jira instances)
            return "10001"

        # Look for preferred type name
        if prefer_name:
            prefer_name_lower = prefer_name.lower().strip()
            for issue_type in issue_types:
                type_name = (issue_type.get("name") or "").lower()
                if type_name == prefer_name_lower:
                    return issue_type.get("id")

            logger.warning(f"Issue type '{prefer_name}' not found in {project_key}, using default")

        # Return first available type
        return issue_types[0].get("id")

    # ============================================================================
    # ENHANCED ADF CONTENT HANDLING
    # ============================================================================

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
        """Create ADF content with support for various input types and validation."""
        if isinstance(content, str):
            if not content.strip():
                return {"type": "doc", "version": 1, "content": []}
            return self._create_adf_paragraph(content)
        elif isinstance(content, dict):
            # Validate ADF structure
            if content.get("type") == "doc" and "content" in content:
                return content
            # Convert simple dict to ADF
            return self._create_adf_paragraph(str(content))
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

    # ============================================================================
    # ENHANCED ISSUE OPERATIONS
    # ============================================================================

    def create_issue(
            self,
            project_key: str,
            summary: str,
            description: Union[str, Dict, List] = "",
            issuetype_id: Optional[str] = None,
            issuetype_name: str = "Task",
            additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create issue with enhanced field support, validation, and error handling."""

        if not summary or not summary.strip():
            raise JiraError("Issue summary cannot be empty")

        if not project_key or not project_key.strip():
            raise JiraError("Project key cannot be empty")

        project_key = project_key.strip().upper()
        summary = summary.strip()

        # Resolve issue type with fallback
        try:
            type_id = issuetype_id or self._resolve_issue_type_id(project_key, issuetype_name)
        except Exception as e:
            logger.warning(f"Failed to resolve issue type, using default: {e}")
            type_id = "10001"  # Default task type

        # Build base fields
        fields = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"id": str(type_id)}
        }

        # Add description if provided
        if description:
            try:
                fields["description"] = self._create_adf_content(description)
            except Exception as e:
                logger.warning(f"Failed to create ADF content, using plain text: {e}")
                fields["description"] = str(description)

        # Add additional fields if provided
        if additional_fields:
            fields.update(additional_fields)

        payload = {"fields": fields}

        try:
            result = self._request("POST", "/issue", json_body=payload)
            issue_key = result.get('key', 'Unknown')
            logger.info(f"Created issue {issue_key} in project {project_key}")

            # Invalidate cache
            self._invalidate_cache_pattern(f"issues:{project_key}")

            return result
        except JiraError as e:
            logger.error(f"Failed to create issue in {project_key}: {e}")
            # Provide helpful error context
            if "field" in str(e).lower():
                available_types = [t.get("name") for t in self._get_project_issue_types(project_key)]
                error_msg = f"Failed to create issue. Available issue types: {', '.join(available_types)}"
                raise JiraError(error_msg)
            raise

    def add_comment(
            self,
            issue_key: str,
            body: Union[str, Dict],
            visibility: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Add comment with enhanced content support and validation."""

        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        if not body:
            raise JiraError("Comment body cannot be empty")

        issue_key = issue_key.strip().upper()

        # Handle different body types
        if isinstance(body, str):
            comment_body = body.strip()
        elif isinstance(body, dict):
            comment_body = body
        else:
            comment_body = str(body)

        payload = {"body": comment_body}

        # Add visibility if specified
        if visibility:
            payload["visibility"] = visibility

        try:
            result = self._request("POST", f"/issue/{issue_key}/comment", json_body=payload)
            logger.info(f"Added comment to issue {issue_key}")
            return result
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found. Please check the issue key.")
            logger.error(f"Failed to add comment to {issue_key}: {e}")
            raise

    def list_transitions(self, issue_key: str) -> Dict[str, Any]:
        """List issue transitions with enhanced caching and validation."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        issue_key = issue_key.strip().upper()
        cache_key = f"transitions:{issue_key}"

        try:
            return self._request("GET", f"/issue/{issue_key}/transitions", cache_key=cache_key)
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found. Please check the issue key.")
            raise

    def transition_issue(
            self,
            issue_key: str,
            transition_id: str,
            comment: Optional[str] = None,
            fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Transition issue with enhanced validation and field updates."""

        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        if not transition_id:
            raise JiraError("Transition ID cannot be empty")

        issue_key = issue_key.strip().upper()
        transition_id = str(transition_id).strip()

        payload = {"transition": {"id": transition_id}}

        # Add comment if provided
        if comment and comment.strip():
            payload["update"] = {
                "comment": [{"add": {"body": comment.strip()}}]
            }

        # Add field updates if provided
        if fields:
            if "fields" not in payload:
                payload["fields"] = {}
            payload["fields"].update(fields)

        try:
            result = self._request("POST", f"/issue/{issue_key}/transitions", json_body=payload)
            logger.info(f"Transitioned issue {issue_key} using transition {transition_id}")

            # Clear transitions cache for this issue
            cache_key = f"transitions:{issue_key}"
            if cache_key in self._cache:
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]

            return result
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found or transition '{transition_id}' not valid.")
            elif e.status_code == 400:
                # Get available transitions for better error message
                try:
                    transitions = self.list_transitions(issue_key)
                    available = [f"{t.get('name')} (ID: {t.get('id')})"
                                 for t in transitions.get('transitions', [])]
                    raise JiraError(
                        f"Invalid transition '{transition_id}' for issue {issue_key}. "
                        f"Available transitions: {', '.join(available)}"
                    )
                except:
                    raise JiraError(f"Invalid transition '{transition_id}' for issue {issue_key}.")
            logger.error(f"Failed to transition {issue_key}: {e}")
            raise

    # ============================================================================
    # ENHANCED SEARCH WITH OPTIMIZATION
    # ============================================================================

    def search(
            self,
            jql: str,
            max_results: int = 50,
            start_at: int = 0,
            fields: Optional[List[str]] = None,
            expand: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhanced search with pagination, field selection, and validation."""

        if not jql or not jql.strip():
            raise JiraError("JQL query cannot be empty")

        # Validate and clamp max_results
        max_results = max(1, min(max_results, 1000))
        start_at = max(0, start_at)

        # Default fields for better performance
        if fields is None:
            fields = ["summary", "status", "issuetype", "assignee", "priority", "created", "updated"]

        body = {
            "jql": jql.strip(),
            "maxResults": max_results,
            "startAt": start_at,
            "fields": fields
        }

        if expand:
            body["expand"] = expand

        try:
            data = self._request("POST", "/search", json_body=body)

            # Process and normalize results
            issues = self._process_search_results(data.get("issues", []))

            return {
                "issues": issues,
                "total": data.get("total", 0),
                "startAt": data.get("startAt", 0),
                "maxResults": data.get("maxResults", max_results)
            }

        except JiraError as e:
            if "jql" in str(e).lower():
                raise JiraError(f"Invalid JQL query: '{jql}'. Please check your syntax.")
            logger.error(f"Search failed for JQL '{jql}': {e}")
            raise

    def _process_search_results(self, raw_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and normalize search results for consistent output with enhanced field extraction."""
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

            # Add additional useful fields if available
            if "description" in fields:
                processed_issue["description"] = fields["description"]
            if "reporter" in fields:
                processed_issue["reporter"] = self._extract_display_name(fields["reporter"])

            processed_issues.append(processed_issue)

        return processed_issues

    def _extract_name(self, obj: Optional[Dict[str, Any]]) -> Optional[str]:
        """Safely extract name from Jira objects."""
        if not obj:
            return None
        return obj.get("name") if isinstance(obj, dict) else str(obj)

    def _extract_display_name(self, obj: Optional[Dict[str, Any]]) -> Optional[str]:
        """Safely extract display name from user objects."""
        if not obj:
            return None
        if isinstance(obj, dict):
            return obj.get("displayName") or obj.get("name") or obj.get("emailAddress")
        return str(obj)

    # ============================================================================
    # ENHANCED BATCH OPERATIONS
    # ============================================================================

    def create_issues_batch(
            self,
            issues: List[Dict[str, Any]],
            parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Create multiple issues efficiently with enhanced error handling."""

        if not issues:
            return []

        if parallel and len(issues) > 1:
            return self._create_issues_parallel(issues)
        else:
            return self._create_issues_sequential(issues)

    def _create_issues_sequential(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create issues sequentially with detailed tracking."""
        results = []

        for i, issue_data in enumerate(issues):
            try:
                result = self.create_issue(**issue_data)
                results.append({"success": True, "result": result, "index": i})
                logger.info(f"Created issue {i + 1}/{len(issues)}: {result.get('key')}")
            except Exception as e:
                logger.error(f"Failed to create issue {i + 1}/{len(issues)}: {e}")
                results.append({"success": False, "error": str(e), "index": i})

        return results

    def _create_issues_parallel(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create issues in parallel for better performance with error isolation."""

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

    # ============================================================================
    # ENHANCED PAGINATION HELPER
    # ============================================================================

    def search_all(
            self,
            jql: str,
            batch_size: int = 50,
            max_total: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search with automatic pagination to get all results."""

        all_issues = []
        start_at = 0
        batch_size = min(batch_size, 100)  # Jira limit

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

    # ============================================================================
    # ENHANCED UTILITY METHODS
    # ============================================================================

    def _invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching a pattern."""
        keys_to_delete = [key for key in self._cache.keys() if key.startswith(pattern)]
        for key in keys_to_delete:
            del self._cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    def get_issue_statuses(self) -> List[Dict[str, Any]]:
        """Get all available issue statuses."""
        try:
            return self._request("GET", "/status", cache_key="all_statuses")
        except JiraError:
            logger.warning("Failed to get issue statuses")
            return []

    def get_priorities(self) -> List[Dict[str, Any]]:
        """Get all available priorities."""
        try:
            return self._request("GET", "/priority", cache_key="all_priorities")
        except JiraError:
            logger.warning("Failed to get priorities")
            return []

    def get_issue_types(self) -> List[Dict[str, Any]]:
        """Get all available issue types."""
        try:
            return self._request("GET", "/issuetype", cache_key="all_issue_types")
        except JiraError:
            logger.warning("Failed to get issue types")
            return []

    def validate_jql(self, jql: str) -> Dict[str, Any]:
        """Validate JQL syntax."""
        try:
            payload = {"queries": [jql]}
            return self._request("POST", "/jql/parse", json_body=payload)
        except JiraError as e:
            return {"valid": False, "error": str(e)}

    # ============================================================================
    # RESOURCE MANAGEMENT AND MONITORING
    # ============================================================================

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

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys()),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "api_version": self.config.api_version,
            "ssl_verify": self.config.verify_ssl
        }

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._get_project_issue_types.cache_clear()
        self._create_adf_paragraph.cache_clear()
        logger.info("Jira client cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check and return detailed status."""
        try:
            start_time = time.time()
            user_info = self.whoami()
            response_time = time.time() - start_time

            # Test project access if configured
            project_access = None
            if hasattr(self, 'default_project') and self.default_project:
                try:
                    project_info = self.project_info(self.default_project)
                    project_access = project_info.get("name", "Access OK")
                except:
                    project_access = "No access or project not found"

            return {
                "healthy": True,
                "response_time": response_time,
                "user": user_info.get("displayName"),
                "email": user_info.get("emailAddress"),
                "base_url": self.base,
                "api_version": self.config.api_version,
                "ssl_verify": self.config.verify_ssl,
                "project_access": project_access,
                "cache_stats": self.get_cache_stats()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "base_url": self.base,
                "api_version": self.config.api_version,
                "ssl_verify": self.config.verify_ssl
            }

    # ============================================================================
    # ADVANCED SEARCH AND FILTERING HELPERS
    # ============================================================================

    def search_my_issues(self, status: Optional[str] = None, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search for issues assigned to the current user."""
        try:
            user_info = self.whoami()
            user_key = user_info.get("accountId") or user_info.get("key") or user_info.get("name")

            if not user_key:
                raise JiraError("Could not determine current user identifier")

            jql = f"assignee = '{user_key}'"
            if status:
                jql += f" AND status = '{status}'"
            jql += " ORDER BY created DESC"

            result = self.search(jql, max_results=max_results)
            return result.get("issues", [])
        except Exception as e:
            logger.error(f"Failed to search user's issues: {e}")
            return []

    def search_project_issues(
            self,
            project_key: str,
            status: Optional[str] = None,
            issue_type: Optional[str] = None,
            max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for issues in a specific project with optional filters."""
        jql_parts = [f"project = '{project_key}'"]

        if status:
            jql_parts.append(f"status = '{status}'")
        if issue_type:
            jql_parts.append(f"issuetype = '{issue_type}'")

        jql = " AND ".join(jql_parts) + " ORDER BY created DESC"

        try:
            result = self.search(jql, max_results=max_results)
            return result.get("issues", [])
        except Exception as e:
            logger.error(f"Failed to search project issues: {e}")
            return []

    def get_recent_activity(self, days: int = 7, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get recently updated issues."""
        jql = f"updated >= -{days}d ORDER BY updated DESC"

        try:
            result = self.search(jql, max_results=max_results)
            return result.get("issues", [])
        except Exception as e:
            logger.error(f"Failed to get recent activity: {e}")
            return []

    # ============================================================================
    # TRANSITION HELPERS
    # ============================================================================

    def find_transition_by_name(self, issue_key: str, transition_name: str) -> Optional[str]:
        """Find transition ID by name with fuzzy matching."""
        try:
            transitions_data = self.list_transitions(issue_key)
            transitions = transitions_data.get("transitions", [])

            transition_name_lower = transition_name.lower().strip()

            # Remove common prefixes
            if transition_name_lower.startswith(':'):
                transition_name_lower = transition_name_lower[1:]

            # Exact match first
            for transition in transitions:
                if transition.get("name", "").lower() == transition_name_lower:
                    return transition.get("id")

            # Partial match
            for transition in transitions:
                trans_name = transition.get("name", "").lower()
                if (transition_name_lower in trans_name or
                        trans_name in transition_name_lower):
                    return transition.get("id")

            # Pattern matching for common status names
            patterns = {
                'done': ['done', 'close', 'resolve', 'complete', 'finish'],
                'in_progress': ['progress', 'start', 'begin', 'doing', 'active'],
                'open': ['open', 'reopen', 'todo', 'backlog']
            }

            for pattern_key, pattern_values in patterns.items():
                if transition_name_lower == pattern_key or transition_name_lower in pattern_values:
                    for transition in transitions:
                        trans_name = transition.get("name", "").lower()
                        for pattern_val in pattern_values:
                            if pattern_val in trans_name:
                                return transition.get("id")

            return None

        except Exception as e:
            logger.error(f"Failed to find transition '{transition_name}' for {issue_key}: {e}")
            return None

    def transition_to_status(self, issue_key: str, target_status: str, comment: Optional[str] = None) -> Dict[str, Any]:
        """Transition issue to a target status by name."""
        transition_id = self.find_transition_by_name(issue_key, target_status)

        if not transition_id:
            # Get available transitions for error message
            try:
                transitions_data = self.list_transitions(issue_key)
                available = [t.get("name") for t in transitions_data.get("transitions", [])]
                raise JiraError(
                    f"No transition found to move {issue_key} to '{target_status}'. "
                    f"Available transitions: {', '.join(available)}"
                )
            except JiraError:
                raise
            except Exception:
                raise JiraError(f"Could not find transition to '{target_status}' for {issue_key}")

        return self.transition_issue(issue_key, transition_id, comment)

    # ============================================================================
    # MISSING JIRA OPERATIONS - IMPLEMENTATION
    # ============================================================================

    def update_issue(
            self,
            issue_key: str,
            summary: Optional[str] = None,
            description: Optional[Union[str, Dict, List]] = None,
            assignee: Optional[str] = None,
            priority: Optional[str] = None,
            labels: Optional[List[str]] = None,
            additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an existing issue with enhanced field support."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        issue_key = issue_key.strip()
        fields = {}

        # Update summary
        if summary is not None and summary.strip():
            fields["summary"] = summary.strip()

        # Update description
        if description is not None:
            if isinstance(description, str) and description.strip():
                fields["description"] = self._create_adf_content(description)
            elif isinstance(description, (dict, list)):
                fields["description"] = self._create_adf_content(description)

        # Update assignee
        if assignee is not None:
            if assignee.strip():
                fields["assignee"] = {"accountId": assignee} if "@" not in assignee else {"emailAddress": assignee}
            else:
                fields["assignee"] = None  # Unassign

        # Update priority
        if priority is not None and priority.strip():
            # Try to find priority by name
            try:
                fields["priority"] = {"name": priority.strip()}
            except Exception as e:
                logger.warning(f"Could not set priority '{priority}': {e}")

        # Update labels
        if labels is not None:
            fields["labels"] = [label.strip() for label in labels if label and label.strip()]

        # Add any additional fields
        if additional_fields:
            fields.update(additional_fields)

        if not fields:
            raise JiraError("No fields provided for update")

        try:
            self._request("PUT", f"/issue/{issue_key}", json={"fields": fields})
            
            # Return updated issue data
            return self._request("GET", f"/issue/{issue_key}")
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found")
            raise JiraError(f"Failed to update issue '{issue_key}': {str(e)}")

    def delete_issue(self, issue_key: str, delete_subtasks: bool = False) -> Dict[str, Any]:
        """Delete an issue with optional subtask deletion."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        issue_key = issue_key.strip()
        
        try:
            params = {}
            if delete_subtasks:
                params["deleteSubtasks"] = "true"
            
            self._request("DELETE", f"/issue/{issue_key}", params=params)
            
            return {"success": True, "message": f"Issue {issue_key} deleted successfully"}
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found")
            elif e.status_code == 403:
                raise JiraError(f"Permission denied: Cannot delete issue '{issue_key}'")
            raise JiraError(f"Failed to delete issue '{issue_key}': {str(e)}")

    def set_issue_labels(self, issue_key: str, labels: List[str]) -> Dict[str, Any]:
        """Set labels for an issue (replaces existing labels)."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        cleaned_labels = [label.strip() for label in labels if label and label.strip()]
        
        return self.update_issue(issue_key, labels=cleaned_labels)

    def add_issue_labels(self, issue_key: str, labels: List[str]) -> Dict[str, Any]:
        """Add labels to an issue (keeps existing labels)."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        issue_key = issue_key.strip()
        
        try:
            # Get current issue to retrieve existing labels
            current_issue = self._request("GET", f"/issue/{issue_key}?fields=labels")
            current_labels = [label for label in current_issue.get("fields", {}).get("labels", [])]
            
            # Combine existing and new labels
            new_labels = [label.strip() for label in labels if label and label.strip()]
            all_labels = list(set(current_labels + new_labels))
            
            return self.update_issue(issue_key, labels=all_labels)
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found")
            raise

    def remove_issue_labels(self, issue_key: str, labels: List[str]) -> Dict[str, Any]:
        """Remove specific labels from an issue."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        issue_key = issue_key.strip()
        
        try:
            # Get current issue to retrieve existing labels
            current_issue = self._request("GET", f"/issue/{issue_key}?fields=labels")
            current_labels = [label for label in current_issue.get("fields", {}).get("labels", [])]
            
            # Remove specified labels
            labels_to_remove = [label.strip().lower() for label in labels if label and label.strip()]
            remaining_labels = [label for label in current_labels 
                              if label.lower() not in labels_to_remove]
            
            return self.update_issue(issue_key, labels=remaining_labels)
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found")
            raise

    def set_issue_priority(self, issue_key: str, priority: str) -> Dict[str, Any]:
        """Set priority for an issue."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")
        
        if not priority or not priority.strip():
            raise JiraError("Priority cannot be empty")

        return self.update_issue(issue_key, priority=priority.strip())

    def assign_issue(self, issue_key: str, assignee: str) -> Dict[str, Any]:
        """Assign issue to a user."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")
        
        if not assignee or not assignee.strip():
            raise JiraError("Assignee cannot be empty")

        return self.update_issue(issue_key, assignee=assignee.strip())

    def unassign_issue(self, issue_key: str) -> Dict[str, Any]:
        """Unassign issue (remove assignee)."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        return self.update_issue(issue_key, assignee="")

    def get_issue(self, issue_key: str, expand_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get detailed information about a specific issue."""
        if not issue_key or not issue_key.strip():
            raise JiraError("Issue key cannot be empty")

        issue_key = issue_key.strip()
        
        try:
            params = {}
            if expand_fields:
                params["expand"] = ",".join(expand_fields)
            
            issue = self._request("GET", f"/issue/{issue_key}", params=params)
            
            # Return processed issue for consistency
            return self._process_search_results([issue])[0] if issue else {}
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Issue '{issue_key}' not found")
            raise JiraError(f"Failed to get issue '{issue_key}': {str(e)}")

    def list_projects(self, expand: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all accessible projects."""
        try:
            params = {}
            if expand:
                params["expand"] = ",".join(expand)
            
            projects = self._request("GET", "/project", params=params, cache_key="projects")
            
            # Process projects for consistent format
            processed_projects = []
            for project in projects:
                processed_project = {
                    "key": project.get("key"),
                    "id": project.get("id"),
                    "name": project.get("name"),
                    "description": project.get("description"),
                    "lead": self._extract_display_name(project.get("lead")),
                    "projectTypeKey": project.get("projectTypeKey"),
                    "url": f"{self.base}/browse/{project.get('key')}" if project.get("key") else None
                }
                
                # Add additional fields if available
                if "issueTypes" in project:
                    processed_project["issueTypes"] = project["issueTypes"]
                if "roles" in project:
                    processed_project["roles"] = project["roles"]
                    
                processed_projects.append(processed_project)
            
            return processed_projects
        except JiraError as e:
            raise JiraError(f"Failed to list projects: {str(e)}")

    def get_project_details(self, project_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific project."""
        if not project_key or not project_key.strip():
            raise JiraError("Project key cannot be empty")

        project_key = project_key.strip().upper()
        
        try:
            project = self._request("GET", f"/project/{project_key}", cache_key=f"project_details:{project_key}")
            
            # Process project for display
            processed_project = {
                "key": project.get("key"),
                "id": project.get("id"),
                "name": project.get("name"),
                "description": project.get("description"),
                "lead": self._extract_display_name(project.get("lead")),
                "projectTypeKey": project.get("projectTypeKey"),
                "url": f"{self.base}/browse/{project.get('key')}" if project.get("key") else None,
                "issueTypes": project.get("issueTypes", []),
                "roles": project.get("roles", {}),
                "components": project.get("components", []),
                "versions": project.get("versions", [])
            }
            
            return processed_project
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Project '{project_key}' not found")
            raise JiraError(f"Failed to get project details for '{project_key}': {str(e)}")

    def update_project(self, project_key: str, **kwargs) -> Dict[str, Any]:
        """Update project settings (limited based on permissions)."""
        if not project_key or not project_key.strip():
            raise JiraError("Project key cannot be empty")

        project_key = project_key.strip().upper()
        
        # Build update payload
        update_data = {}
        if "name" in kwargs and kwargs["name"]:
            update_data["name"] = kwargs["name"].strip()
        if "description" in kwargs:
            update_data["description"] = kwargs["description"]
        if "lead" in kwargs and kwargs["lead"]:
            update_data["lead"] = kwargs["lead"].strip()

        if not update_data:
            raise JiraError("No valid update fields provided")

        try:
            self._request("PUT", f"/project/{project_key}", json=update_data)
            
            # Return updated project
            return self.get_project_details(project_key)
        except JiraError as e:
            if e.status_code == 404:
                raise JiraError(f"Project '{project_key}' not found")
            elif e.status_code == 403:
                raise JiraError(f"Permission denied: Cannot update project '{project_key}'")
            raise JiraError(f"Failed to update project '{project_key}': {str(e)}")