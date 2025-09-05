# clients/github_client.py
from __future__ import annotations
import base64
import json
import requests
import time
import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timezone
import random

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubError(RuntimeError):
    """Enhanced GitHub error with additional context."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


@dataclass
class GitHubConfig:
    """Configuration for GitHub client performance tuning."""
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.3
    pool_connections: int = 15
    pool_maxsize: int = 25
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_per_page: int = 100
    enable_rate_limit_handling: bool = True
    read_only_mode: bool = True
    confirm_destructive: bool = True


def normalize_repo_input(repo_input: str, default_owner: str = "", default_repo: str = "") -> Tuple[str, str]:
    """
    Normalize repository input to (owner, repo) tuple.

    Handles:
    - "owner/repo" format
    - GitHub URLs (https://github.com/owner/repo)
    - Just repo name (uses default_owner)
    """
    if not repo_input:
        return default_owner, default_repo

    # GitHub URL pattern
    url_match = re.match(r'https://github\.com/([^/]+)/([^/]+)/?.*', repo_input)
    if url_match:
        return url_match.group(1), url_match.group(2)

    # owner/repo pattern
    if '/' in repo_input:
        parts = repo_input.split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]

    # Just repo name
    return default_owner, repo_input


def normalize_issue_pr_input(input_str: str) -> int:
    """
    Normalize issue/PR input to issue number.

    Handles:
    - "#123" format
    - "123" number
    - GitHub URLs with issue/PR numbers
    """
    if not input_str:
        raise ValueError("Issue/PR input cannot be empty")

    # Remove # prefix
    if input_str.startswith('#'):
        input_str = input_str[1:]

    # Extract from GitHub URLs
    url_patterns = [
        r'github\.com/[^/]+/[^/]+/issues/(\d+)',
        r'github\.com/[^/]+/[^/]+/pull/(\d+)'
    ]

    for pattern in url_patterns:
        match = re.search(pattern, input_str)
        if match:
            return int(match.group(1))

    # Direct number
    try:
        return int(input_str)
    except ValueError:
        raise ValueError(f"Cannot parse issue/PR number from: {input_str}")


def normalize_workflow_input(input_str: str) -> Union[str, int]:
    """
    Normalize workflow input - can be filename or ID.
    Returns as-is for filename, int for ID.
    """
    if not input_str:
        raise ValueError("Workflow input cannot be empty")

    # Try to parse as int (workflow ID)
    try:
        return int(input_str)
    except ValueError:
        # Return as filename
        return input_str


class GitHubClient:
    """Comprehensive GitHub client with full API surface coverage."""

    # Argument allowlists for each operation
    ARG_ALLOWLISTS = {
        # Repository operations
        'create_repository': {'name', 'description', 'private', 'has_issues', 'has_projects', 'has_wiki', 'auto_init',
                              'gitignore_template', 'license_template'},
        'list_repositories': {'visibility', 'affiliation', 'type', 'sort', 'direction', 'per_page', 'page'},
        'list_repository_contents': {'path', 'ref'},
        'fork_repository': {'organization'},
        'create_branch': {'branch', 'from_branch'},
        'list_branches': {'protected', 'per_page', 'page'},
        'list_commits': {'sha', 'path', 'author', 'since', 'until', 'per_page', 'page'},
        'get_commit': {'ref'},
        'create_or_update_file': {'path', 'content', 'message', 'branch', 'sha', 'committer', 'author'},
        'delete_file': {'path', 'message', 'branch', 'sha', 'committer', 'author'},
        'get_file_contents': {'path', 'ref'},
        'push_files': {'branch', 'files', 'message', 'committer', 'author'},
        'list_tags': {'per_page', 'page'},
        'get_tag': {'tag'},
        'list_releases': {'per_page', 'page'},
        'get_release_by_tag': {'tag'},

        # Issue operations
        'create_issue': {'title', 'body', 'assignees', 'milestone', 'labels', 'assignee'},
        'update_issue': {'title', 'body', 'assignees', 'milestone', 'labels', 'state', 'assignee'},
        'get_issue': {},
        'list_issues': {'milestone', 'state', 'assignee', 'creator', 'mentioned', 'labels', 'sort', 'direction',
                        'since', 'per_page', 'page'},
        'search_issues': {'q', 'sort', 'order', 'per_page', 'page'},
        'add_issue_comment': {'body'},

        # Pull Request operations
        'create_pull_request': {'title', 'body', 'head', 'base', 'draft', 'maintainer_can_modify'},
        'update_pull_request': {'title', 'body', 'state', 'base', 'maintainer_can_modify'},
        'merge_pull_request': {'commit_title', 'commit_message', 'merge_method', 'sha'},
        'list_pull_requests': {'state', 'head', 'base', 'sort', 'direction', 'per_page', 'page'},
        'get_pull_request': {},
        'get_pull_request_files': {'per_page', 'page'},
        'get_pull_request_reviews': {'per_page', 'page'},
        'get_pull_request_comments': {'sort', 'direction', 'since', 'per_page', 'page'},
        'search_pull_requests': {'q', 'sort', 'order', 'per_page', 'page'},

        # Workflow operations
        'list_workflows': {'per_page', 'page'},
        'list_workflow_runs': {'actor', 'branch', 'event', 'status', 'per_page', 'page', 'created',
                               'exclude_pull_requests', 'check_suite_id'},
        'get_workflow_run': {},
        'run_workflow': {'ref', 'inputs'},
        'rerun_workflow_run': {},
        'cancel_workflow_run': {},
        'list_workflow_jobs': {'filter', 'per_page', 'page'},

        # Search operations
        'search_code': {'q', 'sort', 'order', 'per_page', 'page'},

        # Notification operations
        'list_notifications': {'all', 'participating', 'since', 'before', 'per_page', 'page'},
        'mark_notifications_read': {'last_read_at'},

        # User/Org operations
        'search_users': {'q', 'sort', 'order', 'per_page', 'page'},
        'search_orgs': {'q', 'sort', 'order', 'per_page', 'page'},

        # Discussion operations
        'list_discussions': {'category_id', 'labels', 'per_page', 'page'},
        'get_discussion': {},
        'list_discussion_comments': {'per_page', 'page'},

        # Gist operations
        'create_gist': {'files', 'description', 'public'},
        'list_gists': {'since', 'per_page', 'page'},
        'update_gist': {'files', 'description'},
    }

    # Destructive operations that require confirmation
    DESTRUCTIVE_OPERATIONS = {
        'merge_pull_request', 'delete_file', 'push_files', 'cancel_workflow_run',
        'remove_sub_issue', 'reprioritize_sub_issue'
    }

    def __init__(self, token: str, config: Optional[GitHubConfig] = None, default_owner: str = "",
                 default_repo: str = ""):
        if not token:
            raise GitHubError("GitHub token is required")

        self._token = token
        self.config = config or GitHubConfig()
        self.default_owner = default_owner
        self.default_repo = default_repo

        # Setup optimized session
        self.session = self._create_optimized_session()

        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Rate limiting tracking
        self._rate_limit_remaining = 5000
        self._rate_limit_reset_time = 0
        self._rate_limit_lock = threading.RLock()

        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=6)

        logger.info("Initialized comprehensive GitHub client")

    def _create_optimized_session(self) -> requests.Session:
        """Create session with connection pooling and retry strategy."""
        session = requests.Session()

        # Configure retry strategy with jitter
        try:
            # Try with backoff_jitter if supported
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
                backoff_jitter=0.3
            )
        except TypeError:
            # Fallback without jitter parameter
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
            )

        # Configure HTTP adapter
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            max_retries=retry_strategy
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with optimizations."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "EnhancedGitHubClient/2.0"
        }

    def _clean_args(self, operation: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Clean arguments based on operation allowlist."""
        if operation not in self.ARG_ALLOWLISTS:
            logger.warning(f"No allowlist defined for operation: {operation}")
            return args

        allowed = self.ARG_ALLOWLISTS[operation]
        cleaned = {k: v for k, v in args.items() if k in allowed}

        dropped = set(args.keys()) - set(cleaned.keys())
        if dropped:
            logger.debug(f"Dropped invalid args for {operation}: {dropped}")

        return cleaned

    def _check_destructive_operation(self, operation: str, args: Dict[str, Any]) -> bool:
        """Check if operation is destructive and handle confirmation."""
        if operation not in self.DESTRUCTIVE_OPERATIONS:
            return True

        if self.config.read_only_mode:
            logger.info(f"DRY RUN - Would execute {operation} with args: {args}")
            return False

        if self.config.confirm_destructive:
            logger.warning(f"DESTRUCTIVE OPERATION: {operation} with args: {args}")
            # In a real implementation, this would prompt for confirmation
            # For now, we'll log and proceed (you'd implement UI confirmation)
            return True

        return True

    def _resolve_workflow_id(self, owner: str, repo: str, workflow_input: Union[str, int]) -> int:
        """Resolve workflow filename to ID if needed."""
        if isinstance(workflow_input, int):
            return workflow_input

        # It's a filename, need to resolve to ID
        workflows = self.list_workflows(owner, repo)
        if isinstance(workflows, dict) and 'workflows' in workflows:
            for workflow in workflows['workflows']:
                if workflow.get('path', '').endswith(workflow_input) or workflow.get('name') == workflow_input:
                    return workflow['id']

        raise GitHubError(f"Could not resolve workflow '{workflow_input}' to ID")

    def _invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching a pattern."""
        with self._cache_lock:
            keys_to_delete = [key for key in self._cache.keys() if key.startswith(pattern)]
            for key in keys_to_delete:
                del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]

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
        """Enhanced request method with caching, rate limiting, and error handling."""

        # Check cache for GET requests
        if method == "GET" and cache_key:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data

        # Check rate limits
        self._check_rate_limit()

        url = f"{GITHUB_API}{path}"
        request_timeout = timeout or self.config.timeout

        start_time = time.time()

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                json=json_body,
                params=params,
                timeout=request_timeout
            )

            execution_time = time.time() - start_time
            logger.debug(f"{method} {path} completed in {execution_time:.3f}s")

            # Update rate limit info
            self._update_rate_limit_info(response)

            # Handle HTTP errors
            if response.status_code >= 400:
                error_detail = self._parse_error_response(response)
                raise GitHubError(
                    f"{method} {path} failed [{response.status_code}]: {error_detail}",
                    status_code=response.status_code,
                    response_data=error_detail
                )

            # Parse response
            try:
                data = response.json()
            except ValueError:
                data = {"ok": True, "status": response.status_code}

            # Cache successful GET responses
            if method == "GET" and cache_key and data:
                self._set_cache(cache_key, data)

            return data

        except requests.exceptions.Timeout:
            raise GitHubError(f"Request to {path} timed out after {request_timeout}s")
        except requests.exceptions.ConnectionError:
            raise GitHubError(f"Connection error for {path}")
        except requests.exceptions.RequestException as e:
            raise GitHubError(f"Request failed for {path}: {str(e)}")

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Thread-safe cache retrieval with TTL checking."""
        if not self.config.enable_caching:
            return None

        with self._cache_lock:
            if cache_key not in self._cache:
                return None

            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp > self.config.cache_ttl:
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None

            return self._cache[cache_key]

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Thread-safe cache storage."""
        if self.config.enable_caching:
            with self._cache_lock:
                self._cache[cache_key] = data
                self._cache_timestamps[cache_key] = time.time()

    def _update_rate_limit_info(self, response: requests.Response) -> None:
        """Update rate limit information from response headers."""
        if not self.config.enable_rate_limit_handling:
            return

        with self._rate_limit_lock:
            self._rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 5000))
            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                self._rate_limit_reset_time = int(reset_time)

    def _check_rate_limit(self) -> None:
        """Check and handle rate limiting with jitter."""
        if not self.config.enable_rate_limit_handling:
            return

        with self._rate_limit_lock:
            if self._rate_limit_remaining < 10:
                current_time = time.time()
                if current_time < self._rate_limit_reset_time:
                    # Add jitter to prevent thundering herd
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = self._rate_limit_reset_time - current_time + 1 + jitter
                    logger.warning(f"Rate limit low, sleeping for {sleep_time:.1f}s")
                    time.sleep(sleep_time)

    def _parse_error_response(self, response: requests.Response) -> Dict[str, Any]:
        """Parse error response with fallback handling and actionable messages."""
        try:
            error_data = response.json()
            message = error_data.get('message', f"HTTP {response.status_code}")

            # Add actionable suggestions
            if response.status_code == 401:
                message += " (Check your GitHub token and permissions)"
            elif response.status_code == 403:
                message += " (Check token scopes and rate limits)"
            elif response.status_code == 404:
                message += " (Resource not found - check owner/repo names)"
            elif response.status_code == 422:
                message += " (Validation failed - check required fields)"

            return {"message": message, "status_code": response.status_code}
        except ValueError:
            return {
                "message": f"HTTP {response.status_code}: {response.text or 'Unknown error'}",
                "status_code": response.status_code
            }

    # ============================================================================
    # REPOSITORY OPERATIONS
    # ============================================================================

    def get_user(self) -> Dict[str, Any]:
        """Get current user info with caching."""
        return self._request("GET", "/user", cache_key="current_user")
    
    def list_repositories(self, **kwargs) -> Dict[str, Any]:
        """List repositories for the authenticated user."""
        cleaned_args = self._clean_args('list_repositories', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        
        # GitHub API restriction: if visibility or affiliation is specified, type cannot be used
        has_visibility_or_affiliation = any(key in cleaned_args for key in ['visibility', 'affiliation'])
        
        for k, v in cleaned_args.items():
            if k != "per_page":
                # Skip 'type' parameter if visibility or affiliation is already specified
                if k == 'type' and has_visibility_or_affiliation:
                    continue
                params[k] = v
        
        return self._request("GET", "/user/repos", params=params, cache_key=f"user_repos:{hash(str(params))}")
    
    def list_user_repositories(self, username: str, **kwargs) -> Dict[str, Any]:
        """List public repositories for a specific user."""
        cleaned_args = self._clean_args('list_repositories', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})
        
        return self._request("GET", f"/users/{username}/repos", params=params, cache_key=f"repos:{username}:{hash(str(params))}")
    
    def list_organization_repositories(self, org: str, **kwargs) -> Dict[str, Any]:
        """List repositories for an organization."""
        cleaned_args = self._clean_args('list_repositories', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})
        
        return self._request("GET", f"/orgs/{org}/repos", params=params, cache_key=f"org_repos:{org}:{hash(str(params))}")
        

    def get_repository(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get repository information with caching."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cache_key = f"repo:{owner}/{repo}"
        return self._request("GET", f"/repos/{owner}/{repo}", cache_key=cache_key)

    def create_repository(self, name: str, **kwargs) -> Dict[str, Any]:
        """Create a new repository."""
        if not self._check_destructive_operation('create_repository', {'name': name, **kwargs}):
            return {"dry_run": True, "operation": "create_repository", "args": {"name": name, **kwargs}}

        cleaned_args = self._clean_args('create_repository', {'name': name, **kwargs})
        return self._request("POST", "/user/repos", json_body=cleaned_args)

    def fork_repository(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Fork a repository."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('fork_repository', kwargs)
        return self._request("POST", f"/repos/{owner}/{repo}/forks", json_body=cleaned_args)

    def create_branch(self, branch: str, owner: str = "", repo: str = "", from_branch: str = "main", **kwargs) -> Dict[
        str, Any]:
        """Create a new branch."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not self._check_destructive_operation('create_branch', {'branch': branch, 'from_branch': from_branch}):
            return {"dry_run": True, "operation": "create_branch",
                    "args": {"branch": branch, "from_branch": from_branch}}

        # Get SHA of from_branch
        ref_data = self._request("GET", f"/repos/{owner}/{repo}/git/ref/heads/{from_branch}")
        sha = ref_data.get("object", {}).get("sha")
        if not sha:
            raise GitHubError(f"Could not get SHA for branch '{from_branch}'")

        payload = {"ref": f"refs/heads/{branch}", "sha": sha}
        return self._request("POST", f"/repos/{owner}/{repo}/git/refs", json_body=payload)

    def list_branches(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List repository branches."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_branches', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", f"/repos/{owner}/{repo}/branches", params=params,
                             cache_key=f"branches:{owner}/{repo}:{hash(str(params))}")

    def list_commits(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List repository commits."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_commits', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", f"/repos/{owner}/{repo}/commits", params=params)

    def get_commit(self, ref: str, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Get a specific commit."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('get_commit', {'ref': ref, **kwargs})
        return self._request("GET", f"/repos/{owner}/{repo}/commits/{ref}")

    def create_or_update_file(self, path: str, content: Union[str, bytes], message: str,
                              owner: str = "", repo: str = "", branch: str = "main", **kwargs) -> Dict[str, Any]:
        """Create or update a file."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not self._check_destructive_operation('create_or_update_file', {'path': path, 'message': message}):
            return {"dry_run": True, "operation": "create_or_update_file", "args": {"path": path, "message": message}}

        # Check if file exists to get SHA
        sha = None
        try:
            existing_file = self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params={"ref": branch})
            sha = existing_file.get("sha")
        except GitHubError as e:
            if e.status_code != 404:
                raise

        # Prepare content
        if isinstance(content, str):
            content_b64 = base64.b64encode(content.encode('utf-8')).decode('ascii')
        else:
            content_b64 = base64.b64encode(content).decode('ascii')

        payload = {
            "message": message.strip(),
            "content": content_b64,
            "branch": branch
        }

        if sha:
            payload["sha"] = sha

        # Add additional args
        cleaned_args = self._clean_args('create_or_update_file', kwargs)
        payload.update(cleaned_args)

        return self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", json_body=payload)

    def delete_file(self, path: str, message: str, owner: str = "", repo: str = "", branch: str = "main", **kwargs) -> \
    Dict[str, Any]:
        """Delete a file."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not self._check_destructive_operation('delete_file', {'path': path, 'message': message}):
            return {"dry_run": True, "operation": "delete_file", "args": {"path": path, "message": message}}

        # Get file SHA
        try:
            existing_file = self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params={"ref": branch})
            sha = existing_file.get("sha")
            if not sha:
                raise GitHubError(f"Could not get SHA for file '{path}'")
        except GitHubError as e:
            if e.status_code == 404:
                raise GitHubError(f"File '{path}' not found")
            raise

        payload = {
            "message": message.strip(),
            "sha": sha,
            "branch": branch
        }

        cleaned_args = self._clean_args('delete_file', kwargs)
        payload.update(cleaned_args)

        return self._request("DELETE", f"/repos/{owner}/{repo}/contents/{path}", json_body=payload)

    def get_file_contents(self, path: str, owner: str = "", repo: str = "", ref: str = "main", **kwargs) -> Dict[
        str, Any]:
        """Get file contents."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('get_file_contents', {'path': path, 'ref': ref, **kwargs})
        params = {"ref": ref} if ref else {}

        return self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)
    
    def list_repository_contents(self, path: str = "", owner: str = "", repo: str = "", ref: str = "main", **kwargs) -> Dict[str, Any]:
        """List repository contents (files and directories)."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_repository_contents', {'path': path, 'ref': ref, **kwargs})
        params = {"ref": ref} if ref else {}

        return self._request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)

    def push_files(self, branch: str, files: List[Dict[str, str]], message: str,
                   owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Push multiple files to a branch."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not files:
            raise GitHubError("Files list cannot be empty")

        if not self._check_destructive_operation('push_files', {'branch': branch, 'files': files, 'message': message}):
            return {"dry_run": True, "operation": "push_files",
                    "args": {"branch": branch, "files": files, "message": message}}

        results = []
        updated_paths = []
        commit_shas = []

        for file_info in files:
            path = file_info.get('path')
            content = file_info.get('content')

            if not path or content is None:
                results.append({"path": path, "error": "Missing path or content"})
                continue

            try:
                result = self.create_or_update_file(
                    path=path,
                    content=content,
                    message=message,
                    owner=owner,
                    repo=repo,
                    branch=branch,
                    **kwargs
                )

                updated_paths.append(path)
                if result.get('commit', {}).get('sha'):
                    commit_shas.append(result['commit']['sha'])

                results.append({"path": path, "success": True, "sha": result.get('content', {}).get('sha')})

            except Exception as e:
                results.append({"path": path, "error": str(e)})

        return {
            "ok": True,
            "branch": branch,
            "updated_paths": updated_paths,
            "commit_shas": list(set(commit_shas)),  # Unique SHAs
            "results": results,
            "summary": f"Updated {len(updated_paths)}/{len(files)} files on {branch}"
        }

    def list_tags(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List repository tags."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_tags', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}

        return self._request("GET", f"/repos/{owner}/{repo}/tags", params=params)

    def get_tag(self, tag: str, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Get a specific tag."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/git/refs/tags/{tag}")

    def list_releases(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List repository releases."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_releases', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}

        return self._request("GET", f"/repos/{owner}/{repo}/releases", params=params)

    def get_latest_release(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get latest release."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/releases/latest")

    def get_release_by_tag(self, tag: str, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Get release by tag."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/releases/tags/{tag}")

    # ============================================================================
    # ISSUE OPERATIONS
    # ============================================================================

    def create_issue(self, title: str, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Create an issue with enhanced validation."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not title.strip():
            raise GitHubError("Issue title cannot be empty")

        cleaned_args = self._clean_args('create_issue', {'title': title.strip(), **kwargs})

        result = self._request("POST", f"/repos/{owner}/{repo}/issues", json_body=cleaned_args)
        logger.info(f"Created issue #{result.get('number')} in {owner}/{repo}")

        # Clear cache
        self._invalidate_cache_pattern(f"issues:{owner}/{repo}")
        return result

    def update_issue(self, issue_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Update an existing issue."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        issue_num = normalize_issue_pr_input(str(issue_number))
        cleaned_args = self._clean_args('update_issue', kwargs)

        result = self._request("PATCH", f"/repos/{owner}/{repo}/issues/{issue_num}", json_body=cleaned_args)
        self._invalidate_cache_pattern(f"issues:{owner}/{repo}")
        return result

    def get_issue(self, issue_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Get a specific issue."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        issue_num = normalize_issue_pr_input(str(issue_number))
        return self._request("GET", f"/repos/{owner}/{repo}/issues/{issue_num}",
                             cache_key=f"issue:{owner}/{repo}:{issue_num}")

    def list_issues(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List issues with enhanced filtering."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_issues', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        result = self._request("GET", f"/repos/{owner}/{repo}/issues", params=params,
                               cache_key=f"issues:{owner}/{repo}:{hash(str(params))}")

        # Filter out pull requests
        if isinstance(result, list):
            return [item for item in result if "pull_request" not in item]
        return result

    def search_issues(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search issues across GitHub."""
        cleaned_args = self._clean_args('search_issues', {'q': query, **kwargs})
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", "/search/issues", params=params)

    def add_issue_comment(self, issue_number: Union[int, str], body: str, owner: str = "", repo: str = "", **kwargs) -> \
    Dict[str, Any]:
        """Add comment to an issue."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        issue_num = normalize_issue_pr_input(str(issue_number))
        cleaned_args = self._clean_args('add_issue_comment', {'body': body, **kwargs})

        return self._request("POST", f"/repos/{owner}/{repo}/issues/{issue_num}/comments", json_body=cleaned_args)

    def add_sub_issue(self, parent_issue: Union[int, str], sub_issue: Union[int, str], owner: str = "",
                      repo: str = "") -> Dict[str, Any]:
        """Add a sub-issue relationship (using GitHub's sub-issue feature if available)."""
        # Note: This is a placeholder for GitHub's sub-issue feature
        # Implementation would depend on GitHub's actual API when available
        return {"message": "Sub-issue feature not yet implemented by GitHub API", "parent": parent_issue,
                "sub": sub_issue}

    def list_sub_issues(self, parent_issue: Union[int, str], owner: str = "", repo: str = "") -> Dict[str, Any]:
        """List sub-issues for a parent issue."""
        # Note: This is a placeholder for GitHub's sub-issue feature
        return {"message": "Sub-issue listing not yet implemented by GitHub API", "parent": parent_issue}

    def remove_sub_issue(self, parent_issue: Union[int, str], sub_issue: Union[int, str], owner: str = "",
                         repo: str = "") -> Dict[str, Any]:
        """Remove a sub-issue relationship."""
        if not self._check_destructive_operation('remove_sub_issue', {'parent': parent_issue, 'sub': sub_issue}):
            return {"dry_run": True, "operation": "remove_sub_issue",
                    "args": {"parent": parent_issue, "sub": sub_issue}}

        # Note: This is a placeholder for GitHub's sub-issue feature
        return {"message": "Sub-issue removal not yet implemented by GitHub API", "parent": parent_issue,
                "sub": sub_issue}

    def reprioritize_sub_issue(self, parent_issue: Union[int, str], sub_issue: Union[int, str], priority: int,
                               owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Reprioritize a sub-issue."""
        if not self._check_destructive_operation('reprioritize_sub_issue',
                                                 {'parent': parent_issue, 'sub': sub_issue, 'priority': priority}):
            return {"dry_run": True, "operation": "reprioritize_sub_issue",
                    "args": {"parent": parent_issue, "sub": sub_issue, "priority": priority}}

        # Note: This is a placeholder for GitHub's sub-issue feature
        return {"message": "Sub-issue reprioritization not yet implemented by GitHub API"}

    def assign_copilot_to_issue(self, issue_number: Union[int, str], owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Assign GitHub Copilot to an issue (gracefully handle if not available)."""
        try:
            # Note: This is a placeholder - GitHub Copilot issue assignment may not be available via API
            return {"message": "GitHub Copilot assignment not yet available via API", "issue": issue_number}
        except Exception as e:
            logger.warning(f"Copilot assignment failed gracefully: {e}")
            return {"error": "Copilot assignment not available", "issue": issue_number}

    # ============================================================================
    # PULL REQUEST OPERATIONS
    # ============================================================================

    def create_pull_request(self, title: str, head: str, base: str, owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Create a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not title.strip():
            raise GitHubError("Pull request title cannot be empty")

        cleaned_args = self._clean_args('create_pull_request', {
            'title': title.strip(), 'head': head, 'base': base, **kwargs
        })

        result = self._request("POST", f"/repos/{owner}/{repo}/pulls", json_body=cleaned_args)
        logger.info(f"Created PR #{result.get('number')} in {owner}/{repo}")

        self._invalidate_cache_pattern(f"pulls:{owner}/{repo}")
        return result

    def update_pull_request(self, pull_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Update an existing pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))
        cleaned_args = self._clean_args('update_pull_request', kwargs)

        result = self._request("PATCH", f"/repos/{owner}/{repo}/pulls/{pr_num}", json_body=cleaned_args)
        self._invalidate_cache_pattern(f"pulls:{owner}/{repo}")
        return result

    def merge_pull_request(self, pull_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Merge a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))

        if not self._check_destructive_operation('merge_pull_request', {'pull_number': pr_num, **kwargs}):
            return {"dry_run": True, "operation": "merge_pull_request", "args": {"pull_number": pr_num, **kwargs}}

        cleaned_args = self._clean_args('merge_pull_request', kwargs)

        result = self._request("PUT", f"/repos/{owner}/{repo}/pulls/{pr_num}/merge", json_body=cleaned_args)
        self._invalidate_cache_pattern(f"pulls:{owner}/{repo}")
        return result

    def list_pull_requests(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List pull requests."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_pull_requests', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", f"/repos/{owner}/{repo}/pulls", params=params,
                             cache_key=f"pulls:{owner}/{repo}:{hash(str(params))}")

    def get_pull_request(self, pull_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Get a specific pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))
        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_num}",
                             cache_key=f"pr:{owner}/{repo}:{pr_num}")

    def get_pull_request_files(self, pull_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Get files changed in a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))
        cleaned_args = self._clean_args('get_pull_request_files', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}

        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_num}/files", params=params)

    def get_pull_request_reviews(self, pull_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Get reviews for a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))
        cleaned_args = self._clean_args('get_pull_request_reviews', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}

        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_num}/reviews", params=params)

    def get_pull_request_status(self, pull_number: Union[int, str], owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get status checks for a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))

        # Get the PR to find the head SHA
        pr_data = self.get_pull_request(pr_num, owner, repo)
        head_sha = pr_data.get('head', {}).get('sha')

        if not head_sha:
            raise GitHubError("Could not determine head SHA for pull request")

        return self._request("GET", f"/repos/{owner}/{repo}/commits/{head_sha}/status")

    def get_pull_request_comments(self, pull_number: Union[int, str], owner: str = "", repo: str = "", **kwargs) -> \
    Dict[str, Any]:
        """Get comments on a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))
        cleaned_args = self._clean_args('get_pull_request_comments', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_num}/comments", params=params)

    def get_pull_request_diff(self, pull_number: Union[int, str], owner: str = "", repo: str = "") -> str:
        """Get diff for a pull request."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))

        # Use raw request to get diff format
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_num}"
        headers = self._get_headers()
        headers["Accept"] = "application/vnd.github.diff"

        response = self.session.get(url, headers=headers, timeout=self.config.timeout)
        if response.status_code >= 400:
            raise GitHubError(f"Failed to get PR diff: {response.status_code}")

        return response.text

    def request_copilot_review(self, pull_number: Union[int, str], owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Request GitHub Copilot review for a pull request (gracefully handle if not available)."""
        try:
            # Note: This is a placeholder - GitHub Copilot PR reviews may not be available via API
            return {"message": "GitHub Copilot PR review not yet available via API", "pull_request": pull_number}
        except Exception as e:
            logger.warning(f"Copilot review request failed gracefully: {e}")
            return {"error": "Copilot review not available", "pull_request": pull_number}

    def search_pull_requests(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search pull requests across GitHub."""
        # Add is:pr to query if not already present
        if "is:pr" not in query.lower():
            query = f"{query} is:pr"

        cleaned_args = self._clean_args('search_pull_requests', {'q': query, **kwargs})
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", "/search/issues", params=params)

    def update_pull_request_branch(self, pull_number: Union[int, str], owner: str = "", repo: str = "") -> Dict[
        str, Any]:
        """Update pull request branch with latest changes from base."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        pr_num = normalize_issue_pr_input(str(pull_number))

        return self._request("PUT", f"/repos/{owner}/{repo}/pulls/{pr_num}/update-branch",
                             json_body={"expected_head_sha": None})

    # ============================================================================
    # WORKFLOW OPERATIONS
    # ============================================================================

    def list_workflows(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List workflows."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_workflows', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}

        return self._request("GET", f"/repos/{owner}/{repo}/actions/workflows", params=params,
                             cache_key=f"workflows:{owner}/{repo}")

    def list_workflow_runs(self, workflow_input: Union[str, int] = None, owner: str = "", repo: str = "", **kwargs) -> \
    Dict[str, Any]:
        """List workflow runs."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_workflow_runs', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        if workflow_input:
            # Specific workflow
            workflow_id = self._resolve_workflow_id(owner, repo, workflow_input)
            return self._request("GET", f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs", params=params)
        else:
            # All workflows
            return self._request("GET", f"/repos/{owner}/{repo}/actions/runs", params=params)

    def get_workflow_run(self, run_id: int, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Get a specific workflow run."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}",
                             cache_key=f"run:{owner}/{repo}:{run_id}")

    def get_workflow_run_usage(self, run_id: int, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get workflow run usage."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}/timing")

    def get_workflow_run_logs(self, run_id: int, owner: str = "", repo: str = "") -> bytes:
        """Get workflow run logs."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
        response = self.session.get(url, headers=self._get_headers(), timeout=self.config.timeout)

        if response.status_code >= 400:
            raise GitHubError(f"Failed to get workflow logs: {response.status_code}")

        return response.content

    def get_job_logs(self, job_id: int, owner: str = "", repo: str = "") -> bytes:
        """Get job logs."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
        response = self.session.get(url, headers=self._get_headers(), timeout=self.config.timeout)

        if response.status_code >= 400:
            raise GitHubError(f"Failed to get job logs: {response.status_code}")

        return response.content

    def list_workflow_jobs(self, run_id: int, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List jobs for a workflow run."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        cleaned_args = self._clean_args('list_workflow_jobs', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", params=params)

    def list_workflow_run_artifacts(self, run_id: int, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """List artifacts for a workflow run."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts")

    def download_workflow_run_artifact(self, artifact_id: int, owner: str = "", repo: str = "") -> bytes:
        """Download a workflow run artifact."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/artifacts/{artifact_id}/zip"
        response = self.session.get(url, headers=self._get_headers(), timeout=self.config.timeout)

        if response.status_code >= 400:
            raise GitHubError(f"Failed to download artifact: {response.status_code}")

        return response.content

    def run_workflow(self, workflow_input: Union[str, int], ref: str, owner: str = "", repo: str = "", **kwargs) -> \
    Dict[str, Any]:
        """Run a workflow."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        workflow_id = self._resolve_workflow_id(owner, repo, workflow_input)
        cleaned_args = self._clean_args('run_workflow', {'ref': ref, **kwargs})

        payload = {"ref": ref}
        if 'inputs' in cleaned_args:
            payload["inputs"] = cleaned_args['inputs']

        # Workflow dispatch returns 204 on success
        response = self.session.post(
            f"{GITHUB_API}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
            headers=self._get_headers(),
            json=payload,
            timeout=self.config.timeout
        )

        self._update_rate_limit_info(response)

        if response.status_code >= 400:
            error_detail = self._parse_error_response(response)
            raise GitHubError(
                f"Workflow dispatch failed [{response.status_code}]: {error_detail}",
                status_code=response.status_code,
                response_data=error_detail
            )

        logger.info(f"Dispatched workflow '{workflow_id}' on {ref} in {owner}/{repo}")
        return {"ok": True, "status": response.status_code, "workflow_id": workflow_id, "ref": ref}

    def rerun_workflow_run(self, run_id: int, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Rerun a workflow run."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("POST", f"/repos/{owner}/{repo}/actions/runs/{run_id}/rerun", json_body={})

    def rerun_failed_jobs(self, run_id: int, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Rerun failed jobs in a workflow run."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("POST", f"/repos/{owner}/{repo}/actions/runs/{run_id}/rerun-failed-jobs", json_body={})

    def cancel_workflow_run(self, run_id: int, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Cancel a workflow run."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        if not self._check_destructive_operation('cancel_workflow_run', {'run_id': run_id}):
            return {"dry_run": True, "operation": "cancel_workflow_run", "args": {"run_id": run_id}}

        return self._request("POST", f"/repos/{owner}/{repo}/actions/runs/{run_id}/cancel", json_body={})

    # ============================================================================
    # SEARCH OPERATIONS
    # ============================================================================

    def search_code(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search code across GitHub."""
        cleaned_args = self._clean_args('search_code', {'q': query, **kwargs})
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", "/search/code", params=params)

    # ============================================================================
    # SECURITY OPERATIONS
    # ============================================================================

    def list_code_scanning_alerts(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """List code scanning alerts."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/code-scanning/alerts")

    def get_code_scanning_alert(self, alert_number: int, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get a specific code scanning alert."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/code-scanning/alerts/{alert_number}")

    def list_dependabot_alerts(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """List Dependabot alerts."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/dependabot/alerts")

    def get_dependabot_alert(self, alert_number: int, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get a specific Dependabot alert."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/dependabot/alerts/{alert_number}")

    def list_secret_scanning_alerts(self, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """List secret scanning alerts."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/secret-scanning/alerts")

    def get_secret_scanning_alert(self, alert_number: int, owner: str = "", repo: str = "") -> Dict[str, Any]:
        """Get a specific secret scanning alert."""
        owner = owner or self.default_owner
        repo = repo or self.default_repo
        if not (owner and repo):
            raise GitHubError("Owner and repo are required")

        return self._request("GET", f"/repos/{owner}/{repo}/secret-scanning/alerts/{alert_number}")

    # ============================================================================
    # NOTIFICATIONS
    # ============================================================================

    def list_notifications(self, **kwargs) -> Dict[str, Any]:
        """List notifications for the authenticated user."""
        cleaned_args = self._clean_args('list_notifications', kwargs)
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", "/notifications", params=params)

    def mark_notifications_read(self, **kwargs) -> Dict[str, Any]:
        """Mark notifications as read."""
        cleaned_args = self._clean_args('mark_notifications_read', kwargs)
        payload = cleaned_args or {}
        return self._request("PUT", "/notifications", json_body=payload)

    # ============================================================================
    # USER & ORG OPERATIONS
    # ============================================================================

    def search_users(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search users across GitHub."""
        cleaned_args = self._clean_args('search_users', {'q': query, **kwargs})
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", "/search/users", params=params)

    def search_orgs(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search organizations across GitHub."""
        cleaned_args = self._clean_args('search_orgs', {'q': query, **kwargs})
        params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
        params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

        return self._request("GET", "/search/users", params={**params, "type": "org"})

    # ============================================================================
    # DISCUSSION OPERATIONS (graceful fallbacks)
    # ============================================================================

    def list_discussions(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List repository discussions (if available via REST API)."""
        try:
            owner = owner or self.default_owner
            repo = repo or self.default_repo
            if not (owner and repo):
                raise GitHubError("Owner and repo are required")

            cleaned_args = self._clean_args('list_discussions', kwargs)
            params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
            params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

            return self._request("GET", f"/repos/{owner}/{repo}/discussions", params=params)
        except GitHubError as e:
            if e.status_code == 404:
                return {"message": "Discussions not available or not enabled for this repository", "discussions": []}
            raise

    def get_discussion(self, discussion_number: int, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """Get a specific discussion (if available via REST API)."""
        try:
            owner = owner or self.default_owner
            repo = repo or self.default_repo
            if not (owner and repo):
                raise GitHubError("Owner and repo are required")

            return self._request("GET", f"/repos/{owner}/{repo}/discussions/{discussion_number}")
        except GitHubError as e:
            if e.status_code == 404:
                return {"message": "Discussion not found or discussions not available",
                        "discussion_number": discussion_number}
            raise

    def list_discussion_comments(self, discussion_number: int, owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """List comments on a discussion (if available via REST API)."""
        try:
            owner = owner or self.default_owner
            repo = repo or self.default_repo
            if not (owner and repo):
                raise GitHubError("Owner and repo are required")

            cleaned_args = self._clean_args('list_discussion_comments', kwargs)
            params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}

            return self._request("GET", f"/repos/{owner}/{repo}/discussions/{discussion_number}/comments",
                                 params=params)
        except GitHubError as e:
            if e.status_code == 404:
                return {"message": "Discussion comments not available", "comments": []}
            raise

    # ============================================================================
    # GIST OPERATIONS (graceful fallbacks)
    # ============================================================================

    def create_gist(self, files: Dict[str, Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Create a gist (if gist API is accessible)."""
        try:
            cleaned_args = self._clean_args('create_gist', {'files': files, **kwargs})
            return self._request("POST", "/gists", json_body=cleaned_args)
        except GitHubError as e:
            if e.status_code in [404, 403]:
                return {"message": "Gist creation not supported in this build", "error": str(e)}
            raise

    def list_gists(self, **kwargs) -> Dict[str, Any]:
        """List gists for authenticated user (if gist API is accessible)."""
        try:
            cleaned_args = self._clean_args('list_gists', kwargs)
            params = {"per_page": min(cleaned_args.get("per_page", 30), self.config.max_per_page)}
            params.update({k: v for k, v in cleaned_args.items() if k != "per_page"})

            return self._request("GET", "/gists", params=params)
        except GitHubError as e:
            if e.status_code in [404, 403]:
                return {"message": "Gist listing not supported in this build", "gists": []}
            raise

    def update_gist(self, gist_id: str, **kwargs) -> Dict[str, Any]:
        """Update a gist (if gist API is accessible)."""
        try:
            cleaned_args = self._clean_args('update_gist', kwargs)
            return self._request("PATCH", f"/gists/{gist_id}", json_body=cleaned_args)
        except GitHubError as e:
            if e.status_code in [404, 403]:
                return {"message": "Gist updates not supported in this build", "gist_id": gist_id}
            raise

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_timestamps.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "cached_items": len(self._cache),
                "cache_enabled": self.config.enable_caching,
                "cache_ttl": self.config.cache_ttl,
                "rate_limit_remaining": self._rate_limit_remaining,
                "rate_limit_reset": self._rate_limit_reset_time
            }

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return self._request("GET", "/rate_limit")

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # ============================================================================
    # CONVENIENCE ALIASES FOR UI COMPATIBILITY
    # ============================================================================

    # These aliases ensure compatibility with the existing UI action mapping
    def open_pull_request(self, title: str, head: str, base: str, owner: str = "", repo: str = "", **kwargs) -> Dict[
        str, Any]:
        """Alias for create_pull_request to maintain UI compatibility."""
        return self.create_pull_request(title, head, base, owner, repo, **kwargs)

    def list_open_issues(self, owner: str = "", repo: str = "", **kwargs) -> Dict[str, Any]:
        """List open issues (convenience method)."""
        return self.list_issues(owner, repo, state="open", **kwargs)