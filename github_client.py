# clients/github_client.py
from __future__ import annotations
import base64
import json
import requests
import time
import logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

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


class GitHubClient:
    """High-performance GitHub client with connection pooling, caching, and rate limit handling."""

    def __init__(self, token: str, config: Optional[GitHubConfig] = None):
        if not token:
            raise GitHubError("GitHub token is required")

        self._token = token
        self.config = config or GitHubConfig()

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

        logger.info("Initialized optimized GitHub client")

    def _create_optimized_session(self) -> requests.Session:
        """Create session with connection pooling and retry strategy."""
        session = requests.Session()

        # Configure retry strategy
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
            "User-Agent": "OptimizedGitHubClient/1.0"
        }

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
        """Check and handle rate limiting."""
        if not self.config.enable_rate_limit_handling:
            return

        with self._rate_limit_lock:
            if self._rate_limit_remaining < 10:
                current_time = time.time()
                if current_time < self._rate_limit_reset_time:
                    sleep_time = self._rate_limit_reset_time - current_time + 1
                    logger.warning(f"Rate limit low, sleeping for {sleep_time}s")
                    time.sleep(sleep_time)

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
    def get_user(self) -> Dict[str, Any]:
        """Get current user info with caching."""
        return self._request("GET", "/user", cache_key="current_user")

    def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information with caching."""
        cache_key = f"repo:{owner}/{repo}"
        return self._request("GET", f"/repos/{owner}/{repo}", cache_key=cache_key)

    def list_branches(self, owner: str, repo: str, per_page: int = 100) -> List[Dict[str, Any]]:
        """List branches with pagination support and caching."""
        cache_key = f"branches:{owner}/{repo}"
        params = {"per_page": min(per_page, self.config.max_per_page)}

        return self._request(
            "GET",
            f"/repos/{owner}/{repo}/branches",
            params=params,
            cache_key=cache_key
        )

    # ---------------- Optimized Repository Information ----------------

    @lru_cache(maxsize=128)
    def _get_default_branch_cached(self, owner: str, repo: str) -> str:
        """Get default branch with caching."""
        repo_data = self.get_repository(owner, repo)
        return repo_data.get("default_branch", "main")

    def _get_branch_sha(self, owner: str, repo: str, branch: str) -> str:
        """Get branch SHA with caching."""
        cache_key = f"branch_sha:{owner}/{repo}:{branch}"

        # Check cache first
        cached_sha = self._get_from_cache(cache_key)
        if cached_sha:
            return cached_sha

        ref_data = self._request("GET", f"/repos/{owner}/{repo}/git/ref/heads/{branch}")
        sha = (ref_data.get("object") or {}).get("sha")

        if sha:
            self._set_cache(cache_key, sha)

        return sha

    # ---------------- Enhanced Issue Operations ----------------

    def create_issue(
            self,
            owner: str,
            repo: str,
            title: str,
            body: str = "",
            labels: Optional[List[str]] = None,
            assignees: Optional[List[str]] = None,
            milestone: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create issue with enhanced validation and options."""

        if not title.strip():
            raise GitHubError("Issue title cannot be empty")

        payload = {"title": title.strip()}

        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        if milestone:
            payload["milestone"] = milestone

        try:
            result = self._request("POST", f"/repos/{owner}/{repo}/issues", json_body=payload)
            logger.info(f"Created issue #{result.get('number')} in {owner}/{repo}")

            # Clear issues cache
            self._invalidate_cache_pattern(f"issues:{owner}/{repo}")

            return result
        except GitHubError as e:
            logger.error(f"Failed to create issue in {owner}/{repo}: {e}")
            raise

    def list_issues(
            self,
            owner: str,
            repo: str,
            state: str = "open",
            per_page: int = 100,
            labels: Optional[List[str]] = None,
            assignee: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List issues with enhanced filtering and caching."""

        params = {
            "state": state,
            "per_page": min(per_page, self.config.max_per_page)
        }

        if labels:
            params["labels"] = ",".join(labels)
        if assignee:
            params["assignee"] = assignee

        cache_key = f"issues:{owner}/{repo}:{state}:{hash(str(params))}"

        try:
            items = self._request("GET", f"/repos/{owner}/{repo}/issues", params=params, cache_key=cache_key)
            # Filter out pull requests
            return [item for item in items if "pull_request" not in item]
        except GitHubError as e:
            logger.error(f"Failed to list issues for {owner}/{repo}: {e}")
            raise

    # ---------------- Enhanced Pull Request Operations ----------------

    def create_pull_request(
            self,
            owner: str,
            repo: str,
            head: str,
            base: str,
            title: str,
            body: str = "",
            draft: bool = False,
            maintainer_can_modify: bool = True
    ) -> Dict[str, Any]:
        """Create pull request with enhanced options."""

        if not title.strip():
            raise GitHubError("Pull request title cannot be empty")

        payload = {
            "title": title.strip(),
            "head": head,
            "base": base,
            "body": body,
            "draft": draft,
            "maintainer_can_modify": maintainer_can_modify
        }

        try:
            result = self._request("POST", f"/repos/{owner}/{repo}/pulls", json_body=payload)
            logger.info(f"Created PR #{result.get('number')} in {owner}/{repo}")

            # Clear PR cache
            self._invalidate_cache_pattern(f"pulls:{owner}/{repo}")

            return result
        except GitHubError as e:
            logger.error(f"Failed to create PR in {owner}/{repo}: {e}")
            raise

    # Alias for compatibility
    def open_pull_request(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias for create_pull_request for backward compatibility."""
        return self.create_pull_request(*args, **kwargs)

    def list_pull_requests(
            self,
            owner: str,
            repo: str,
            state: str = "open",
            per_page: int = 100,
            base: Optional[str] = None,
            head: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List pull requests with enhanced filtering."""

        params = {
            "state": state,
            "per_page": min(per_page, self.config.max_per_page)
        }

        if base:
            params["base"] = base
        if head:
            params["head"] = head

        cache_key = f"pulls:{owner}/{repo}:{state}:{hash(str(params))}"

        return self._request("GET", f"/repos/{owner}/{repo}/pulls", params=params, cache_key=cache_key)

    # ---------------- Enhanced Branch and File Operations ----------------

    def create_branch(
            self,
            owner: str,
            repo: str,
            branch: str,
            from_branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create branch with enhanced error handling."""

        try:
            base_branch = from_branch or self._get_default_branch_cached(owner, repo)
            sha = self._get_branch_sha(owner, repo, base_branch)

            payload = {"ref": f"refs/heads/{branch}", "sha": sha}
            result = self._request("POST", f"/repos/{owner}/{repo}/git/refs", json_body=payload)

            logger.info(f"Created branch '{branch}' from '{base_branch}' in {owner}/{repo}")

            # Clear branches cache
            self._invalidate_cache_pattern(f"branches:{owner}/{repo}")

            return result
        except GitHubError as e:
            logger.error(f"Failed to create branch '{branch}' in {owner}/{repo}: {e}")
            raise

    def create_or_update_file(
            self,
            owner: str,
            repo: str,
            path: str,
            content: Union[str, bytes],
            message: str,
            branch: str,
            committer: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Create or update file with enhanced content handling."""

        if not message.strip():
            raise GitHubError("Commit message cannot be empty")

        try:
            # Check if file exists
            sha = None
            try:
                existing_file = self._request(
                    "GET",
                    f"/repos/{owner}/{repo}/contents/{path}",
                    params={"ref": branch}
                )
                sha = existing_file.get("sha")
            except GitHubError as e:
                if e.status_code != 404:
                    raise
                # File doesn't exist, which is fine for creation

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

            if committer:
                payload["committer"] = committer

            result = self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", json_body=payload)

            action = "Updated" if sha else "Created"
            logger.info(f"{action} file '{path}' in {owner}/{repo}:{branch}")

            return result
        except GitHubError as e:
            logger.error(f"Failed to create/update file '{path}' in {owner}/{repo}: {e}")
            raise

    # ---------------- Enhanced Workflow Operations ----------------

    def list_workflows(self, owner: str, repo: str) -> Dict[str, Any]:
        """List workflows with caching."""
        cache_key = f"workflows:{owner}/{repo}"
        return self._request("GET", f"/repos/{owner}/{repo}/actions/workflows", cache_key=cache_key)

    def run_workflow(
            self,
            owner: str,
            repo: str,
            workflow_id: str,
            ref: str,
            inputs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run workflow with enhanced error handling."""

        payload = {"ref": ref}
        if inputs:
            payload["inputs"] = inputs

        try:
            # Use direct session call for workflow dispatch (returns 204)
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
            return {"ok": True, "status": response.status_code}

        except requests.exceptions.RequestException as e:
            raise GitHubError(f"Workflow dispatch failed: {str(e)}")

    # ---------------- Batch Operations ----------------

    def create_issues_batch(
            self,
            owner: str,
            repo: str,
            issues: List[Dict[str, Any]],
            parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Create multiple issues efficiently."""

        if not issues:
            return []

        if parallel and len(issues) > 1:
            return self._create_issues_parallel(owner, repo, issues)
        else:
            return self._create_issues_sequential(owner, repo, issues)

    def _create_issues_sequential(
            self,
            owner: str,
            repo: str,
            issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create issues sequentially."""
        results = []

        for i, issue_data in enumerate(issues):
            try:
                result = self.create_issue(owner, repo, **issue_data)
                results.append({"success": True, "result": result, "index": i})
            except Exception as e:
                results.append({"success": False, "error": str(e), "index": i})

        return results

    def _create_issues_parallel(
            self,
            owner: str,
            repo: str,
            issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create issues in parallel."""

        def create_single(issue_with_index):
            index, issue_data = issue_with_index
            try:
                result = self.create_issue(owner, repo, **issue_data)
                return {"success": True, "result": result, "index": index}
            except Exception as e:
                return {"success": False, "error": str(e), "index": index}

        indexed_issues = list(enumerate(issues))
        results = list(self._executor.map(create_single, indexed_issues))

        return sorted(results, key=lambda x: x["index"])

    # ---------------- Pagination Support ----------------

    def list_issues_all(
            self,
            owner: str,
            repo: str,
            state: str = "open",
            max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """List all issues with automatic pagination."""

        all_issues = []
        page = 1

        while page <= max_pages:
            try:
                issues = self.list_issues(owner, repo, state=state, per_page=100)
                if not issues:
                    break

                all_issues.extend(issues)

                if len(issues) < 100:
                    break  # Last page

                page += 1

            except GitHubError as e:
                logger.error(f"Pagination failed on page {page}: {e}")
                break

        return all_issues

    # ---------------- Cache Management ----------------

    def _invalidate_cache_pattern(self, pattern: str) -> None:
        """Invalidate cache entries matching a pattern."""
        with self._cache_lock:
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_timestamps.clear()

        # Clear LRU caches
        self.get_user.cache_clear()
        self._get_default_branch_cached.cache_clear()

        logger.info("GitHub client cache cleared")

    # ---------------- Resource Management ----------------

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("GitHub client resources cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---------------- Monitoring and Statistics ----------------

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self._rate_limit_lock:
            return {
                "remaining": self._rate_limit_remaining,
                "reset_time": self._rate_limit_reset_time,
                "reset_in_seconds": max(0, self._rate_limit_reset_time - time.time())
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "cache_size": len(self._cache),
                "cache_keys": list(self._cache.keys()),
                "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
                "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            start_time = time.time()
            user_info = self.get_user()
            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": response_time,
                "user": user_info.get("login"),
                "rate_limit": self.get_rate_limit_status()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "rate_limit": self.get_rate_limit_status()
            }