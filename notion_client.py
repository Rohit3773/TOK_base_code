# clients/notion_client.py
from __future__ import annotations
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

NOTION_API = "https://api.notion.com"


class NotionError(RuntimeError):
    """Enhanced Notion error with additional context."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


@dataclass
class NotionConfig:
    """Configuration for Notion client performance tuning."""
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.3
    pool_connections: int = 10
    pool_maxsize: int = 20
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_per_page: int = 100
    enable_rate_limit_handling: bool = True


class NotionClient:
    """High-performance Notion client with connection pooling, caching, and rate limit handling."""

    def __init__(self, token: str, config: Optional[NotionConfig] = None):
        if not token:
            raise NotionError("Notion token is required")

        self._token = token
        self.config = config or NotionConfig()

        # Setup optimized session
        self.session = self._create_optimized_session()

        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

        # Rate limiting tracking
        self._rate_limit_remaining = 1000
        self._rate_limit_reset_time = 0
        self._rate_limit_lock = threading.RLock()

        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info("Initialized optimized Notion client")

    def _create_optimized_session(self) -> requests.Session:
        """Create session with connection pooling and retry strategy."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST", "PATCH"]
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
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
            "User-Agent": "OptimizedNotionClient/1.0"
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
            # Notion uses different rate limit headers
            remaining = response.headers.get('X-RateLimit-Remaining')
            if remaining:
                self._rate_limit_remaining = int(remaining)

            reset_time = response.headers.get('X-RateLimit-Reset')
            if reset_time:
                self._rate_limit_reset_time = int(reset_time)

    def _check_rate_limit(self) -> None:
        """Check and handle rate limiting."""
        if not self.config.enable_rate_limit_handling:
            return

        with self._rate_limit_lock:
            if self._rate_limit_remaining < 5:
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

        url = f"{NOTION_API}{path}"
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
                raise NotionError(
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
            raise NotionError(f"Request to {path} timed out after {request_timeout}s")
        except requests.exceptions.ConnectionError:
            raise NotionError(f"Connection error for {path}")
        except requests.exceptions.RequestException as e:
            raise NotionError(f"Request failed for {path}: {str(e)}")

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

    def get_users(self) -> List[Dict[str, Any]]:
        """Get all users with caching."""
        return self._request("GET", "/v1/users", cache_key="all_users")

    def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get specific user info with caching."""
        cache_key = f"user:{user_id}"
        return self._request("GET", f"/v1/users/{user_id}", cache_key=cache_key)

    def get_bot_info(self) -> Dict[str, Any]:
        """Get bot user info with caching."""
        return self._request("GET", "/v1/users/me", cache_key="bot_info")

    # ---------------- Enhanced Database Operations ----------------

    def get_database(self, database_id: str) -> Dict[str, Any]:
        """Get database information with caching."""
        cache_key = f"database:{database_id}"
        return self._request("GET", f"/v1/databases/{database_id}", cache_key=cache_key)

    def query_database(
            self,
            database_id: str,
            filter_conditions: Optional[Dict] = None,
            sorts: Optional[List[Dict]] = None,
            start_cursor: Optional[str] = None,
            page_size: int = 100
    ) -> Dict[str, Any]:
        """Query database with enhanced filtering and pagination."""

        page_size = min(page_size, self.config.max_per_page)

        body = {"page_size": page_size}

        if filter_conditions:
            body["filter"] = filter_conditions
        if sorts:
            body["sorts"] = sorts
        if start_cursor:
            body["start_cursor"] = start_cursor

        return self._request("POST", f"/v1/databases/{database_id}/query", json_body=body)

    def create_database(
            self,
            parent_page_id: str,
            title: str,
            properties: Dict[str, Any],
            description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create database with enhanced property support."""

        parent = {"type": "page_id", "page_id": parent_page_id}

        title_content = [{"type": "text", "text": {"content": title}}]

        body = {
            "parent": parent,
            "title": title_content,
            "properties": properties
        }

        if description:
            body["description"] = [{"type": "text", "text": {"content": description}}]

        result = self._request("POST", "/v1/databases", json_body=body)

        # Clear database cache
        self._invalidate_cache_pattern("database:")

        return result

    # ---------------- Enhanced Page Operations ----------------

    def get_page(self, page_id: str) -> Dict[str, Any]:
        """Get page information with caching."""
        cache_key = f"page:{page_id}"
        return self._request("GET", f"/v1/pages/{page_id}", cache_key=cache_key)

    def create_page(
            self,
            parent_database_id: Optional[str] = None,
            parent_page_id: Optional[str] = None,
            properties: Optional[Dict[str, Any]] = None,
            children: Optional[List[Dict]] = None,
            icon: Optional[Dict] = None,
            cover: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create page with enhanced content support."""

        if not parent_database_id and not parent_page_id:
            raise NotionError("Either parent_database_id or parent_page_id must be provided")

        if parent_database_id:
            parent = {"type": "database_id", "database_id": parent_database_id}
        else:
            parent = {"type": "page_id", "page_id": parent_page_id}

        body = {"parent": parent}

        if properties:
            body["properties"] = properties
        if children:
            body["children"] = children
        if icon:
            body["icon"] = icon
        if cover:
            body["cover"] = cover

        result = self._request("POST", "/v1/pages", json_body=body)

        # Clear page cache
        self._invalidate_cache_pattern("page:")

        logger.info(f"Created page {result.get('id')}")
        return result

    def update_page(
            self,
            page_id: str,
            properties: Optional[Dict[str, Any]] = None,
            archived: Optional[bool] = None,
            icon: Optional[Dict] = None,
            cover: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Update page with enhanced field support."""

        body = {}

        if properties:
            body["properties"] = properties
        if archived is not None:
            body["archived"] = archived
        if icon:
            body["icon"] = icon
        if cover:
            body["cover"] = cover

        result = self._request("PATCH", f"/v1/pages/{page_id}", json_body=body)

        # Clear page cache
        self._invalidate_cache_pattern(f"page:{page_id}")

        logger.info(f"Updated page {page_id}")
        return result

    # ---------------- Enhanced Block Operations ----------------

    def get_block_children(
            self,
            block_id: str,
            start_cursor: Optional[str] = None,
            page_size: int = 100
    ) -> Dict[str, Any]:
        """Get block children with pagination."""

        params = {"page_size": min(page_size, self.config.max_per_page)}

        if start_cursor:
            params["start_cursor"] = start_cursor

        cache_key = f"block_children:{block_id}:{hash(str(params))}"

        return self._request("GET", f"/v1/blocks/{block_id}/children", params=params, cache_key=cache_key)

    def append_block_children(
            self,
            block_id: str,
            children: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Append blocks to a page or block."""

        body = {"children": children}

        result = self._request("PATCH", f"/v1/blocks/{block_id}/children", json_body=body)

        # Clear block cache
        self._invalidate_cache_pattern(f"block_children:{block_id}")

        logger.info(f"Appended {len(children)} blocks to {block_id}")
        return result

    def update_block(self, block_id: str, block_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update a block."""

        result = self._request("PATCH", f"/v1/blocks/{block_id}", json_body=block_data)

        # Clear block cache
        self._invalidate_cache_pattern(f"block:")

        logger.info(f"Updated block {block_id}")
        return result

    def delete_block(self, block_id: str) -> Dict[str, Any]:
        """Delete a block."""

        result = self._request("DELETE", f"/v1/blocks/{block_id}")

        # Clear block cache
        self._invalidate_cache_pattern(f"block:")

        logger.info(f"Deleted block {block_id}")
        return result

    # ---------------- Enhanced Search Operations ----------------

    def search(
            self,
            query: str,
            sort_direction: str = "descending",
            sort_timestamp: str = "last_edited_time",
            filter_value: Optional[str] = None,
            start_cursor: Optional[str] = None,
            page_size: int = 100
    ) -> Dict[str, Any]:
        """Search with enhanced filtering and sorting."""

        body = {
            "query": query,
            "sort": {
                "direction": sort_direction,
                "timestamp": sort_timestamp
            },
            "page_size": min(page_size, self.config.max_per_page)
        }

        if filter_value:
            body["filter"] = {"value": filter_value, "property": "object"}
        if start_cursor:
            body["start_cursor"] = start_cursor

        return self._request("POST", "/v1/search", json_body=body)

    # ---------------- Helper Methods for Content Creation ----------------

    def create_text_block(self, text: str, block_type: str = "paragraph") -> Dict[str, Any]:
        """Create a text block with specified type."""
        return {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    def create_heading_block(self, text: str, level: int = 1) -> Dict[str, Any]:
        """Create a heading block."""
        heading_type = f"heading_{level}"
        return {
            "object": "block",
            "type": heading_type,
            heading_type: {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    def create_to_do_block(self, text: str, checked: bool = False) -> Dict[str, Any]:
        """Create a to-do block."""
        return {
            "object": "block",
            "type": "to_do",
            "to_do": {
                "rich_text": [{"type": "text", "text": {"content": text}}],
                "checked": checked
            }
        }

    def create_bulleted_list_block(self, text: str) -> Dict[str, Any]:
        """Create a bulleted list item block."""
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    def create_numbered_list_block(self, text: str) -> Dict[str, Any]:
        """Create a numbered list item block."""
        return {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            }
        }

    # ---------------- Property Helper Methods ----------------

    def create_title_property(self, text: str) -> Dict[str, Any]:
        """Create a title property."""
        return {
            "title": [{"type": "text", "text": {"content": text}}]
        }

    def create_rich_text_property(self, text: str) -> Dict[str, Any]:
        """Create a rich text property."""
        return {
            "rich_text": [{"type": "text", "text": {"content": text}}]
        }

    def create_number_property(self, number: Union[int, float]) -> Dict[str, Any]:
        """Create a number property."""
        return {"number": number}

    def create_select_property(self, name: str) -> Dict[str, Any]:
        """Create a select property."""
        return {"select": {"name": name}}

    def create_multi_select_property(self, names: List[str]) -> Dict[str, Any]:
        """Create a multi-select property."""
        return {"multi_select": [{"name": name} for name in names]}

    def create_date_property(self, start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Create a date property."""
        date_obj = {"start": start_date}
        if end_date:
            date_obj["end"] = end_date
        return {"date": date_obj}

    def create_checkbox_property(self, checked: bool) -> Dict[str, Any]:
        """Create a checkbox property."""
        return {"checkbox": checked}

    def create_url_property(self, url: str) -> Dict[str, Any]:
        """Create a URL property."""
        return {"url": url}

    def create_email_property(self, email: str) -> Dict[str, Any]:
        """Create an email property."""
        return {"email": email}

    def create_phone_property(self, phone: str) -> Dict[str, Any]:
        """Create a phone number property."""
        return {"phone_number": phone}

    # ---------------- Batch Operations ----------------

    def create_pages_batch(
            self,
            pages_data: List[Dict[str, Any]],
            parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """Create multiple pages efficiently."""

        if not pages_data:
            return []

        if parallel and len(pages_data) > 1:
            return self._create_pages_parallel(pages_data)
        else:
            return self._create_pages_sequential(pages_data)

    def _create_pages_sequential(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create pages sequentially."""
        results = []

        for i, page_data in enumerate(pages_data):
            try:
                result = self.create_page(**page_data)
                results.append({"success": True, "result": result, "index": i})
            except Exception as e:
                results.append({"success": False, "error": str(e), "index": i})

        return results

    def _create_pages_parallel(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create pages in parallel."""

        def create_single(page_with_index):
            index, page_data = page_with_index
            try:
                result = self.create_page(**page_data)
                return {"success": True, "result": result, "index": index}
            except Exception as e:
                return {"success": False, "error": str(e), "index": index}

        indexed_pages = list(enumerate(pages_data))
        results = list(self._executor.map(create_single, indexed_pages))

        return sorted(results, key=lambda x: x["index"])

    # ---------------- Pagination Support ----------------

    def query_database_all(
            self,
            database_id: str,
            filter_conditions: Optional[Dict] = None,
            sorts: Optional[List[Dict]] = None,
            max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Query database with automatic pagination."""

        all_results = []
        start_cursor = None
        page_count = 0

        while page_count < max_pages:
            try:
                result = self.query_database(
                    database_id,
                    filter_conditions=filter_conditions,
                    sorts=sorts,
                    start_cursor=start_cursor,
                    page_size=100
                )

                results = result.get("results", [])
                if not results:
                    break

                all_results.extend(results)

                # Check for next page
                if not result.get("has_more"):
                    break

                start_cursor = result.get("next_cursor")
                page_count += 1

            except NotionError as e:
                logger.error(f"Pagination failed on page {page_count + 1}: {e}")
                break

        return all_results

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

        logger.info("Notion client cache cleared")

    # ---------------- Resource Management ----------------

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("Notion client resources cleaned up")

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
            bot_info = self.get_bot_info()
            response_time = time.time() - start_time

            return {
                "healthy": True,
                "response_time": response_time,
                "bot_id": bot_info.get("id"),
                "bot_name": bot_info.get("name"),
                "rate_limit": self.get_rate_limit_status()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "rate_limit": self.get_rate_limit_status()
            }