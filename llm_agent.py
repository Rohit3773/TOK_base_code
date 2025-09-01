# llm_agent.py
from __future__ import annotations
import json, re
from typing import Any, Dict, Optional, List, Tuple, Union
from functools import lru_cache
from dataclasses import dataclass
import logging
import time
from openai import OpenAI

# Setup logging
logger = logging.getLogger(__name__)


class LLMPlanningError(RuntimeError):
    pass


# Optimized tool definitions with categories for faster lookup
@dataclass(frozen=True)
class ToolDefinition:
    name: str
    signature: str
    category: str
    priority: int = 1


# Organized tools by category for efficient filtering
GITHUB_TOOLS = [
    ToolDefinition('gh_create_issue', 'gh_create_issue(owner, repo, title, body?, labels?, assignees?)', 'create', 8),
    ToolDefinition('gh_list_open_issues', 'gh_list_open_issues(owner, repo, state?)', 'read', 5),
    ToolDefinition('gh_open_pull_request', 'gh_open_pull_request(owner, repo, head, base, title, body?, draft?)',
                   'create', 9),
    ToolDefinition('gh_create_branch', 'gh_create_branch(owner, repo, branch, from_branch?)', 'create', 7),
    ToolDefinition('gh_create_or_update_file', 'gh_create_or_update_file(owner, repo, path, content, message, branch)',
                   'modify', 8),
    ToolDefinition('gh_list_pull_requests', 'gh_list_pull_requests(owner, repo, state?)', 'read', 5),
    ToolDefinition('gh_run_workflow', 'gh_run_workflow(owner, repo, workflow_id, ref, inputs_json?)', 'execute', 9),
    ToolDefinition('gh_list_workflows', 'gh_list_workflows(owner, repo)', 'read', 4),
    ToolDefinition('gh_list_branches', 'gh_list_branches(owner, repo)', 'read', 4),
]

JIRA_TOOLS = [
    ToolDefinition('jira_create_issue', 'jira_create_issue(project_key, summary, description?, issuetype_name?)',
                   'create', 8),
    ToolDefinition('jira_add_comment', 'jira_add_comment(issue_key, body)', 'modify', 6),
    ToolDefinition('jira_transition_issue', 'jira_transition_issue(issue_key, transition_id)', 'modify', 7),
    ToolDefinition('jira_search', 'jira_search(jql, max_results?)', 'read', 5),
    ToolDefinition('jira_project_info', 'jira_project_info(project_key)', 'read', 4),
    ToolDefinition('jira_whoami', 'jira_whoami()', 'read', 3),
    ToolDefinition('jira_list_transitions', 'jira_list_transitions(issue_key)', 'read', 4),
]


# Cached compiled regex patterns for better performance
@lru_cache(maxsize=32)
def _get_repo_patterns():
    """Get compiled regex patterns for repository extraction."""
    return [
        re.compile(r"\brepo\s*\(\s*([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\s*\)", re.I),
        re.compile(r"\brepo\s*[:=]\s*([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I),
        re.compile(r"\bgithub\s+repo(?:sitory)?\s+([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I),
        re.compile(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b", re.I),
    ]


@lru_cache(maxsize=64)
def _extract_owner_repo(text: str) -> Optional[Tuple[str, str]]:
    """Cached repository extraction with optimized patterns."""
    if not text:
        return None

    text = text.strip()
    patterns = _get_repo_patterns()

    for pattern in patterns:
        match = pattern.search(text)
        if match:
            owner, repo = match.group(1), match.group(2)
            # Validate that this looks like a real repo (not too generic)
            if len(owner) > 1 and len(repo) > 1 and not owner.isdigit():
                return (owner, repo)

    return None


@lru_cache(maxsize=128)
def _infer_platform_intent(
        user_message: str,
        has_gh_context: bool,
        has_jira_context: bool
) -> Tuple[bool, bool]:
    """Cached platform intent inference with improved logic."""
    if not user_message:
        return (has_gh_context, has_jira_context)

    msg_lower = user_message.lower()

    # Enhanced keyword detection
    github_keywords = {'github', 'repo', 'repository', 'pull request', 'pr', 'branch', 'commit', 'workflow'}
    jira_keywords = {'jira', 'issue', 'ticket', 'project', 'jql', 'epic', 'story', 'bug', 'task'}

    # Count keyword occurrences for confidence scoring
    gh_score = sum(1 for kw in github_keywords if kw in msg_lower)
    jira_score = sum(1 for kw in jira_keywords if kw in msg_lower)

    # Check for explicit repo patterns
    has_repo_pattern = _extract_owner_repo(user_message) is not None
    if has_repo_pattern:
        gh_score += 2

    # Check for Jira issue patterns (PROJECT-123)
    if re.search(r'\b[A-Z][A-Z0-9]+-\d+\b', user_message):
        jira_score += 2

    mentions_github = gh_score > 0 or has_repo_pattern
    mentions_jira = jira_score > 0

    # Decision logic with context consideration
    if mentions_github and not mentions_jira:
        return (True, False)
    elif mentions_jira and not mentions_github:
        return (False, True)
    elif mentions_github and mentions_jira:
        return (True, True)  # Mixed intent - allow both
    else:
        # No clear platform mentioned - use context
        return (has_gh_context, has_jira_context)


def _filter_tools_by_intent(
        allow_gh: bool,
        allow_jira: bool,
        user_message: str
) -> List[ToolDefinition]:
    """Filter and prioritize tools based on user intent."""
    tools = []

    if allow_gh:
        # Prioritize based on message content
        if any(kw in user_message.lower() for kw in ['create', 'new', 'add']):
            tools.extend([t for t in GITHUB_TOOLS if t.category == 'create'])
            tools.extend([t for t in GITHUB_TOOLS if t.category != 'create'])
        elif any(kw in user_message.lower() for kw in ['list', 'show', 'get', 'find']):
            tools.extend([t for t in GITHUB_TOOLS if t.category == 'read'])
            tools.extend([t for t in GITHUB_TOOLS if t.category != 'read'])
        else:
            tools.extend(sorted(GITHUB_TOOLS, key=lambda t: -t.priority))

    if allow_jira:
        # Similar prioritization for Jira tools
        if any(kw in user_message.lower() for kw in ['create', 'new', 'add']):
            tools.extend([t for t in JIRA_TOOLS if t.category == 'create'])
            tools.extend([t for t in JIRA_TOOLS if t.category != 'create'])
        elif any(kw in user_message.lower() for kw in ['list', 'search', 'find']):
            tools.extend([t for t in JIRA_TOOLS if t.category == 'read'])
            tools.extend([t for t in JIRA_TOOLS if t.category != 'read'])
        else:
            tools.extend(sorted(JIRA_TOOLS, key=lambda t: -t.priority))

    return tools


@lru_cache(maxsize=32)
def _build_optimized_system_prompt(
        allow_gh: bool,
        allow_jira: bool,
        intent_hash: str
) -> str:
    """Cached system prompt generation with intent-based optimization."""

    # Build contextual tool list based on intent
    if allow_gh and allow_jira:
        tool_sections = [
            "GitHub Tools (server='github'):",
            *[f"- {t.signature}" for t in GITHUB_TOOLS[:5]],  # Top 5 most important
            "",
            "Jira Tools (server='jira'):",
            *[f"- {t.signature}" for t in JIRA_TOOLS[:5]],  # Top 5 most important
        ]
        routing_rules = [
            "- Analyze user intent to choose the appropriate platform",
            "- For repository operations (owner/repo), use GitHub tools",
            "- For issue tracking (project keys, JQL), use Jira tools",
            "- If both platforms mentioned, create coordinated actions"
        ]
    elif allow_gh:
        tool_sections = [
            "Available GitHub Tools:",
            *[f"- {t.signature}" for t in GITHUB_TOOLS],
        ]
        routing_rules = [
            "- Use ONLY GitHub tools",
            "- Extract owner/repo from user input",
            "- Focus on repository-based operations"
        ]
    elif allow_jira:
        tool_sections = [
            "Available Jira Tools:",
            *[f"- {t.signature}" for t in JIRA_TOOLS],
        ]
        routing_rules = [
            "- Use ONLY Jira tools",
            "- Focus on issue tracking operations",
            "- Use provided project keys"
        ]
    else:
        tool_sections = ["No tools available"]
        routing_rules = ["- Return empty actions array"]

    return f"""You are an intelligent task planner. Analyze the user request and generate an optimized execution plan.

RESPONSE FORMAT (JSON only):
{{
  "actions": [
    {{"tool": "tool_name", "args": {{"param": "value"}}}},
  ],
  "reasoning": "brief explanation of the plan"
}}

{chr(10).join(tool_sections)}

ROUTING RULES:
{chr(10).join(routing_rules)}

OPTIMIZATION GUIDELINES:
- Prioritize actions by importance and dependencies
- Group related operations logically
- Use exact parameter names as specified
- Never invent owner/repo/project values
- Be concise but complete in your plan"""


# Client connection pool for better performance
_openai_clients: Dict[str, OpenAI] = {}


def _get_openai_client(api_key: str) -> OpenAI:
    """Get or create cached OpenAI client."""
    if api_key not in _openai_clients:
        _openai_clients[api_key] = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=2
        )
    return _openai_clients[api_key]


def _parse_llm_response(raw_response: str) -> Dict[str, Any]:
    """Enhanced JSON parsing with multiple fallback strategies."""
    if not raw_response:
        return {"actions": []}

    # Strategy 1: Direct JSON parse
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from code blocks or mixed content
    json_patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Code blocks
        r'(\{(?:[^{}]|{[^{}]*})*\})',  # Balanced braces
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, raw_response, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Strategy 3: Try to fix common JSON issues
    try:
        # Remove common formatting issues
        cleaned = raw_response.strip()
        cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
        cleaned = re.sub(r',\s*]', ']', cleaned)
        cleaned = cleaned.replace("'", '"')  # Fix quotes

        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Extract actions array specifically
    actions_match = re.search(r'"actions"\s*:\s*\[(.*?)\]', raw_response, re.DOTALL)
    if actions_match:
        try:
            actions_json = f'[{actions_match.group(1)}]'
            actions = json.loads(actions_json)
            return {"actions": actions}
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse LLM response: {raw_response[:200]}...")
    return {"actions": []}


def _validate_and_filter_actions(
        actions: List[Dict[str, Any]],
        allow_gh: bool,
        allow_jira: bool
) -> List[Dict[str, Any]]:
    """Validate and filter actions based on platform permissions."""
    if not isinstance(actions, list):
        return []

    valid_actions = []

    for action in actions:
        if not isinstance(action, dict):
            continue

        tool = action.get("tool", "")
        if not tool:
            continue

        # Platform-based filtering with validation
        if tool.startswith("gh_") and allow_gh:
            # Validate GitHub tool exists
            if any(t.name == tool for t in GITHUB_TOOLS):
                valid_actions.append(action)
        elif tool.startswith("jira_") and allow_jira:
            # Validate Jira tool exists
            if any(t.name == tool for t in JIRA_TOOLS):
                valid_actions.append(action)

    return valid_actions


def propose_actions(
        openai_key: str,
        model: str,
        user_message: str,
        gh_owner: Optional[str] = None,
        gh_repo: Optional[str] = None,
        jira_project_key: Optional[str] = None,
        max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Generate action plan with comprehensive optimizations and error handling.

    Args:
        openai_key: OpenAI API key
        model: Model name (e.g., 'gpt-4o-mini')
        user_message: User's task description
        gh_owner: GitHub repository owner
        gh_repo: GitHub repository name
        jira_project_key: Jira project key
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with actions array and metadata
    """
    if not openai_key:
        raise LLMPlanningError("OpenAI key is required")

    start_time = time.time()

    # Extract repository info if not provided
    if not (gh_owner and gh_repo):
        repo_info = _extract_owner_repo(user_message or "")
        if repo_info:
            gh_owner, gh_repo = repo_info

    # Determine platform intent
    has_gh_ctx = bool(gh_owner and gh_repo)
    has_jira_ctx = bool(jira_project_key)
    allow_gh, allow_jira = _infer_platform_intent(user_message, has_gh_ctx, has_jira_ctx)

    # Build context efficiently
    context_parts = []
    if gh_owner and gh_repo and allow_gh:
        context_parts.append(f"GitHub: {gh_owner}/{gh_repo}")
    if jira_project_key and allow_jira:
        context_parts.append(f"Jira: {jira_project_key}")

    context = " | ".join(context_parts) if context_parts else "No specific context"

    # Generate intent hash for caching
    intent_hash = hash(f"{allow_gh}:{allow_jira}:{user_message[:100]}")

    # Get optimized system prompt
    system_prompt = _build_optimized_system_prompt(allow_gh, allow_jira, str(intent_hash))

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nTask: {user_message}"}
    ]

    # Execute with retry logic
    client = _get_openai_client(openai_key)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,  # Lower for more consistent results
                max_tokens=1000,  # Reasonable limit
            )

            raw_response = response.choices[0].message.content or ""
            break

        except Exception as e:
            last_error = str(e)
            logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                continue
            else:
                raise LLMPlanningError(f"LLM request failed after {max_retries} attempts: {last_error}")

    # Parse response
    try:
        parsed = _parse_llm_response(raw_response)
        actions = parsed.get("actions", [])

        # Validate and filter actions
        valid_actions = _validate_and_filter_actions(actions, allow_gh, allow_jira)

        # Calculate execution time
        execution_time = time.time() - start_time

        return {
            "actions": valid_actions,
            "metadata": {
                "execution_time": round(execution_time, 3),
                "model": model,
                "attempts": attempt + 1,
                "platforms": {
                    "github": allow_gh,
                    "jira": allow_jira
                },
                "context": context,
                "reasoning": parsed.get("reasoning", ""),
                "original_action_count": len(actions),
                "filtered_action_count": len(valid_actions)
            }
        }

    except Exception as e:
        raise LLMPlanningError(f"Failed to process LLM response: {e}")


# Enhanced batch processing for multiple requests
def propose_actions_batch(
        requests: List[Dict[str, Any]],
        openai_key: str,
        model: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """
    Process multiple action proposal requests efficiently.

    Args:
        requests: List of request dictionaries with user_message and optional context
        openai_key: OpenAI API key
        model: Model name

    Returns:
        List of results corresponding to input requests
    """
    results = []

    for i, request in enumerate(requests):
        try:
            result = propose_actions(
                openai_key=openai_key,
                model=model,
                user_message=request.get("user_message", ""),
                gh_owner=request.get("gh_owner"),
                gh_repo=request.get("gh_repo"),
                jira_project_key=request.get("jira_project_key"),
            )
            result["metadata"]["batch_index"] = i
            results.append(result)

        except Exception as e:
            results.append({
                "actions": [],
                "error": str(e),
                "metadata": {"batch_index": i, "failed": True}
            })

    return results


# Cache management utilities
def clear_caches():
    """Clear all caches to free memory."""
    _get_repo_patterns.cache_clear()
    _extract_owner_repo.cache_clear()
    _infer_platform_intent.cache_clear()
    _build_optimized_system_prompt.cache_clear()
    _openai_clients.clear()


def get_cache_stats():
    """Get cache statistics for monitoring."""
    return {
        "repo_patterns": _get_repo_patterns.cache_info()._asdict(),
        "extract_owner_repo": _extract_owner_repo.cache_info()._asdict(),
        "platform_intent": _infer_platform_intent.cache_info()._asdict(),
        "system_prompts": _build_optimized_system_prompt.cache_info()._asdict(),
        "openai_clients": len(_openai_clients)
    }


# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitoring for the LLM agent."""

    def __init__(self):
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
        }

    def record_request(self, success: bool, execution_time: float):
        """Record request statistics."""
        self.stats["total_requests"] += 1
        self.stats["total_execution_time"] += execution_time

        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1

        self.stats["average_execution_time"] = (
                self.stats["total_execution_time"] / self.stats["total_requests"]
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.stats.copy()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()