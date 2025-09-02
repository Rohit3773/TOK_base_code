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


# Expanded and organized GitHub tools by category for efficient filtering
GITHUB_TOOLS = [
    # Issue operations
    ToolDefinition('gh_create_issue', 'gh_create_issue(owner, repo, title, body?, labels?, assignees?)', 'create', 8),
    ToolDefinition('gh_update_issue', 'gh_update_issue(owner, repo, issue_number, title?, body?, state?, assignees?)',
                   'modify', 7),
    ToolDefinition('gh_get_issue', 'gh_get_issue(owner, repo, issue_number)', 'read', 5),
    ToolDefinition('gh_list_issues', 'gh_list_issues(owner, repo, state?, assignee?, labels?, sort?)', 'read', 5),
    ToolDefinition('gh_list_open_issues', 'gh_list_open_issues(owner, repo)', 'read', 5),
    ToolDefinition('gh_search_issues', 'gh_search_issues(query, sort?, order?)', 'read', 6),
    ToolDefinition('gh_add_issue_comment', 'gh_add_issue_comment(owner, repo, issue_number, body)', 'modify', 6),

    # Pull request operations
    ToolDefinition('gh_create_pull_request', 'gh_create_pull_request(owner, repo, title, head, base, body?, draft?)',
                   'create', 9),
    ToolDefinition('gh_open_pull_request', 'gh_open_pull_request(owner, repo, head, base, title, body?, draft?)',
                   'create', 9),
    ToolDefinition('gh_update_pull_request', 'gh_update_pull_request(owner, repo, pull_number, title?, body?, state?)',
                   'modify', 7),
    ToolDefinition('gh_merge_pull_request',
                   'gh_merge_pull_request(owner, repo, pull_number, commit_title?, commit_message?, merge_method?)',
                   'modify', 8),
    ToolDefinition('gh_list_pull_requests', 'gh_list_pull_requests(owner, repo, state?, head?, base?)', 'read', 5),
    ToolDefinition('gh_get_pull_request', 'gh_get_pull_request(owner, repo, pull_number)', 'read', 5),
    ToolDefinition('gh_get_pull_request_files', 'gh_get_pull_request_files(owner, repo, pull_number)', 'read', 5),
    ToolDefinition('gh_get_pull_request_reviews', 'gh_get_pull_request_reviews(owner, repo, pull_number)', 'read', 5),
    ToolDefinition('gh_get_pull_request_status', 'gh_get_pull_request_status(owner, repo, pull_number)', 'read', 5),
    ToolDefinition('gh_get_pull_request_comments', 'gh_get_pull_request_comments(owner, repo, pull_number)', 'read', 5),
    ToolDefinition('gh_get_pull_request_diff', 'gh_get_pull_request_diff(owner, repo, pull_number)', 'read', 5),
    ToolDefinition('gh_update_pull_request_branch', 'gh_update_pull_request_branch(owner, repo, pull_number)', 'modify',
                   6),
    ToolDefinition('gh_search_pull_requests', 'gh_search_pull_requests(query, sort?, order?)', 'read', 6),

    # Repository and file operations
    ToolDefinition('gh_create_branch', 'gh_create_branch(owner, repo, branch, from_branch?)', 'create', 7),
    ToolDefinition('gh_list_branches', 'gh_list_branches(owner, repo)', 'read', 4),
    ToolDefinition('gh_list_commits', 'gh_list_commits(owner, repo, sha?, path?, author?)', 'read', 4),
    ToolDefinition('gh_get_commit', 'gh_get_commit(owner, repo, ref)', 'read', 5),
    ToolDefinition('gh_get_file_contents', 'gh_get_file_contents(owner, repo, path, ref?)', 'read', 5),
    ToolDefinition('gh_create_or_update_file', 'gh_create_or_update_file(owner, repo, path, content, message, branch?)',
                   'modify', 8),
    ToolDefinition('gh_delete_file', 'gh_delete_file(owner, repo, path, message, branch?)', 'modify', 7),
    ToolDefinition('gh_push_files', 'gh_push_files(owner, repo, branch, files, message)', 'modify', 8),

    # Tags and releases
    ToolDefinition('gh_list_tags', 'gh_list_tags(owner, repo)', 'read', 4),
    ToolDefinition('gh_get_tag', 'gh_get_tag(owner, repo, tag)', 'read', 4),
    ToolDefinition('gh_list_releases', 'gh_list_releases(owner, repo)', 'read', 4),
    ToolDefinition('gh_get_latest_release', 'gh_get_latest_release(owner, repo)', 'read', 5),
    ToolDefinition('gh_get_release_by_tag', 'gh_get_release_by_tag(owner, repo, tag)', 'read', 5),

    # Workflow operations
    ToolDefinition('gh_list_workflows', 'gh_list_workflows(owner, repo)', 'read', 4),
    ToolDefinition('gh_list_workflow_runs',
                   'gh_list_workflow_runs(owner, repo, workflow_input?, actor?, branch?, event?, status?)', 'read', 5),
    ToolDefinition('gh_get_workflow_run', 'gh_get_workflow_run(owner, repo, run_id)', 'read', 5),
    ToolDefinition('gh_get_workflow_run_usage', 'gh_get_workflow_run_usage(owner, repo, run_id)', 'read', 5),
    ToolDefinition('gh_get_workflow_run_logs', 'gh_get_workflow_run_logs(owner, repo, run_id)', 'read', 5),
    ToolDefinition('gh_get_job_logs', 'gh_get_job_logs(owner, repo, job_id)', 'read', 5),
    ToolDefinition('gh_list_workflow_jobs', 'gh_list_workflow_jobs(owner, repo, run_id, filter?)', 'read', 5),
    ToolDefinition('gh_list_workflow_run_artifacts', 'gh_list_workflow_run_artifacts(owner, repo, run_id)', 'read', 5),
    ToolDefinition('gh_download_workflow_run_artifact', 'gh_download_workflow_run_artifact(owner, repo, artifact_id)',
                   'read', 5),
    ToolDefinition('gh_run_workflow', 'gh_run_workflow(owner, repo, workflow_input, ref, inputs?)', 'execute', 9),
    ToolDefinition('gh_rerun_workflow_run', 'gh_rerun_workflow_run(owner, repo, run_id)', 'execute', 7),
    ToolDefinition('gh_rerun_failed_jobs', 'gh_rerun_failed_jobs(owner, repo, run_id)', 'execute', 7),
    ToolDefinition('gh_cancel_workflow_run', 'gh_cancel_workflow_run(owner, repo, run_id)', 'execute', 6),

    # Search and security operations
    ToolDefinition('gh_search_code', 'gh_search_code(query, sort?, order?)', 'read', 6),
    ToolDefinition('gh_list_code_scanning_alerts', 'gh_list_code_scanning_alerts(owner, repo)', 'read', 5),
    ToolDefinition('gh_get_code_scanning_alert', 'gh_get_code_scanning_alert(owner, repo, alert_number)', 'read', 5),
    ToolDefinition('gh_list_dependabot_alerts', 'gh_list_dependabot_alerts(owner, repo)', 'read', 5),
    ToolDefinition('gh_get_dependabot_alert', 'gh_get_dependabot_alert(owner, repo, alert_number)', 'read', 5),
    ToolDefinition('gh_list_secret_scanning_alerts', 'gh_list_secret_scanning_alerts(owner, repo)', 'read', 5),
    ToolDefinition('gh_get_secret_scanning_alert', 'gh_get_secret_scanning_alert(owner, repo, alert_number)', 'read',
                   5),

    # Notifications and users
    ToolDefinition('gh_list_notifications', 'gh_list_notifications(all?, participating?, since?, before?)', 'read', 4),
    ToolDefinition('gh_mark_notifications_read', 'gh_mark_notifications_read(last_read_at?)', 'modify', 5),
    ToolDefinition('gh_search_users', 'gh_search_users(query, sort?, order?)', 'read', 4),
    ToolDefinition('gh_search_orgs', 'gh_search_orgs(query, sort?, order?)', 'read', 4),

    # Discussion and gist operations (graceful fallbacks)
    ToolDefinition('gh_list_discussions', 'gh_list_discussions(owner, repo, category_id?, labels?)', 'read', 4),
    ToolDefinition('gh_get_discussion', 'gh_get_discussion(owner, repo, discussion_number)', 'read', 4),
    ToolDefinition('gh_list_discussion_comments', 'gh_list_discussion_comments(owner, repo, discussion_number)', 'read',
                   4),
    ToolDefinition('gh_create_gist', 'gh_create_gist(files, description?, public?)', 'create', 5),
    ToolDefinition('gh_list_gists', 'gh_list_gists(since?)', 'read', 4),
    ToolDefinition('gh_update_gist', 'gh_update_gist(gist_id, files?, description?)', 'modify', 5),
]

JIRA_TOOLS = [
    ToolDefinition('jira_create_issue', 'jira_create_issue(project_key, summary, description?, issuetype_name?)',
                   'create', 8),
    ToolDefinition('jira_add_comment', 'jira_add_comment(issue_key, body)', 'modify', 6),
    ToolDefinition('jira_transition_issue', 'jira_transition_issue(issue_key, transition_id)', 'modify', 7),
    ToolDefinition('jira_list_transitions', 'jira_list_transitions(issue_key)', 'read', 4),
    ToolDefinition('jira_search', 'jira_search(jql, max_results?)', 'read', 5),
    ToolDefinition('jira_project_info', 'jira_project_info(project_key)', 'read', 4),
    ToolDefinition('jira_whoami', 'jira_whoami()', 'read', 3),
]


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


@lru_cache(maxsize=64)
def _extract_issue_number(text: str) -> Optional[int]:
    """Extract issue/PR number from text like '#123' or URLs."""
    if not text:
        return None

    # Look for #number pattern
    number_match = re.search(r'#(\d+)', text)
    if number_match:
        return int(number_match.group(1))

    # Look for GitHub URLs with issue/PR numbers
    url_patterns = [
        r'github\.com/[^/]+/[^/]+/issues/(\d+)',
        r'github\.com/[^/]+/[^/]+/pull/(\d+)'
    ]

    for pattern in url_patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    # Look for standalone numbers if context suggests it's an issue
    if 'issue' in text.lower() or 'pr' in text.lower():
        number_match = re.search(r'\b(\d+)\b', text)
        if number_match:
            return int(number_match.group(1))

    return None


@lru_cache(maxsize=64)
def _extract_jira_key(text: str) -> Optional[str]:
    """Extract Jira issue key like 'PROJECT-123'."""
    if not text:
        return None

    # Look for PROJECT-123 pattern
    jira_match = re.search(r'\b([A-Z][A-Z0-9]+-\d+)\b', text)
    if jira_match:
        return jira_match.group(1)

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
    github_keywords = {'github', 'repo', 'repository', 'pull request', 'pr', 'branch', 'commit', 'workflow', 'release',
                       'tag', 'gist'}
    jira_keywords = {'jira', 'issue', 'ticket', 'project', 'jql', 'epic', 'story', 'bug', 'task'}

    # Count keyword occurrences for confidence scoring
    gh_score = sum(1 for kw in github_keywords if kw in msg_lower)
    jira_score = sum(1 for kw in jira_keywords if kw in msg_lower)

    # Check for explicit repo patterns
    has_repo_pattern = _extract_owner_repo(user_message) is not None
    if has_repo_pattern:
        gh_score += 2

    # Check for Jira issue patterns (PROJECT-123)
    if _extract_jira_key(user_message):
        jira_score += 2

    # Check for GitHub issue patterns (#123)
    if _extract_issue_number(user_message):
        gh_score += 1

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


def _detect_status_change_intent(user_message: str) -> Optional[Tuple[str, str]]:
    """Detect status change intent and return (platform, new_state)."""
    msg_lower = user_message.lower()

    # Status change keywords
    close_keywords = ['close', 'closed', 'mark closed', 'set closed']
    reopen_keywords = ['reopen', 'open', 'mark open', 'set open']
    resolve_keywords = ['resolve', 'resolved', 'mark resolved', 'done', 'complete', 'finished']

    # Check for GitHub or Jira context
    has_github_ref = _extract_issue_number(user_message) or 'github' in msg_lower
    has_jira_ref = _extract_jira_key(user_message) or 'jira' in msg_lower

    # Determine new state
    new_state = None
    if any(kw in msg_lower for kw in close_keywords):
        new_state = 'closed'
    elif any(kw in msg_lower for kw in reopen_keywords):
        new_state = 'open'
    elif any(kw in msg_lower for kw in resolve_keywords):
        new_state = 'done'  # For Jira, we'll map this to transitions

    if not new_state:
        return None

    # Determine platform preference
    if has_github_ref and not has_jira_ref:
        return ('github', new_state)
    elif has_jira_ref and not has_github_ref:
        return ('jira', new_state)
    elif has_github_ref and has_jira_ref:
        # Both mentioned - prefer the one with specific reference
        if _extract_issue_number(user_message):
            return ('github', new_state)
        elif _extract_jira_key(user_message):
            return ('jira', new_state)

    return None


def _build_github_search_query(user_message: str, owner: str, repo: str, item_type: str = 'issue') -> str:
    """Build a proper GitHub search query."""
    query_parts = []

    # Add repo constraint if available
    if owner and repo:
        query_parts.append(f'repo:{owner}/{repo}')

    # Extract title text for search
    msg_lower = user_message.lower()

    # Remove common words and extract meaningful title terms
    title_words = []
    words = user_message.split()
    skip_words = {'issue', 'pr', 'pull', 'request', 'github', 'about', 'details', 'show', 'me', 'find', 'get', 'the',
                  'a', 'an'}

    for word in words:
        cleaned = re.sub(r'[^\w]', '', word.lower())
        if len(cleaned) > 2 and cleaned not in skip_words:
            title_words.append(word)

    if title_words:
        title_text = ' '.join(title_words[:5])  # Limit to 5 words
        query_parts.append(f'in:title "{title_text}"')

    # Add type constraint
    query_parts.append(f'type:{item_type}')

    return ' '.join(query_parts)


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
            "GitHub Tools (service='github'):",
            *[f"- {t.signature}" for t in GITHUB_TOOLS[:15]],  # Top 15 most important
            "... and many more GitHub operations for files, workflows, security, etc.",
            "",
            "Jira Tools (service='jira'):",
            *[f"- {t.signature}" for t in JIRA_TOOLS],  # All Jira tools
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
            *[f"- {t.signature}" for t in GITHUB_TOOLS[:20]],  # Show more when only GitHub
            "... and additional operations for security, notifications, discussions, gists",
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
  "message": "Brief response to user - NO MARKDOWN TABLES, just summary text",
  "actions": [
    {{"service": "github|jira|notion", "action": "tool_name", "args": {{"param": "value"}}, "description": "what this does"}},
  ]
}}

{chr(10).join(tool_sections)}

ROUTING RULES:
{chr(10).join(routing_rules)}

CRITICAL SEARCH RULES:
1. GitHub search_issues REQUIRES query parameter - never call without it
2. For issue details by title: use search_issues with proper query format
3. For issue by #number: use get_issue directly
4. For status changes: use update_issue (GitHub) or transition_issue (Jira)

ISSUE REFERENCE PARSING:
- "#123" or URLs → get_issue with issue_number
- Title text → search_issues with query="repo:owner/repo in:title \"text\" type:issue"
- Jira keys like "PROJECT-123" → use directly in Jira calls

STATUS CHANGES:
- GitHub: update_issue with state="closed|open"
- Jira: list_transitions then transition_issue with matching transition_id

MESSAGE RULES:
- Keep messages concise - just counts and summaries
- NO markdown tables in message - data shows in UI tables
- Example: "Found 5 issues" not full issue listings

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
        return {"message": "", "actions": []}

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
            return {"message": "Task processed", "actions": actions}
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse LLM response: {raw_response[:200]}...")
    return {"message": raw_response[:500], "actions": []}


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

        service = action.get("service", "")
        action_name = action.get("action", "")

        if not service or not action_name:
            continue

        # Platform-based filtering with validation
        if service == "github" and allow_gh:
            # Validate GitHub tool exists
            tool_name = f"gh_{action_name}" if not action_name.startswith("gh_") else action_name
            if any(t.name == tool_name for t in GITHUB_TOOLS):
                valid_actions.append(action)
        elif service == "jira" and allow_jira:
            # Validate Jira tool exists
            tool_name = f"jira_{action_name}" if not action_name.startswith("jira_") else action_name
            if any(t.name == tool_name for t in JIRA_TOOLS):
                valid_actions.append(action)
        elif service == "notion":
            # Keep notion actions as-is for compatibility
            valid_actions.append(action)

    return valid_actions


def _enhance_actions_for_context(
        actions: List[Dict[str, Any]],
        user_message: str,
        gh_owner: str,
        gh_repo: str
) -> List[Dict[str, Any]]:
    """Enhance actions with context-specific improvements."""
    enhanced_actions = []

    # Check for status change intent
    status_intent = _detect_status_change_intent(user_message)
    issue_number = _extract_issue_number(user_message)
    jira_key = _extract_jira_key(user_message)

    for action in actions:
        service = action.get("service", "")
        action_name = action.get("action", "")
        args = action.get("args", {})

        # Handle GitHub search issues - ensure query is present
        if service == "github" and action_name == "search_issues":
            if "query" not in args or not args["query"]:
                # Build a proper search query
                query = _build_github_search_query(user_message, gh_owner, gh_repo, "issue")
                args["query"] = query
                action["args"] = args

        # Handle GitHub search pull requests - ensure query is present
        elif service == "github" and action_name == "search_pull_requests":
            if "query" not in args or not args["query"]:
                query = _build_github_search_query(user_message, gh_owner, gh_repo, "pr")
                args["query"] = query
                action["args"] = args

        # Handle issue details by number
        elif service == "github" and issue_number and (
                "detail" in user_message.lower() or "show" in user_message.lower()):
            if action_name == "search_issues":
                # Replace with direct get_issue call
                action["action"] = "get_issue"
                action["args"] = {"issue_number": issue_number}
                action["description"] = f"Get details for issue #{issue_number}"

        # Handle status changes
        elif status_intent:
            platform, new_state = status_intent

            if service == "github" and platform == "github":
                if action_name == "search_issues" and issue_number:
                    # Add follow-up status change action
                    enhanced_actions.append(action)  # Keep search
                    enhanced_actions.append({
                        "service": "github",
                        "action": "update_issue",
                        "args": {"issue_number": issue_number, "state": new_state if new_state != "done" else "closed"},
                        "description": f"Change issue #{issue_number} to {new_state}"
                    })
                    continue
                elif action_name == "get_issue" and issue_number:
                    # Add follow-up status change action
                    enhanced_actions.append(action)  # Keep get
                    enhanced_actions.append({
                        "service": "github",
                        "action": "update_issue",
                        "args": {"issue_number": issue_number, "state": new_state if new_state != "done" else "closed"},
                        "description": f"Change issue #{issue_number} to {new_state}"
                    })
                    continue

            elif service == "jira" and platform == "jira":
                if jira_key:
                    # For Jira status changes, we need to get transitions first
                    enhanced_actions.append({
                        "service": "jira",
                        "action": "list_transitions",
                        "args": {"issue_key": jira_key},
                        "description": f"Get available transitions for {jira_key}"
                    })
                    enhanced_actions.append({
                        "service": "jira",
                        "action": "transition_issue",
                        "args": {"issue_key": jira_key, "target_state": new_state},
                        "description": f"Move {jira_key} to {new_state}"
                    })
                    continue

        enhanced_actions.append(action)

    return enhanced_actions


def propose_actions(
        openai_key: str,
        model: str,
        user_message: str,
        gh_owner: Optional[str] = None,
        gh_repo: Optional[str] = None,
        jira_project_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,  # Added for compatibility
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
        notion_database_id: Notion database ID (for compatibility)
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
                max_tokens=1500,  # Increased for more complex responses
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
        message = parsed.get("message", "Task processed")

        # Validate and filter actions
        valid_actions = _validate_and_filter_actions(actions, allow_gh, allow_jira)

        # Enhance actions with context-specific improvements
        enhanced_actions = _enhance_actions_for_context(valid_actions, user_message, gh_owner or "", gh_repo or "")

        # Calculate execution time
        execution_time = time.time() - start_time

        return {
            "message": message,
            "actions": enhanced_actions,
            "metadata": {
                "execution_time": round(execution_time, 3),
                "model": model,
                "attempts": attempt + 1,
                "platforms": {
                    "github": allow_gh,
                    "jira": allow_jira
                },
                "context": context,
                "original_action_count": len(actions),
                "filtered_action_count": len(valid_actions),
                "enhanced_action_count": len(enhanced_actions)
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
                notion_database_id=request.get("notion_database_id"),
            )
            result["metadata"]["batch_index"] = i
            results.append(result)

        except Exception as e:
            results.append({
                "message": f"Error processing request: {str(e)}",
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
    _extract_issue_number.cache_clear()
    _extract_jira_key.cache_clear()
    _infer_platform_intent.cache_clear()
    _build_optimized_system_prompt.cache_clear()
    _openai_clients.clear()


def get_cache_stats():
    """Get cache statistics for monitoring."""
    return {
        "repo_patterns": _get_repo_patterns.cache_info()._asdict(),
        "extract_owner_repo": _extract_owner_repo.cache_info()._asdict(),
        "extract_issue_number": _extract_issue_number.cache_info()._asdict(),
        "extract_jira_key": _extract_jira_key.cache_info()._asdict(),
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