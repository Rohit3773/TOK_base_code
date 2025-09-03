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
    ToolDefinition('gh_search_issues', 'gh_search_issues(query, sort?, order?, per_page?)', 'read', 6),
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
    ToolDefinition('gh_search_pull_requests', 'gh_search_pull_requests(query, sort?, order?, per_page?)', 'read', 6),

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
    """Extract issue/PR number from text like '#123' or URLs with improved accuracy."""
    if not text:
        return None

    # Look for #number pattern first (most specific)
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
    issue_keywords = ['issue', 'pr', 'pull request', 'bug', 'ticket', 'github']
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in issue_keywords):
        number_match = re.search(r'\b(\d+)\b', text)
        if number_match:
            num = int(number_match.group(1))
            # Reasonable range check for issue numbers
            if 1 <= num <= 999999:
                return num

    return None


@lru_cache(maxsize=64)
def _extract_jira_key(text: str) -> Optional[str]:
    """Extract Jira issue key like 'PROJECT-123' with improved patterns."""
    if not text:
        return None

    # Look for PROJECT-123 pattern (more robust)
    jira_match = re.search(r'\b([A-Z][A-Z0-9]{1,9}-\d+)\b', text)
    if jira_match:
        return jira_match.group(1)

    return None


@lru_cache(maxsize=64)
def _extract_issue_title(text: str) -> Optional[str]:
    """Extract issue title from various patterns."""
    if not text:
        return None

    # Patterns for extracting titles
    title_patterns = [
        r'issue\s+(?:titled|named|called)\s*["\']([^"\']+)["\']',
        r'issue\s+["\']([^"\']+)["\']',
        r'(?:details|info|about)\s+(?:issue\s+)?["\']([^"\']+)["\']',
        r'(?:details|info|about)\s+(?:GitHub\s+)?issue\s+([A-Za-z][A-Za-z0-9\s]+)',
        r'GitHub\s+issue\s+([A-Za-z][A-Za-z0-9\s]+?)(?:\s|$)',
    ]

    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Filter out common false positives
            if len(title) > 1 and not title.isdigit() and title.lower() not in ['details', 'info', 'about']:
                return title

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
                       'tag', 'gist', 'file contents', 'merge'}
    jira_keywords = {'jira', 'issue', 'ticket', 'project', 'jql', 'epic', 'story', 'bug', 'task', 'transition'}

    # Count keyword occurrences for confidence scoring
    gh_score = sum(1 for kw in github_keywords if kw in msg_lower)
    jira_score = sum(1 for kw in jira_keywords if kw in msg_lower)

    # Check for explicit patterns
    has_repo_pattern = _extract_owner_repo(user_message) is not None
    has_gh_issue = _extract_issue_number(user_message) is not None
    has_jira_key = _extract_jira_key(user_message) is not None

    if has_repo_pattern:
        gh_score += 2
    if has_gh_issue:
        gh_score += 2
    if has_jira_key:
        jira_score += 3

    mentions_github = gh_score > 0 or has_repo_pattern or has_gh_issue
    mentions_jira = jira_score > 0 or has_jira_key

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


def _is_greeting(user_message: str) -> bool:
    """Detect if message is a greeting."""
    if not user_message:
        return False

    msg_lower = user_message.lower().strip()
    greeting_patterns = [
        r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))[\s!.]*$',
        r'^(what\'s\s+up|sup|howdy)[\s!.]*$',
        r'^(how\s+are\s+you|how\'s\s+it\s+going)[\s!.?]*$'
    ]

    return any(re.match(pattern, msg_lower) for pattern in greeting_patterns)


def _detect_status_change_intent(user_message: str) -> Optional[Tuple[str, str]]:
    """Detect status change intent and return (platform, new_state)."""
    msg_lower = user_message.lower()

    # Status change keywords with more patterns
    close_keywords = ['close', 'closed', 'mark closed', 'set closed', 'mark as closed', 'set to closed']
    reopen_keywords = ['reopen', 'open', 'mark open', 'set open', 'mark as open', 'set to open']
    resolve_keywords = ['resolve', 'resolved', 'mark resolved', 'done', 'complete', 'finished', 'mark done',
                        'mark as done', 'mark to done', 'move to done', 'set to done']
    progress_keywords = ['in progress', 'start', 'started', 'begin', 'working on']

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
    elif any(kw in msg_lower for kw in progress_keywords):
        new_state = 'in_progress'

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
    """Build a proper GitHub search query with improved title extraction."""
    query_parts = []

    # Add repo constraint if available
    if owner and repo:
        query_parts.append(f'repo:{owner}/{repo}')

    # Extract title text for search
    title_text = _extract_issue_title(user_message)

    if title_text:
        query_parts.append(f'in:title "{title_text}"')
        query_parts.append('state:all')  # Search all states by default
    else:
        # Extract meaningful search terms
        msg_lower = user_message.lower()

        # Look for quoted phrases first
        quoted_matches = re.findall(r'"([^"]+)"', user_message)
        for quote in quoted_matches:
            if len(quote.strip()) > 2:
                query_parts.append(f'"{quote.strip()}"')

        if not quoted_matches:
            # Extract key terms, excluding common words
            words = user_message.split()
            skip_words = {'issue', 'pr', 'pull', 'request', 'github', 'about', 'details', 'show', 'me', 'find', 'get',
                          'the', 'a', 'an', 'search', 'for', 'with', 'in', 'of', 'and', 'or', 'but', 'is', 'are',
                          'tell', 'what', 'how', 'why', 'when', 'where'}

            search_terms = []
            for word in words:
                cleaned = re.sub(r'[^\w]', '', word.lower())
                if len(cleaned) > 2 and cleaned not in skip_words and not cleaned.isdigit():
                    search_terms.append(cleaned)

            if search_terms:
                # Use the first few meaningful terms
                for term in search_terms[:2]:
                    query_parts.append(f'"{term}"')

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
            "- gh_get_issue(owner, repo, issue_number) - Get specific issue details",
            "- gh_search_issues(query, per_page?) - Search issues (query REQUIRED)",
            "- gh_create_issue(owner, repo, title, body?) - Create new issue",
            "- gh_update_issue(owner, repo, issue_number, state?, title?, body?) - Update issue",
            "- gh_add_issue_comment(owner, repo, issue_number, body) - Add comment",
            "- gh_list_issues(owner, repo, state?, labels?) - List issues",
            "- gh_create_pull_request(owner, repo, title, head, base, body?) - Create PR",
            "- gh_list_branches(owner, repo) - List branches",
            "- gh_get_file_contents(owner, repo, path, ref?) - Get file contents",
            "- gh_create_or_update_file(owner, repo, path, content, message, branch?) - Create/update file",
            "- gh_list_tags(owner, repo) - List tags",
            "... and many more GitHub operations for workflows, security, etc.",
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
            "- gh_get_issue(owner, repo, issue_number) - Get specific issue details",
            "- gh_search_issues(query, per_page?) - Search issues (query REQUIRED)",
            "- gh_create_issue(owner, repo, title, body?) - Create new issue",
            "- gh_update_issue(owner, repo, issue_number, state?, title?, body?) - Update issue",
            "- gh_add_issue_comment(owner, repo, issue_number, body) - Add comment",
            "- gh_list_issues(owner, repo, state?, labels?) - List issues",
            "- gh_create_pull_request(owner, repo, title, head, base, body?) - Create PR",
            "- gh_list_branches(owner, repo) - List branches",
            "- gh_get_file_contents(owner, repo, path, ref?) - Get file contents",
            "- gh_create_or_update_file(owner, repo, path, content, message, branch?) - Create/update file",
            "- gh_list_tags(owner, repo) - List tags",
            "... and additional operations for security, notifications, workflows, etc.",
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

CRITICAL ENTITY NORMALIZATION RULES:

1. GREETING DETECTION:
   - For simple greetings like "Hey", "Hello", "Hi" → return ONLY friendly message, NO actions
   - Message: "Hi! I'm ready to help. I can list GitHub/Jira issues, show details, create issues/PRs, update statuses, etc."

2. GITHUB SEARCH WITH NO RESULTS HANDLING:
   - For searches like "Search GitHub issues for memory leak" → use search_issues with query
   - If no results found, message should be "No issues found for search term" 
   - ALWAYS include a search action even if expecting no results

3. GITHUB ISSUE REFERENCES:
   - "#123" or "issue #123" → get_issue with issue_number=123
   - "GitHub issue Blue" or "issue titled Blue" → search_issues first with query like 'repo:owner/repo "Blue" type:issue state:all'
   - GitHub URLs → extract number, use get_issue

4. JIRA ISSUE REFERENCES AND STATUS CHANGES:
   - "ABC-123" → use issue_key="ABC-123" directly
   - "set Jira issue MTP-11 to Done" → ALWAYS use list_transitions FIRST, then transition_issue
   - For Jira status changes: MUST get transitions first to find correct transition_id
   - Pattern: list_transitions(issue_key) → then transition_issue(issue_key, transition_id)

5. SEARCH QUERIES ALWAYS REQUIRED:
   - search_issues MUST have query parameter - never call without it
   - For title searches: query = 'repo:owner/repo "Blue" type:issue state:all'
   - For text searches: query = 'repo:owner/repo "memory leak" type:issue'
   - search_pull_requests MUST have query parameter with type:pr

6. PULL REQUEST CREATION:
   - "Create pull request title X head Y base Z body W" → create_pull_request with ALL required params
   - REQUIRED params: owner, repo, title, head, base
   - OPTIONAL params: body, draft
   - Example: {{"service": "github", "action": "create_pull_request", "args": {{"title": "Update docs", "head": "feature/test-cli", "base": "main", "body": "Add hello doc"}}}}

7. FILE OPERATIONS:
   - "Get file contents path: docs/HELLO.md from main branch" → get_file_contents(path="docs/HELLO.md", ref="main")
   - ALWAYS include both path AND ref parameters

8. BRANCH AND TAG LISTING:
   - "List tags in GitHub" → list_tags(owner, repo) 
   - "Tell me name of branches" → list_branches(owner, repo)
   - These should return data for display

9. DEFAULT GITHUB LISTING:
   - "List GitHub issues" → ALWAYS use list_issues (not search_issues)
   - "Show details for GitHub issue #N" → ALWAYS use get_issue(issue_number=N)
   - Only use search_issues when explicitly searching with terms

10. JIRA OPERATIONS:
    - "Show my Jira issues" → search with JQL for current project
    - "Mark Jira issue ABC-123 to Done" → list_transitions + transition_issue
    - "Add comment to ABC-123" → add_comment(issue_key="ABC-123", body="...")

11. ENHANCED ERROR HANDLING:
    - For search operations that might return no results, always include helpful message
    - For Jira transitions, always get available transitions first
    - For PR creation, validate all required parameters are present

MESSAGE RULES:
- Keep messages concise: "Found 5 issues", "Retrieved file contents", "Branch created"
- NO markdown tables in message - data shows in UI tables
- Be specific about what was done
- For searches with no results: "No issues found matching your criteria"

PARAMETER MAPPING:
- Always use exact parameter names: issue_number (not issue_id), branch (not branch_name)
- Include owner/repo in GitHub calls when available from context
- For search_issues/search_pull_requests: query is MANDATORY
- For get_file_contents: both path and ref are required
- For Jira transitions: ALWAYS get transitions first, then use correct transition_id

CRITICAL FIXES FOR COMMON FAILURES:

1. SEARCH WITH NO RESULTS:
   - Always perform search even if no results expected
   - Message should indicate "No issues found" rather than generic response

2. PULL REQUEST CREATION:
   - Ensure ALL required parameters: owner, repo, title, head, base
   - Do not skip any required parameters

3. JIRA STATUS CHANGES:
   - MUST use list_transitions first to get available transitions
   - Then use transition_issue with correct transition_id from the list
   - Never guess transition IDs

4. FILE CONTENT RETRIEVAL:
   - Always include both path and ref parameters
   - Default ref to "main" if not specified

5. BRANCH AND TAG LISTING:
   - Use list_branches and list_tags with proper owner/repo
   - Ensure data is returned for display

Remember: Every action must have a clear purpose and proper parameters. Always provide helpful feedback even when operations fail or return no results."""


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
    """Enhance actions with context-specific improvements and intent detection."""
    enhanced_actions = []

    # Extract entities from user message
    issue_number = _extract_issue_number(user_message)
    jira_key = _extract_jira_key(user_message)
    issue_title = _extract_issue_title(user_message)
    status_intent = _detect_status_change_intent(user_message)

    # Check if this is a title-based GitHub issue lookup that needs search first
    github_title_lookup = False
    if issue_title and not issue_number and 'github' in user_message.lower():
        # Look for patterns like "show details for GitHub issue Blue"
        detail_patterns = [
            r'(?:show|get|view)\s+(?:details|info)\s+(?:for\s+)?(?:github\s+)?issue\s+[\'"]?([^\'"\s]+)[\'"]?',
            r'(?:details|info)\s+(?:for\s+)?(?:github\s+)?issue\s+[\'"]?([^\'"\s]+)[\'"]?'
        ]
        for pattern in detail_patterns:
            if re.search(pattern, user_message, re.IGNORECASE):
                github_title_lookup = True
                break

    for action in actions:
        service = action.get("service", "")
        action_name = action.get("action", "")
        args = action.get("args", {})

        # Enhanced GitHub issue handling
        if service == "github":
            # Add default owner/repo if missing
            if 'owner' not in args and gh_owner:
                args['owner'] = gh_owner
            if 'repo' not in args and gh_repo:
                args['repo'] = gh_repo

            # Handle title-based lookups - convert to search first, then get_issue
            if github_title_lookup and action_name == "get_issue" and issue_title:
                # Replace with search action
                search_action = {
                    "service": "github",
                    "action": "search_issues",
                    "args": {
                        "query": f'repo:{gh_owner}/{gh_repo} "{issue_title}" type:issue state:all',
                        "per_page": 10
                    },
                    "description": f"Search for issue titled '{issue_title}'"
                }
                enhanced_actions.append(search_action)
                continue

            # Handle issue details by number - prioritize direct get_issue over search
            if action_name == "search_issues" and issue_number and any(
                    word in user_message.lower() for word in ['detail', 'show', 'info', 'about']):
                # Replace search with direct get_issue call for better results
                action["action"] = "get_issue"
                action["args"] = {"issue_number": issue_number}
                action["description"] = f"Get details for issue #{issue_number}"

            # Handle search_issues - ensure query is present and properly formatted
            elif action_name == "search_issues":
                if "query" not in args or not args["query"]:
                    # Build a proper search query
                    if issue_title:
                        query = f'repo:{gh_owner}/{gh_repo} "{issue_title}" type:issue state:all'
                    else:
                        query = _build_github_search_query(user_message, gh_owner, gh_repo, "issue")
                    args["query"] = query
                    action["args"] = args

            # Handle search_pull_requests - ensure query is present and properly formatted
            elif action_name == "search_pull_requests":
                if "query" not in args or not args["query"]:
                    query = _build_github_search_query(user_message, gh_owner, gh_repo, "pr")
                    args["query"] = query
                    action["args"] = args

            # Handle status changes with proper GitHub routing
            if status_intent and status_intent[0] == "github" and issue_number:
                platform, new_state = status_intent
                if action_name in ["search_issues", "get_issue"]:
                    # Keep the original action, then add status change
                    enhanced_actions.append(action)
                    enhanced_actions.append({
                        "service": "github",
                        "action": "update_issue",
                        "args": {"issue_number": issue_number,
                                 "state": new_state if new_state in ['closed', 'open'] else "closed"},
                        "description": f"{'Close' if new_state in ['closed', 'done'] else 'Reopen'} issue #{issue_number}"
                    })
                    continue

        # Enhanced Jira handling
        elif service == "jira":
            # Handle status changes with proper Jira routing
            if status_intent and status_intent[0] == "jira" and jira_key:
                platform, new_state = status_intent
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

    # Check for greeting first
    if _is_greeting(user_message):
        return {
            "message": "Hi! I'm ready to help. I can list GitHub/Jira issues, show details, create issues/PRs, update statuses, etc.",
            "actions": [],
            "metadata": {
                "execution_time": time.time() - start_time,
                "model": model,
                "attempts": 1,
                "greeting": True
            }
        }

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
    _extract_issue_title.cache_clear()
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
        "extract_issue_title": _extract_issue_title.cache_info()._asdict(),
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