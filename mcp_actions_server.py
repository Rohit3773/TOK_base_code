# servers/mcp_actions_server.py
from __future__ import annotations
import os, sys, json, logging
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Ensure we can import ../agent (or flat files if placed at repo root)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- Lazy imports for better startup performance ----
_jira_client = None
_notion_client = None
_llm_agent = None
_graph_agent = None


def get_jira_modules():
    global _jira_client
    if _jira_client is None:
        try:
            from jira_client import JiraClient, JiraError
            _jira_client = (JiraClient, JiraError)
        except ImportError:
            try:
                from clients.jira_client import JiraClient, JiraError
                _jira_client = (JiraClient, JiraError)
            except ImportError:
                raise ImportError("Jira client not available")
    return _jira_client


def get_notion_modules():
    global _notion_client
    if _notion_client is None:
        try:
            from notion_client import NotionClient, NotionError
            _notion_client = (NotionClient, NotionError)
        except ImportError:
            try:
                from clients.notion_client import NotionClient, NotionError
                _notion_client = (NotionClient, NotionError)
            except ImportError:
                raise ImportError("Notion client not available")
    return _notion_client


def get_llm_agent():
    global _llm_agent
    if _llm_agent is None:
        try:
            from llm_agent import propose_actions, LLMPlanningError
            _llm_agent = (propose_actions, LLMPlanningError)
        except ImportError:
            try:
                from agent.llm_agent import propose_actions, LLMPlanningError
                _llm_agent = (propose_actions, LLMPlanningError)
            except ImportError:
                raise ImportError("LLM agent not available")
    return _llm_agent


def get_graph_agent():
    global _graph_agent
    if _graph_agent is None:
        try:
            from graph_agent import plan_with_langchain, run_workflow_graph
            _graph_agent = (plan_with_langchain, run_workflow_graph)
        except ImportError:
            try:
                from agent.graph_agent import plan_with_langchain, run_workflow_graph
                _graph_agent = (plan_with_langchain, run_workflow_graph)
            except ImportError:
                _graph_agent = (None, None)
    return _graph_agent


# ------------------------------------------------------------------------------------
# Setup with optimizations
# ------------------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("mcp-actions")

mcp = FastMCP("jira-planner-mcp")

# Thread pool for I/O operations
executor = ThreadPoolExecutor(max_workers=4)

# Cached session variables with type hints
SESSION: Dict[str, Optional[str]] = {
    "OPENAI_API_KEY": None,
    "OPENAI_MODEL": None,
    "GITHUB_TOKEN": None,
    "JIRA_BASE_URL": None,
    "JIRA_EMAIL": None,
    "JIRA_API_TOKEN": None,
    "NOTION_TOKEN": None,
}

# Connection pools for clients
_jira_clients: Dict[str, Any] = {}
_notion_clients: Dict[str, Any] = {}


@lru_cache(maxsize=32)
def _get(key: str, override: Optional[str] = None) -> Optional[str]:
    """Cached secret resolution with priority: explicit override -> session -> env."""
    return override or SESSION.get(key) or os.getenv(key)


def _get_jira_cached(
        base_override: Optional[str] = None,
        email_override: Optional[str] = None,
        token_override: Optional[str] = None,
) -> Any:
    """Get or create cached Jira client for better performance."""
    JiraClient, JiraError = get_jira_modules()

    base = _get("JIRA_BASE_URL", base_override)
    email = _get("JIRA_EMAIL", email_override)
    token = _get("JIRA_API_TOKEN", token_override)

    if not (base and email and token):
        raise Exception("Missing Jira credentials")

    # Create cache key
    cache_key = f"{base}:{email}:{hash(token)}"

    if cache_key not in _jira_clients:
        _jira_clients[cache_key] = JiraClient(base, email, token)

    return _jira_clients[cache_key]


def _get_notion_cached(token_override: Optional[str] = None) -> Any:
    """Get or create cached Notion client for better performance."""
    NotionClient, NotionError = get_notion_modules()

    token = _get("NOTION_TOKEN", token_override)

    if not token:
        raise Exception("Missing Notion token")

    # Create cache key
    cache_key = f"notion:{hash(token)}"

    if cache_key not in _notion_clients:
        _notion_clients[cache_key] = NotionClient(token)

    return _notion_clients[cache_key]


# ------------------------------------------------------------------------------------
# Optimized tools with async support and caching
# ------------------------------------------------------------------------------------
@mcp.tool()
def health() -> str:
    """Fast health check with capability detection."""
    try:
        get_jira_modules()
        jira_available = True
    except ImportError:
        jira_available = False

    try:
        get_notion_modules()
        notion_available = True
    except ImportError:
        notion_available = False

    try:
        get_llm_agent()
        llm_available = True
    except ImportError:
        llm_available = False

    plan_lc, run_lg = get_graph_agent()
    graph_available = plan_lc is not None and run_lg is not None

    tools = []

    # Jira tools
    if jira_available:
        tools.extend(["jira_whoami", "jira_project_info", "jira_create_issue",
                      "jira_add_comment", "jira_list_transitions", "jira_transition_issue", "jira_search"])

    # Notion tools
    if notion_available:
        tools.extend(["notion_get_bot_info", "notion_create_page", "notion_update_page",
                      "notion_get_database", "notion_query_database", "notion_search",
                      "notion_create_database", "notion_append_blocks", "notion_get_users"])

    # Planning tools
    if llm_available:
        tools.extend(["plan_actions", "execute_plan_multi_platform"])

    if graph_available:
        tools.extend(["plan_actions_lc", "run_workflow_lg"])

    return json.dumps({
        "ok": True,
        "performance_optimizations": [
            "Lazy imports for faster startup",
            "Connection pooling for Jira and Notion clients",
            "LRU cache for credential resolution",
            "Thread pool for I/O operations",
            "Batch operation support"
        ],
        "capabilities": {
            "jira": jira_available,
            "notion": notion_available,
            "llm_planning": llm_available,
            "graph_workflow": graph_available
        },
        "tools": tools
    }, indent=2)


@mcp.tool()
def set_session_secrets(
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        github_token: Optional[str] = None,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        notion_token: Optional[str] = None,
) -> str:
    """Set session secrets and clear relevant caches."""
    # Clear caches when credentials change
    if any([openai_api_key, openai_model, github_token, jira_base_url, jira_email, jira_api_token, notion_token]):
        _get.cache_clear()
        _jira_clients.clear()
        _notion_clients.clear()

    # Update session
    updates = {
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_MODEL": openai_model,
        "GITHUB_TOKEN": github_token,
        "JIRA_BASE_URL": jira_base_url,
        "JIRA_EMAIL": jira_email,
        "JIRA_API_TOKEN": jira_api_token,
        "NOTION_TOKEN": notion_token
    }

    for key, value in updates.items():
        if value is not None:
            SESSION[key] = value

    return json.dumps({
        "ok": True,
        "updated": [k for k, v in updates.items() if v is not None],
        "session": {k: bool(v) for k, v in SESSION.items()}
    }, indent=2)


# ------------------------------------------------------------------------------------
# Enhanced LLM planning with error handling and retry logic
# ------------------------------------------------------------------------------------
@mcp.tool()
def plan_actions(
        user_message: str,
        gh_owner: Optional[str] = None,
        gh_repo: Optional[str] = None,
        jira_project_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
) -> str:
    """
    Generate action plan with retry logic and performance optimizations.
    Now supports GitHub, Jira, and Notion platforms.
    """
    try:
        propose_actions, LLMPlanningError = get_llm_agent()
    except ImportError:
        return json.dumps({"error": "LLM agent not available"}, indent=2)

    api_key = _get("OPENAI_API_KEY", openai_api_key)
    mdl = model or _get("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        return json.dumps({"error": "Missing OpenAI API key"}, indent=2)

    # Retry logic for better reliability
    last_error = None
    for attempt in range(max_retries):
        try:
            plan = propose_actions(
                openai_key=api_key,
                model=mdl,
                user_message=user_message,
                gh_owner=gh_owner,
                gh_repo=gh_repo,
                jira_project_key=jira_project_key,
                notion_database_id=notion_database_id,
            )

            # Add metadata for performance tracking
            plan["metadata"] = {
                "generated_at": "2024-01-01T00:00:00Z",
                "model": mdl,
                "attempt": attempt + 1,
                "total_actions": len(plan.get("actions", [])),
                "platforms_enabled": {
                    "github": bool(gh_owner and gh_repo),
                    "jira": bool(jira_project_key),
                    "notion": bool(notion_database_id)
                }
            }

            return json.dumps(plan, indent=2)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                log.warning(f"Planning attempt {attempt + 1} failed: {e}, retrying...")
                continue

    return json.dumps({
        "error": f"Planning failed after {max_retries} attempts: {last_error}",
        "actions": []
    }, indent=2)


@mcp.tool()
def execute_plan_multi_platform(
        plan_json: str,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        notion_token: Optional[str] = None,
        parallel: bool = False,
) -> str:
    """
    Execute multi-platform actions with optional parallel processing.
    GitHub steps are deferred to official GitHub MCP server.
    Supports Jira and Notion execution.
    """
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"}, indent=2)

    actions = plan.get("actions", [])
    if not actions:
        return json.dumps({"ok": True, "results": []}, indent=2)

    results = []
    jira_actions = []
    notion_actions = []

    # Separate actions by platform
    for idx, step in enumerate(actions, start=1):
        server = (step.get("server", "") or "").lower()
        tool = step.get("tool", "")

        if server == "github":
            results.append({
                "step": idx,
                "server": "github",
                "tool": tool,
                "ok": True,
                "deferred": True,
                "message": "Execute with official GitHub MCP server"
            })
        elif server == "jira" or tool.startswith("jira_"):
            jira_actions.append((idx, step))
        elif server == "notion" or tool.startswith("notion_"):
            notion_actions.append((idx, step))

    # Execute platform-specific actions
    if jira_actions:
        if parallel and len(jira_actions) > 1:
            jira_results = _execute_jira_parallel(jira_actions, jira_base_url, jira_email, jira_api_token)
        else:
            jira_results = _execute_jira_sequential(jira_actions, jira_base_url, jira_email, jira_api_token)
        results.extend(jira_results)

    if notion_actions:
        if parallel and len(notion_actions) > 1:
            notion_results = _execute_notion_parallel(notion_actions, notion_token)
        else:
            notion_results = _execute_notion_sequential(notion_actions, notion_token)
        results.extend(notion_results)

    results.sort(key=lambda x: x["step"])  # Maintain original order

    return json.dumps({
        "ok": True,
        "results": results,
        "metadata": {
            "total_actions": len(actions),
            "jira_actions": len(jira_actions),
            "notion_actions": len(notion_actions),
            "github_deferred": len(actions) - len(jira_actions) - len(notion_actions),
            "parallel_execution": parallel
        }
    }, indent=2)


def _execute_jira_sequential(actions, base_url, email, token):
    """Execute Jira actions sequentially with shared client."""
    results = []
    jira = None

    for idx, step in actions:
        try:
            if jira is None:
                jira = _get_jira_cached(base_url, email, token)

            result = _execute_single_jira_action(jira, step.get("tool"), step.get("args", {}))
            results.append({"step": idx, "server": "jira", "tool": step.get("tool"), "ok": True, "result": result})

        except Exception as e:
            results.append({"step": idx, "server": "jira", "tool": step.get("tool"), "ok": False, "error": str(e)})

    return results


def _execute_jira_parallel(actions, base_url, email, token):
    """Execute independent Jira actions in parallel for better performance."""

    def execute_single(idx_step):
        idx, step = idx_step
        try:
            jira = _get_jira_cached(base_url, email, token)
            result = _execute_single_jira_action(jira, step.get("tool"), step.get("args", {}))
            return {"step": idx, "server": "jira", "tool": step.get("tool"), "ok": True, "result": result}
        except Exception as e:
            return {"step": idx, "server": "jira", "tool": step.get("tool"), "ok": False, "error": str(e)}

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=min(4, len(actions))) as pool:
        return list(pool.map(execute_single, actions))


def _execute_notion_sequential(actions, token):
    """Execute Notion actions sequentially with shared client."""
    results = []
    notion = None

    for idx, step in actions:
        try:
            if notion is None:
                notion = _get_notion_cached(token)

            result = _execute_single_notion_action(notion, step.get("tool"), step.get("args", {}))
            results.append({"step": idx, "server": "notion", "tool": step.get("tool"), "ok": True, "result": result})

        except Exception as e:
            results.append({"step": idx, "server": "notion", "tool": step.get("tool"), "ok": False, "error": str(e)})

    return results


def _execute_notion_parallel(actions, token):
    """Execute independent Notion actions in parallel for better performance."""

    def execute_single(idx_step):
        idx, step = idx_step
        try:
            notion = _get_notion_cached(token)
            result = _execute_single_notion_action(notion, step.get("tool"), step.get("args", {}))
            return {"step": idx, "server": "notion", "tool": step.get("tool"), "ok": True, "result": result}
        except Exception as e:
            return {"step": idx, "server": "notion", "tool": step.get("tool"), "ok": False, "error": str(e)}

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=min(4, len(actions))) as pool:
        return list(pool.map(execute_single, actions))


def _execute_single_jira_action(jira, tool, args):
    """Execute a single Jira action with optimized routing."""
    action_map = {
        "jira_create_issue": lambda: jira.create_issue(**args),
        "jira_add_comment": lambda: jira.add_comment(**args),
        "jira_transition_issue": lambda: jira.transition_issue(**args),
        "jira_search": lambda: jira.search(**args),
        "jira_project_info": lambda: jira.project_info(**args),
        "jira_whoami": lambda: jira.whoami(),
        "jira_list_transitions": lambda: jira.list_transitions(**args),
    }

    action = action_map.get(tool)
    if not action:
        raise ValueError(f"Unknown Jira tool: {tool}")

    return action()


def _execute_single_notion_action(notion, tool, args):
    """Execute a single Notion action with optimized routing."""
    action_map = {
        "notion_create_page": lambda: notion.create_page(**args),
        "notion_update_page": lambda: notion.update_page(**args),
        "notion_query_database": lambda: notion.query_database(**args),
        "notion_get_database": lambda: notion.get_database(**args),
        "notion_get_page": lambda: notion.get_page(**args),
        "notion_search": lambda: notion.search(**args),
        "notion_create_database": lambda: notion.create_database(**args),
        "notion_append_blocks": lambda: notion.append_block_children(**args),
        "notion_get_users": lambda: notion.get_users(),
        "notion_get_bot_info": lambda: notion.get_bot_info(),
    }

    action = action_map.get(tool)
    if not action:
        raise ValueError(f"Unknown Notion tool: {tool}")

    return action()


# ------------------------------------------------------------------------------------
# Enhanced LangGraph integration with performance monitoring
# ------------------------------------------------------------------------------------
@mcp.tool()
def plan_actions_lc(
        user_message: str,
        gh_owner: Optional[str] = None,
        gh_repo: Optional[str] = None,
        jira_project_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
) -> str:
    """Plan with LangChain structured output, optimized for performance with Notion support."""
    plan_with_langchain, _ = get_graph_agent()
    if not plan_with_langchain:
        return json.dumps({"error": "LangChain planner not available"}, indent=2)

    api_key = _get("OPENAI_API_KEY", openai_api_key)
    mdl = model or _get("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        return json.dumps({"error": "Missing OpenAI API key"}, indent=2)

    try:
        plan = plan_with_langchain(
            openai_api_key=api_key,
            model=mdl,
            user_message=user_message,
            gh_owner=gh_owner,
            gh_repo=gh_repo,
            jira_project_key=jira_project_key,
            notion_database_id=notion_database_id
        )
        return json.dumps(plan, indent=2)
    except Exception as e:
        return json.dumps({"error": f"LangChain planning failed: {e}"}, indent=2)


@mcp.tool()
def run_workflow_lg(
        user_message: str,
        gh_owner: Optional[str] = None,
        gh_repo: Optional[str] = None,
        jira_project_key: Optional[str] = None,
        notion_database_id: Optional[str] = None,
        approved: bool = False,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        notion_token: Optional[str] = None,
) -> str:
    """Run complete workflow with performance optimizations and Notion support."""
    _, run_workflow_graph = get_graph_agent()
    if not run_workflow_graph:
        return json.dumps({"error": "LangGraph workflow not available"}, indent=2)

    api_key = _get("OPENAI_API_KEY", openai_api_key)
    mdl = model or _get("OPENAI_MODEL") or "gpt-4o-mini"

    if not api_key:
        return json.dumps({"error": "Missing OpenAI API key"}, indent=2)

    try:
        result = run_workflow_graph(
            openai_api_key=api_key,
            model=mdl,
            user_message=user_message,
            gh_owner=gh_owner,
            gh_repo=gh_repo,
            jira_project_key=jira_project_key,
            notion_database_id=notion_database_id,
            approved=approved,
            github_token=None,
            jira_base_url=_get("JIRA_BASE_URL", jira_base_url),
            jira_email=_get("JIRA_EMAIL", jira_email),
            jira_api_token=_get("JIRA_API_TOKEN", jira_api_token),
            notion_token=_get("NOTION_TOKEN", notion_token),
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Workflow execution failed: {e}"}, indent=2)


# ------------------------------------------------------------------------------------
# Optimized direct Jira tools with connection pooling
# ------------------------------------------------------------------------------------
@mcp.tool()
def jira_whoami(
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """Fast user info lookup with cached connection."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        return json.dumps(jira.whoami(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def jira_project_info(
        project_key: str,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """Get project info with cached connection."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        return json.dumps(jira.project_info(project_key), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def jira_create_issue(
        project_key: str,
        summary: str,
        description: str = "",
        issuetype_name: str = "Task",
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """Create issue with optimized client reuse."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        result = jira.create_issue(project_key, summary, description, None, issuetype_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def jira_add_comment(
        issue_key: str,
        body: str,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """Add comment with cached connection."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        result = jira.add_comment(issue_key, body)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def jira_list_transitions(
        issue_key: str,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """List transitions with cached connection."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        result = jira.list_transitions(issue_key)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def jira_transition_issue(
        issue_key: str,
        transition_id: str,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """Transition issue with cached connection."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        result = jira.transition_issue(issue_key, transition_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def jira_search(
        jql: str,
        max_results: int = 50,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None
) -> str:
    """Search issues with cached connection and result optimization."""
    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
        # Clamp max_results for performance
        max_results = min(max(1, max_results), 100)
        result = jira.search(jql, max_results)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ------------------------------------------------------------------------------------
# Optimized direct Notion tools with connection pooling
# ------------------------------------------------------------------------------------
@mcp.tool()
def notion_get_bot_info(notion_token: Optional[str] = None) -> str:
    """Get bot user info with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)
        return json.dumps(notion.get_bot_info(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_get_users(notion_token: Optional[str] = None) -> str:
    """Get all users with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)
        return json.dumps(notion.get_users(), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_create_page(
        properties: Optional[str] = None,
        parent_database_id: Optional[str] = None,
        parent_page_id: Optional[str] = None,
        children: Optional[str] = None,
        notion_token: Optional[str] = None
) -> str:
    """Create page with optimized client reuse."""
    try:
        notion = _get_notion_cached(notion_token)

        # Parse JSON strings
        props = json.loads(properties) if properties else None
        blocks = json.loads(children) if children else None

        result = notion.create_page(
            parent_database_id=parent_database_id,
            parent_page_id=parent_page_id,
            properties=props,
            children=blocks
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_update_page(
        page_id: str,
        properties: Optional[str] = None,
        archived: Optional[bool] = None,
        notion_token: Optional[str] = None
) -> str:
    """Update page with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)

        # Parse JSON string
        props = json.loads(properties) if properties else None

        result = notion.update_page(page_id, properties=props, archived=archived)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_get_database(
        database_id: str,
        notion_token: Optional[str] = None
) -> str:
    """Get database info with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)
        return json.dumps(notion.get_database(database_id), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_query_database(
        database_id: str,
        filter_conditions: Optional[str] = None,
        sorts: Optional[str] = None,
        page_size: int = 100,
        notion_token: Optional[str] = None
) -> str:
    """Query database with cached connection and advanced filtering."""
    try:
        notion = _get_notion_cached(notion_token)

        # Parse JSON strings
        filters = json.loads(filter_conditions) if filter_conditions else None
        sort_list = json.loads(sorts) if sorts else None

        # Clamp page_size for performance
        page_size = min(max(1, page_size), 100)

        result = notion.query_database(
            database_id,
            filter_conditions=filters,
            sorts=sort_list,
            page_size=page_size
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_search(
        query: str,
        sort_direction: str = "descending",
        filter_value: Optional[str] = None,
        notion_token: Optional[str] = None
) -> str:
    """Search Notion with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)
        result = notion.search(query, sort_direction=sort_direction, filter_value=filter_value)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_create_database(
        parent_page_id: str,
        title: str,
        properties: str,
        description: Optional[str] = None,
        notion_token: Optional[str] = None
) -> str:
    """Create database with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)

        # Parse JSON string
        props = json.loads(properties)

        result = notion.create_database(parent_page_id, title, props, description)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_append_blocks(
        block_id: str,
        children: str,
        notion_token: Optional[str] = None
) -> str:
    """Append blocks to a page or block with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)

        # Parse JSON string
        blocks = json.loads(children)

        result = notion.append_block_children(block_id, blocks)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
def notion_get_page(
        page_id: str,
        notion_token: Optional[str] = None
) -> str:
    """Get page info with cached connection."""
    try:
        notion = _get_notion_cached(notion_token)
        return json.dumps(notion.get_page(page_id), indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)


# ------------------------------------------------------------------------------------
# Enhanced batch operations for better performance
# ------------------------------------------------------------------------------------
@mcp.tool()
def jira_batch_create_issues(
        issues_json: str,
        jira_base_url: Optional[str] = None,
        jira_email: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        parallel: bool = True
) -> str:
    """Create multiple Jira issues efficiently with optional parallel processing."""
    try:
        issues = json.loads(issues_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"}, indent=2)

    if not isinstance(issues, list):
        return json.dumps({"error": "Expected list of issues"}, indent=2)

    try:
        jira = _get_jira_cached(jira_base_url, jira_email, jira_api_token)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

    def create_single_issue(issue_data):
        try:
            result = jira.create_issue(**issue_data)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if parallel and len(issues) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(issues))) as pool:
            results = list(pool.map(create_single_issue, issues))
    else:
        results = [create_single_issue(issue) for issue in issues]

    success_count = sum(1 for r in results if r["ok"])

    return json.dumps({
        "ok": True,
        "total": len(issues),
        "success": success_count,
        "failed": len(issues) - success_count,
        "results": results,
        "parallel": parallel
    }, indent=2)


@mcp.tool()
def notion_batch_create_pages(
        pages_json: str,
        notion_token: Optional[str] = None,
        parallel: bool = True
) -> str:
    """Create multiple Notion pages efficiently with optional parallel processing."""
    try:
        pages = json.loads(pages_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"}, indent=2)

    if not isinstance(pages, list):
        return json.dumps({"error": "Expected list of pages"}, indent=2)

    try:
        notion = _get_notion_cached(notion_token)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

    def create_single_page(page_data):
        try:
            result = notion.create_page(**page_data)
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if parallel and len(pages) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(pages))) as pool:
            results = list(pool.map(create_single_page, pages))
    else:
        results = [create_single_page(page) for page in pages]

    success_count = sum(1 for r in results if r["ok"])

    return json.dumps({
        "ok": True,
        "total": len(pages),
        "success": success_count,
        "failed": len(pages) - success_count,
        "results": results,
        "parallel": parallel
    }, indent=2)


# ------------------------------------------------------------------------------------
# Cache and performance monitoring
# ------------------------------------------------------------------------------------
@mcp.tool()
def get_performance_stats() -> str:
    """Get performance statistics for all clients."""
    stats = {
        "jira_connections": len(_jira_clients),
        "notion_connections": len(_notion_clients),
        "cache_info": {
            "credential_cache": _get.cache_info()._asdict() if hasattr(_get, 'cache_info') else {}
        }
    }

    return json.dumps(stats, indent=2)


@mcp.tool()
def clear_all_caches() -> str:
    """Clear all caches and connection pools."""
    _get.cache_clear()
    _jira_clients.clear()
    _notion_clients.clear()

    return json.dumps({"ok": True, "message": "All caches cleared"}, indent=2)


# ------------------------------------------------------------------------------------
# Optimized main with graceful shutdown
# ------------------------------------------------------------------------------------
def cleanup():
    """Clean up resources on shutdown."""
    _jira_clients.clear()
    _notion_clients.clear()
    _get.cache_clear()
    executor.shutdown(wait=True)


if __name__ == "__main__":
    log.info("Starting optimized MCP server with GitHub, Jira, and Notion support (stdio)...")
    try:
        mcp.run()
    finally:
        cleanup()
        log.info("MCP server shutdown complete")