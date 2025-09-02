# agent/graph_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Union
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
import time
import logging

from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, AnyMessage

# Lazy loading for better startup performance
_jira_client = None
_notion_client = None


def get_jira_client():
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


def get_notion_client():
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


# Setup logging
logger = logging.getLogger(__name__)

# Connection pool for clients
_jira_connection_pool: Dict[str, Any] = {}
_notion_connection_pool: Dict[str, Any] = {}


def get_cached_jira_client(base_url: str, email: str, token: str):
    """Get or create cached Jira client for connection reuse."""
    cache_key = f"{base_url}:{email}:{hash(token)}"

    if cache_key not in _jira_connection_pool:
        JiraClient, _ = get_jira_client()
        _jira_connection_pool[cache_key] = JiraClient(base_url, email, token)

    return _jira_connection_pool[cache_key]


def get_cached_notion_client(token: str):
    """Get or create cached Notion client for connection reuse."""
    cache_key = f"notion:{hash(token)}"

    if cache_key not in _notion_connection_pool:
        NotionClient, _ = get_notion_client()
        _notion_connection_pool[cache_key] = NotionClient(token)

    return _notion_connection_pool[cache_key]


# ---------- Enhanced Pydantic schemas with validation ----------
class Action(BaseModel):
    server: str = Field(..., description='Target server: "github", "jira", or "notion"')
    tool: str = Field(..., description="Tool name")
    args: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, description="Execution priority (1-10, higher = more important)")
    depends_on: List[int] = Field(default_factory=list, description="Step dependencies")

    @validator('server')
    def validate_server(cls, v):
        if v.lower() not in ['github', 'jira', 'notion']:
            raise ValueError('Server must be "github", "jira", or "notion"')
        return v.lower()

    @validator('priority')
    def validate_priority(cls, v):
        return max(1, min(10, v))


class Plan(BaseModel):
    actions: List[Action] = Field(default_factory=list)
    estimated_duration: Optional[int] = Field(None, description="Estimated execution time in seconds")
    parallel_groups: List[List[int]] = Field(default_factory=list, description="Actions that can run in parallel")

    def get_execution_order(self) -> List[List[int]]:
        """Calculate optimal execution order considering dependencies and parallelization."""
        if self.parallel_groups:
            return self.parallel_groups

        # Simple dependency-aware ordering
        remaining = set(range(len(self.actions)))
        ordered_groups = []

        while remaining:
            ready = []
            for i in remaining:
                action = self.actions[i]
                if all(dep not in remaining for dep in action.depends_on):
                    ready.append(i)

            if not ready:
                # Break dependency cycles by taking highest priority
                ready = [max(remaining, key=lambda i: self.actions[i].priority)]

            ordered_groups.append(sorted(ready, key=lambda i: -self.actions[i].priority))
            remaining -= set(ready)

        return ordered_groups


# Expanded system prompt with comprehensive GitHub tool surface
SYSTEM_PROMPT = """You are an intelligent task planner. Analyze the user request and return ONLY valid JSON.

Schema:
{
  "actions": [
    {
      "server": "github|jira|notion",
      "tool": "tool_name", 
      "args": {...},
      "priority": 1-10,
      "depends_on": [step_indices]
    }
  ],
  "estimated_duration": seconds,
  "parallel_groups": [[0,1], [2,3]]
}

GitHub Tools (server="github") - Comprehensive surface:
ISSUES: create_issue, update_issue, get_issue, list_issues, search_issues, add_issue_comment
PULL REQUESTS: create_pull_request, update_pull_request, merge_pull_request, list_pull_requests, get_pull_request, get_pull_request_files, get_pull_request_reviews, get_pull_request_status, get_pull_request_comments, get_pull_request_diff, update_pull_request_branch, search_pull_requests
REPOSITORY: create_branch, list_branches, list_commits, get_commit, get_file_contents, create_or_update_file, delete_file, push_files
RELEASES: list_tags, get_tag, list_releases, get_latest_release, get_release_by_tag
WORKFLOWS: list_workflows, list_workflow_runs, get_workflow_run, get_workflow_run_usage, get_workflow_run_logs, get_job_logs, list_workflow_jobs, list_workflow_run_artifacts, download_workflow_run_artifact, run_workflow, rerun_workflow_run, rerun_failed_jobs, cancel_workflow_run
SEARCH/SECURITY: search_code, list_code_scanning_alerts, get_code_scanning_alert, list_dependabot_alerts, get_dependabot_alert, list_secret_scanning_alerts, get_secret_scanning_alert
OTHER: list_notifications, mark_notifications_read, search_users, search_orgs, list_discussions, get_discussion, list_discussion_comments, create_gist, list_gists, update_gist

Jira Tools (server="jira"):
- jira_create_issue(project_key, summary, description?, issuetype_name?)
- jira_add_comment(issue_key, body)
- jira_transition_issue(issue_key, transition_id)
- jira_search(jql, max_results?)
- jira_project_info(project_key)
- jira_whoami()
- jira_list_transitions(issue_key)

Notion Tools (server="notion"):
- notion_create_page(parent_database_id?, parent_page_id?, properties?, children?)
- notion_update_page(page_id, properties?, archived?)
- notion_query_database(database_id, filter_conditions?, sorts?)
- notion_get_database(database_id)
- notion_get_page(page_id)
- notion_search(query, filter_value?)
- notion_create_database(parent_page_id, title, properties)
- notion_append_blocks(block_id, children)

Guidelines:
- Use provided owner/repo/project_key/database_id from context
- Set higher priority (8-10) for critical operations
- Group independent operations for parallel execution
- Add dependencies for sequential operations
- Estimate realistic duration
"""


# ---------- Enhanced LangGraph state with performance tracking ----------
class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    user_message: str
    gh_owner: str
    gh_repo: str
    jira_project_key: str
    notion_database_id: str
    plan: Plan
    results: List[Dict[str, Any]]
    needs_approval: bool
    approved: bool
    execution_start: float
    execution_end: float
    performance_metrics: Dict[str, Any]


# ---------- Optimized nodes with caching and error handling ----------
@lru_cache(maxsize=32)
def get_llm_client(api_key: str, model: str) -> ChatOpenAI:
    """Cached LLM client creation for better performance."""
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=0.2,
        request_timeout=30,
        max_retries=2
    )


def plan_node(state: AgentState, *, openai_api_key: str, model: str) -> AgentState:
    """Enhanced planning node with better prompt engineering and error handling."""
    start_time = time.time()

    # Build context more efficiently
    context_parts = []
    if state.get("gh_owner") and state.get("gh_repo"):
        context_parts.append(f"GitHub: owner={state['gh_owner']} repo={state['gh_repo']}")
    if state.get("jira_project_key"):
        context_parts.append(f"Jira: project={state['jira_project_key']}")
    if state.get("notion_database_id"):
        context_parts.append(f"Notion: database={state['notion_database_id']}")

    context = "\n".join(context_parts) if context_parts else "No specific context provided"

    # Use cached LLM client
    try:
        llm = get_llm_client(openai_api_key, model)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context: {context}\n\nTask: {state['user_message']}\n\nReturn optimized execution plan as JSON:"}
        ]

        response = llm.invoke(prompt)
        output = response.content or "{}"

    except Exception as e:
        logger.error(f"LLM planning failed: {e}")
        output = '{"actions": [], "estimated_duration": 0}'

    # Enhanced JSON parsing with multiple fallback strategies
    plan = parse_plan_response(output)

    state["plan"] = plan

    # Intelligent approval logic - expanded to include more GitHub operations
    write_tools = {
        # Jira write operations
        "jira_create_issue", "jira_add_comment", "jira_transition_issue",
        # GitHub write operations
        "create_issue", "update_issue", "create_pull_request", "update_pull_request", "merge_pull_request",
        "create_or_update_file", "delete_file", "push_files", "create_branch",
        "run_workflow", "cancel_workflow_run", "rerun_workflow_run", "rerun_failed_jobs",
        "update_pull_request_branch", "mark_notifications_read", "create_gist", "update_gist",
        # Notion write operations
        "notion_create_page", "notion_update_page", "notion_create_database", "notion_append_blocks"
    }

    needs_approval = any(
        (action.server in ["jira", "notion"] and action.tool in write_tools) or
        (action.server == "github" and action.tool in write_tools)
        for action in plan.actions
    )

    state["needs_approval"] = needs_approval

    # Track performance
    planning_time = time.time() - start_time
    state["performance_metrics"] = {"planning_time": planning_time}

    return state


def parse_plan_response(output: str) -> Plan:
    """Robust plan parsing with multiple fallback strategies."""
    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(output)
        return Plan(**data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract JSON from markdown or mixed content
    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = output[start:end]
            data = json.loads(json_str)
            return Plan(**data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: Try to fix common JSON issues
    try:
        # Fix common issues like trailing commas, single quotes, etc.
        fixed = output.replace("'", '"').replace(",}", "}").replace(",]", "]")
        data = json.loads(fixed)
        return Plan(**data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 4: Return empty plan
    logger.warning(f"Could not parse plan response: {output[:200]}...")
    return Plan(actions=[], estimated_duration=0)


def approval_gate_node(state: AgentState) -> AgentState:
    """Optimized approval gate with logging."""
    if state.get("needs_approval") and not state.get("approved"):
        logger.info("Plan requires approval - waiting for user confirmation")
        return state

    logger.info("Plan approved or no approval required - proceeding to execution")
    return state


def execute_node(
        state: AgentState,
        *,
        jira_base_url: Optional[str],
        jira_email: Optional[str],
        jira_api_token: Optional[str],
        notion_token: Optional[str],
) -> AgentState:
    """Enhanced execution node with parallel processing and better error handling."""
    if state.get("needs_approval") and not state.get("approved"):
        return state

    start_time = time.time()
    state["execution_start"] = start_time

    plan = state.get("plan", Plan())
    if not plan.actions:
        state["results"] = []
        return state

    # Get clients if needed
    jira_client = None
    notion_client = None

    if jira_base_url and jira_email and jira_api_token:
        try:
            jira_client = get_cached_jira_client(jira_base_url, jira_email, jira_api_token)
        except Exception as e:
            logger.error(f"Failed to create Jira client: {e}")

    if notion_token:
        try:
            notion_client = get_cached_notion_client(notion_token)
        except Exception as e:
            logger.error(f"Failed to create Notion client: {e}")

    # Execute with optimal ordering and parallelization
    execution_groups = plan.get_execution_order()
    results = []

    for group_idx, action_indices in enumerate(execution_groups):
        if len(action_indices) == 1:
            # Single action - execute directly
            result = execute_single_action(
                action_indices[0], plan.actions[action_indices[0]], jira_client, notion_client
            )
            results.append(result)
        else:
            # Multiple actions - execute in parallel
            group_results = execute_parallel_actions(
                [(idx, plan.actions[idx]) for idx in action_indices], jira_client, notion_client
            )
            results.extend(group_results)

    # Sort results by step number
    results.sort(key=lambda x: x["step"])

    end_time = time.time()
    state["execution_end"] = end_time
    state["results"] = results

    # Update performance metrics
    metrics = state.get("performance_metrics", {})
    metrics.update({
        "execution_time": end_time - start_time,
        "total_actions": len(plan.actions),
        "successful_actions": sum(1 for r in results if r["ok"]),
        "parallel_groups": len(execution_groups)
    })
    state["performance_metrics"] = metrics

    return state


def execute_single_action(step_idx: int, action: Action, jira_client: Any, notion_client: Any) -> Dict[str, Any]:
    """Execute a single action with comprehensive error handling."""
    try:
        if action.server == "github":
            return {
                "step": step_idx + 1,
                "server": "github",
                "tool": action.tool,
                "ok": True,
                "deferred": True,
                "message": "Execute with official GitHub MCP server",
                "priority": action.priority
            }

        elif action.server == "jira":
            if not jira_client:
                raise RuntimeError("Jira client not configured")

            result = execute_jira_action(jira_client, action.tool, action.args)
            return {
                "step": step_idx + 1,
                "server": "jira",
                "tool": action.tool,
                "ok": True,
                "result": result,
                "priority": action.priority
            }

        elif action.server == "notion":
            if not notion_client:
                raise RuntimeError("Notion client not configured")

            result = execute_notion_action(notion_client, action.tool, action.args)
            return {
                "step": step_idx + 1,
                "server": "notion",
                "tool": action.tool,
                "ok": True,
                "result": result,
                "priority": action.priority
            }

        else:
            raise ValueError(f"Unknown server: {action.server}")

    except Exception as e:
        logger.error(f"Action {step_idx + 1} failed: {e}")
        return {
            "step": step_idx + 1,
            "server": action.server,
            "tool": action.tool,
            "ok": False,
            "error": str(e),
            "priority": action.priority
        }


def execute_parallel_actions(actions: List[tuple], jira_client: Any, notion_client: Any) -> List[Dict[str, Any]]:
    """Execute multiple actions in parallel for better performance."""

    def execute_wrapper(action_data):
        step_idx, action = action_data
        return execute_single_action(step_idx, action, jira_client, notion_client)

    # Use thread pool for I/O-bound operations
    with ThreadPoolExecutor(max_workers=min(4, len(actions))) as executor:
        return list(executor.map(execute_wrapper, actions))


@lru_cache(maxsize=64)
def get_jira_action_handler(tool_name: str):
    """Cached action handler lookup for better performance."""
    handlers = {
        "jira_create_issue": lambda client, args: client.create_issue(**args),
        "jira_add_comment": lambda client, args: client.add_comment(**args),
        "jira_transition_issue": lambda client, args: client.transition_issue(**args),
        "jira_search": lambda client, args: client.search(**args),
        "jira_project_info": lambda client, args: client.project_info(**args),
        "jira_whoami": lambda client, args: client.whoami(),
        "jira_list_transitions": lambda client, args: client.list_transitions(**args),
    }
    return handlers.get(tool_name)


@lru_cache(maxsize=64)
def get_notion_action_handler(tool_name: str):
    """Cached Notion action handler lookup for better performance."""
    handlers = {
        "notion_create_page": lambda client, args: client.create_page(**args),
        "notion_update_page": lambda client, args: client.update_page(**args),
        "notion_query_database": lambda client, args: client.query_database(**args),
        "notion_get_database": lambda client, args: client.get_database(**args),
        "notion_get_page": lambda client, args: client.get_page(**args),
        "notion_search": lambda client, args: client.search(**args),
        "notion_create_database": lambda client, args: client.create_database(**args),
        "notion_append_blocks": lambda client, args: client.append_block_children(**args),
    }
    return handlers.get(tool_name)


def execute_jira_action(client: Any, tool: str, args: Dict[str, Any]) -> Any:
    """Execute Jira action with optimized handler lookup."""
    handler = get_jira_action_handler(tool)
    if not handler:
        raise ValueError(f"Unknown Jira tool: {tool}")

    return handler(client, args)


def execute_notion_action(client: Any, tool: str, args: Dict[str, Any]) -> Any:
    """Execute Notion action with optimized handler lookup."""
    handler = get_notion_action_handler(tool)
    if not handler:
        raise ValueError(f"Unknown Notion tool: {tool}")

    return handler(client, args)


# ---------- Optimized graph builder with caching ----------
@lru_cache(maxsize=8)
def build_graph_cached(openai_api_key: str, model: str) -> Any:
    """Cached graph compilation for better performance."""

    def plan_wrapper(state):
        return plan_node(state, openai_api_key=openai_api_key, model=model)

    g = StateGraph(AgentState)
    g.add_node("plan", plan_wrapper)
    g.add_node("approval", approval_gate_node)
    g.add_node("execute",
               lambda s: execute_node(s, jira_base_url=None, jira_email=None, jira_api_token=None, notion_token=None))

    g.set_entry_point("plan")
    g.add_edge("plan", "approval")
    g.add_edge("approval", "execute")
    g.add_edge("execute", END)

    return g.compile()


def build_graph(*, openai_api_key: str, model: str,
                jira_base_url: Optional[str], jira_email: Optional[str], jira_api_token: Optional[str],
                notion_token: Optional[str]):
    """Build execution graph with Jira and Notion configuration."""

    def execute_wrapper(state):
        return execute_node(
            state,
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            notion_token=notion_token
        )

    def plan_wrapper(state):
        return plan_node(state, openai_api_key=openai_api_key, model=model)

    g = StateGraph(AgentState)
    g.add_node("plan", plan_wrapper)
    g.add_node("approval", approval_gate_node)
    g.add_node("execute", execute_wrapper)

    g.set_entry_point("plan")
    g.add_edge("plan", "approval")
    g.add_edge("approval", "execute")
    g.add_edge("execute", END)

    return g.compile()


# ---------- Enhanced public API with performance optimizations ----------
def plan_with_langchain(
        *, openai_api_key: str, model: str, user_message: str,
        gh_owner: Optional[str], gh_repo: Optional[str], jira_project_key: Optional[str],
        notion_database_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate plan with LangChain - optimized for performance."""
    state: AgentState = {
        "user_message": user_message,
        "gh_owner": gh_owner or "",
        "gh_repo": gh_repo or "",
        "jira_project_key": jira_project_key or "",
        "notion_database_id": notion_database_id or "",
    }

    # Use cached graph for planning-only operations
    try:
        graph = build_graph_cached(openai_api_key, model)
        result = graph.invoke(state)

        plan = result.get("plan")
        if isinstance(plan, Plan):
            plan_dict = json.loads(plan.model_dump_json())
            # Add performance metrics if available
            if "performance_metrics" in result:
                plan_dict["performance_metrics"] = result["performance_metrics"]
            return plan_dict

        return {"actions": []}

    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {"actions": [], "error": str(e)}


def run_workflow_graph(
        *,
        openai_api_key: str, model: str, user_message: str,
        gh_owner: Optional[str], gh_repo: Optional[str], jira_project_key: Optional[str],
        notion_database_id: Optional[str], approved: bool,
        github_token: Optional[str],  # not used but kept for compatibility
        jira_base_url: Optional[str], jira_email: Optional[str], jira_api_token: Optional[str],
        notion_token: Optional[str],
) -> Dict[str, Any]:
    """Run complete workflow with performance optimization and comprehensive reporting."""
    state: AgentState = {
        "user_message": user_message,
        "gh_owner": gh_owner or "",
        "gh_repo": gh_repo or "",
        "jira_project_key": jira_project_key or "",
        "notion_database_id": notion_database_id or "",
        "approved": approved,
    }

    try:
        graph = build_graph(
            openai_api_key=openai_api_key,
            model=model,
            jira_base_url=jira_base_url,
            jira_email=jira_email,
            jira_api_token=jira_api_token,
            notion_token=notion_token,
        )

        result = graph.invoke(state)

        # Build comprehensive response
        response = {
            "needs_approval": bool(result.get("needs_approval")),
            "approved": bool(result.get("approved")),
            "results": result.get("results", [])
        }

        # Include plan details
        plan = result.get("plan")
        if isinstance(plan, Plan):
            response["plan"] = json.loads(plan.model_dump_json())
        else:
            response["plan"] = {"actions": []}

        # Include performance metrics
        if "performance_metrics" in result:
            response["performance_metrics"] = result["performance_metrics"]

        # Calculate summary statistics
        results = result.get("results", [])
        response["summary"] = {
            "total_actions": len(results),
            "successful": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
            "deferred": sum(1 for r in results if r.get("deferred"))
        }

        return response

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {
            "error": str(e),
            "needs_approval": False,
            "approved": False,
            "plan": {"actions": []},
            "results": [],
            "summary": {"total_actions": 0, "successful": 0, "failed": 1, "deferred": 0}
        }


# ---------- Cleanup utilities ----------
def clear_caches():
    """Clear all caches for memory management."""
    get_llm_client.cache_clear()
    get_jira_action_handler.cache_clear()
    get_notion_action_handler.cache_clear()
    build_graph_cached.cache_clear()
    _jira_connection_pool.clear()
    _notion_connection_pool.clear()


def get_cache_info():
    """Get cache statistics for monitoring."""
    return {
        "llm_client_cache": get_llm_client.cache_info()._asdict(),
        "jira_action_handler_cache": get_jira_action_handler.cache_info()._asdict(),
        "notion_action_handler_cache": get_notion_action_handler.cache_info()._asdict(),
        "graph_cache": build_graph_cached.cache_info()._asdict(),
        "jira_connections": len(_jira_connection_pool),
        "notion_connections": len(_notion_connection_pool)
    }