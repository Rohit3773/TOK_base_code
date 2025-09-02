# streamlit_app.py
import streamlit as st
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import traceback
import re
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules with comprehensive error handling
try:
    from github_client import GitHubClient, GitHubError, GitHubConfig
    from jira_client import JiraClient, JiraError, JiraConfig
    from notion_client import NotionClient, NotionError, NotionConfig
    from llm_agent import propose_actions, clear_caches, get_cache_stats, performance_monitor

    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Core import error: {e}")
    IMPORTS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Multi-Platform Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }

    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }

    .assistant-message {
        background: #f1f8e9;
        border-left: 4px solid #4caf50;
        margin-right: 2rem;
    }

    .success-execution {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
    }

    .error-execution {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 4px;
    }

    .status-indicator {
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }

    .status-connected {
        background: #d4edda;
        color: #155724;
    }

    .status-disconnected {
        background: #f8d7da;
        color: #721c24;
    }

    .results-table {
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state with required variables."""
    defaults = {
        'github_client': None,
        'jira_client': None,
        'notion_client': None,
        'chat_history': [],
        'openai_key': '',
        'gh_owner': '',
        'gh_repo': '',
        'jira_project': '',
        'notion_database_id': '',
        'current_input': ''
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_openai_client():
    """Get OpenAI client with error handling."""
    try:
        import openai
        return openai.OpenAI(api_key=st.session_state.openai_key)
    except ImportError:
        st.error("OpenAI library not installed")
        return None
    except Exception as e:
        st.error(f"OpenAI client error: {e}")
        return None


def initialize_clients():
    """Initialize clients with proper error handling."""
    success_messages = []
    error_messages = []

    # Clear existing clients
    st.session_state.github_client = None
    st.session_state.jira_client = None
    st.session_state.notion_client = None

    # Get values from sidebar
    openai_key = st.session_state.get('sidebar_openai_key', '')
    github_token = st.session_state.get('sidebar_github_token', '')
    gh_owner = st.session_state.get('sidebar_gh_owner', '')
    gh_repo = st.session_state.get('sidebar_gh_repo', '')
    jira_url = st.session_state.get('sidebar_jira_url', '')
    jira_email = st.session_state.get('sidebar_jira_email', '')
    jira_token = st.session_state.get('sidebar_jira_token', '')
    jira_project = st.session_state.get('sidebar_jira_project', '')
    notion_token = st.session_state.get('sidebar_notion_token', '')
    notion_database_id = st.session_state.get('sidebar_notion_database_id', '')

    # Update session state
    st.session_state.openai_key = openai_key
    st.session_state.gh_owner = gh_owner
    st.session_state.gh_repo = gh_repo
    st.session_state.jira_project = jira_project
    st.session_state.notion_database_id = notion_database_id

    # Initialize GitHub client
    if github_token and gh_owner and gh_repo:
        try:
            config = GitHubConfig(enable_caching=True, max_retries=3, timeout=30)
            client = GitHubClient(github_token, config, default_owner=gh_owner, default_repo=gh_repo)
            # Test connection
            user_info = client.get_user()
            st.session_state.github_client = client
            success_messages.append(f"GitHub: Connected as {user_info.get('login')}")
        except Exception as e:
            error_messages.append(f"GitHub: {str(e)[:100]}")

    # Initialize Jira client
    if jira_url and jira_email and jira_token:
        try:
            config = JiraConfig(enable_caching=True, max_retries=3, timeout=30)
            client = JiraClient(jira_url, jira_email, jira_token, config)
            # Test connection
            user_info = client.whoami()
            st.session_state.jira_client = client
            success_messages.append(f"Jira: Connected as {user_info.get('displayName')}")
        except Exception as e:
            error_messages.append(f"Jira: {str(e)[:100]}")

    # Initialize Notion client
    if notion_token:
        try:
            config = NotionConfig(enable_caching=True, max_retries=3, timeout=30)
            client = NotionClient(notion_token, config)
            # Test connection
            bot_info = client.get_bot_info()
            st.session_state.notion_client = client
            success_messages.append(f"Notion: Connected as {bot_info.get('name', 'Bot')}")
        except Exception as e:
            error_messages.append(f"Notion: {str(e)[:100]}")

    # Display results
    if success_messages:
        st.success("✅ " + " | ".join(success_messages))
    if error_messages:
        st.error("❌ " + " | ".join(error_messages))


def execute_github_action(client, tool: str, args: dict) -> dict:
    """Execute GitHub action."""
    try:
        # Add default owner/repo if not provided
        if 'owner' not in args and st.session_state.gh_owner:
            args['owner'] = st.session_state.gh_owner
        if 'repo' not in args and st.session_state.gh_repo:
            args['repo'] = st.session_state.gh_repo

        # Clean tool name
        if tool.startswith('gh_'):
            tool = tool[3:]

        # Handle parameter mapping for GitHub
        if tool == 'create_issue':
            # Map common parameter names
            if 'content' in args:
                args['body'] = args.pop('content')

            # Clean up parameters - only keep valid GitHub API parameters
            valid_params = {'owner', 'repo', 'title', 'body', 'assignees', 'milestone', 'labels'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'create_pull_request':
            # Map common parameter names
            if 'content' in args:
                args['body'] = args.pop('content')

            valid_params = {'owner', 'repo', 'title', 'body', 'head', 'base', 'draft'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'search_issues':
            # Ensure query parameter is present
            if 'query' not in args:
                return {'error': 'GitHub search_issues requires query parameter'}

        elif tool == 'search_pull_requests':
            # Ensure query parameter is present
            if 'query' not in args:
                return {'error': 'GitHub search_pull_requests requires query parameter'}

        action_map = {
            # Issues
            'create_issue': lambda: client.create_issue(**args),
            'update_issue': lambda: client.update_issue(**args),
            'get_issue': lambda: client.get_issue(**args),
            'list_issues': lambda: client.list_issues(**args),
            'list_open_issues': lambda: client.list_issues(state="open", **args),
            'search_issues': lambda: client.search_issues(**args),
            'add_issue_comment': lambda: client.add_issue_comment(**args),

            # Pull Requests
            'create_pull_request': lambda: client.create_pull_request(**args),
            'open_pull_request': lambda: client.create_pull_request(**args),
            'update_pull_request': lambda: client.update_pull_request(**args),
            'merge_pull_request': lambda: client.merge_pull_request(**args),
            'list_pull_requests': lambda: client.list_pull_requests(**args),
            'get_pull_request': lambda: client.get_pull_request(**args),
            'get_pull_request_files': lambda: client.get_pull_request_files(**args),
            'get_pull_request_reviews': lambda: client.get_pull_request_reviews(**args),
            'get_pull_request_status': lambda: client.get_pull_request_status(**args),
            'get_pull_request_comments': lambda: client.get_pull_request_comments(**args),
            'get_pull_request_diff': lambda: client.get_pull_request_diff(**args),
            'update_pull_request_branch': lambda: client.update_pull_request_branch(**args),
            'search_pull_requests': lambda: client.search_pull_requests(**args),

            # Repository and Files
            'create_branch': lambda: client.create_branch(**args),
            'list_branches': lambda: client.list_branches(**args),
            'list_commits': lambda: client.list_commits(**args),
            'get_commit': lambda: client.get_commit(**args),
            'get_file_contents': lambda: client.get_file_contents(**args),
            'create_or_update_file': lambda: client.create_or_update_file(**args),
            'delete_file': lambda: client.delete_file(**args),
            'push_files': lambda: client.push_files(**args),

            # Tags and Releases
            'list_tags': lambda: client.list_tags(**args),
            'get_tag': lambda: client.get_tag(**args),
            'list_releases': lambda: client.list_releases(**args),
            'get_latest_release': lambda: client.get_latest_release(**args),
            'get_release_by_tag': lambda: client.get_release_by_tag(**args),

            # Workflows
            'list_workflows': lambda: client.list_workflows(**args),
            'list_workflow_runs': lambda: client.list_workflow_runs(**args),
            'get_workflow_run': lambda: client.get_workflow_run(**args),
            'get_workflow_run_usage': lambda: client.get_workflow_run_usage(**args),
            'get_workflow_run_logs': lambda: client.get_workflow_run_logs(**args),
            'get_job_logs': lambda: client.get_job_logs(**args),
            'list_workflow_jobs': lambda: client.list_workflow_jobs(**args),
            'list_workflow_run_artifacts': lambda: client.list_workflow_run_artifacts(**args),
            'download_workflow_run_artifact': lambda: client.download_workflow_run_artifact(**args),
            'run_workflow': lambda: client.run_workflow(**args),
            'rerun_workflow_run': lambda: client.rerun_workflow_run(**args),
            'rerun_failed_jobs': lambda: client.rerun_failed_jobs(**args),
            'cancel_workflow_run': lambda: client.cancel_workflow_run(**args),

            # Search
            'search_code': lambda: client.search_code(**args),

            # Security
            'list_code_scanning_alerts': lambda: client.list_code_scanning_alerts(**args),
            'get_code_scanning_alert': lambda: client.get_code_scanning_alert(**args),
            'list_dependabot_alerts': lambda: client.list_dependabot_alerts(**args),
            'get_dependabot_alert': lambda: client.get_dependabot_alert(**args),
            'list_secret_scanning_alerts': lambda: client.list_secret_scanning_alerts(**args),
            'get_secret_scanning_alert': lambda: client.get_secret_scanning_alert(**args),

            # Notifications and Users
            'list_notifications': lambda: client.list_notifications(**args),
            'mark_notifications_read': lambda: client.mark_notifications_read(**args),
            'search_users': lambda: client.search_users(**args),
            'search_orgs': lambda: client.search_orgs(**args),

            # Discussions and Gists
            'list_discussions': lambda: client.list_discussions(**args),
            'get_discussion': lambda: client.get_discussion(**args),
            'list_discussion_comments': lambda: client.list_discussion_comments(**args),
            'create_gist': lambda: client.create_gist(**args),
            'list_gists': lambda: client.list_gists(**args),
            'update_gist': lambda: client.update_gist(**args),
        }

        action = action_map.get(tool)
        if not action:
            return {'error': f'Unknown GitHub action: {tool}'}

        result = action()
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}


def find_best_jira_transition(transitions: List[Dict], target_state: str) -> Optional[str]:
    """Find the best matching Jira transition for the target state."""
    if not transitions:
        return None

    # Normalize target state
    target_lower = target_state.lower()

    # Priority order for different states
    state_mappings = {
        'done': ['done', 'close', 'closed', 'resolve', 'resolved', 'complete', 'completed'],
        'closed': ['close', 'closed', 'done', 'resolve', 'resolved'],
        'open': ['reopen', 'open', 'start', 'in progress', 'to do'],
    }

    search_terms = state_mappings.get(target_lower, [target_lower])

    # Look for exact matches first
    for term in search_terms:
        for transition in transitions:
            transition_name = transition.get('name', '').lower()
            if term == transition_name:
                return transition.get('id')

    # Look for partial matches
    for term in search_terms:
        for transition in transitions:
            transition_name = transition.get('name', '').lower()
            if term in transition_name or transition_name in term:
                return transition.get('id')

    return None


def execute_jira_action(client, tool: str, args: dict) -> dict:
    """Execute Jira action."""
    try:
        if tool.startswith('jira_'):
            tool = tool[5:]

        # Handle create_issue parameter mapping and defaults
        if tool == 'create_issue':
            # Map common parameter names to Jira API expected names
            if 'title' in args:
                args['summary'] = args.pop('title')
            if 'body' in args:
                args['description'] = args.pop('body')
            if 'content' in args:
                args['description'] = args.pop('content')

            # Handle project parameter mapping
            if 'project' in args:
                args['project_key'] = args.pop('project')

            # Add default project if not specified
            if 'project_key' not in args and st.session_state.jira_project:
                args['project_key'] = st.session_state.jira_project

            # Remove issue_type - let the client use its defaults
            if 'issue_type' in args:
                args.pop('issue_type')

            # Clean up any other parameters that might cause issues
            # Only keep the core parameters that Jira create_issue expects
            valid_params = {'project_key', 'summary', 'description', 'priority', 'assignee', 'labels'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'search' and 'jql' not in args and st.session_state.jira_project:
            args['jql'] = f'project = "{st.session_state.jira_project}" ORDER BY created DESC'

        elif tool == 'transition_issue':
            # Handle smart transition resolution
            if 'target_state' in args:
                target_state = args.pop('target_state')
                issue_key = args.get('issue_key')

                if issue_key:
                    # Get available transitions
                    transitions_result = client.list_transitions(issue_key)
                    transitions = transitions_result.get('transitions', [])

                    # Find best matching transition
                    transition_id = find_best_jira_transition(transitions, target_state)

                    if transition_id:
                        args['transition_id'] = transition_id
                    else:
                        return {'error': f'No suitable transition found for state "{target_state}"'}

        action_map = {
            'create_issue': lambda: client.create_issue(**args),
            'search': lambda: client.search(**args),
            'add_comment': lambda: client.add_comment(**args),
            'transition_issue': lambda: client.transition_issue(**args),
            'list_transitions': lambda: client.list_transitions(**args),
            'project_info': lambda: client.project_info(**args),
            'whoami': lambda: client.whoami(),
        }

        action = action_map.get(tool)
        if not action:
            return {'error': f'Unknown Jira action: {tool}'}

        result = action()
        if tool == 'search' and isinstance(result, dict) and 'issues' in result:
            return {'success': True, 'data': result['issues']}
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}


def execute_notion_action(client, tool: str, args: dict) -> dict:
    """Execute Notion action."""
    try:
        if tool.startswith('notion_'):
            tool = tool[7:]

        # Add default database_id if not provided
        if tool == 'query_database' and 'database_id' not in args and st.session_state.notion_database_id:
            args['database_id'] = st.session_state.notion_database_id

        # Handle property formatting for create_page
        if tool == 'create_page':
            # Handle parameter mapping
            if 'content' in args:
                args['body'] = args.pop('content')

            if 'properties' in args:
                props = args.get('properties', {})
                if isinstance(props, dict):
                    formatted_props = {}
                    for key, value in props.items():
                        if isinstance(value, str):
                            if key.lower() in ['title', 'name']:
                                formatted_props[key] = {"title": [{"type": "text", "text": {"content": value}}]}
                            else:
                                formatted_props[key] = {"rich_text": [{"type": "text", "text": {"content": value}}]}
                        elif isinstance(value, (int, float)):
                            formatted_props[key] = {"number": value}
                        elif isinstance(value, bool):
                            formatted_props[key] = {"checkbox": value}
                        else:
                            formatted_props[key] = value
                    args['properties'] = formatted_props

            # Clean up parameters - only keep valid Notion API parameters
            valid_params = {'parent', 'properties', 'children', 'icon', 'cover'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'query_database':
            # Clean parameters for database queries
            valid_params = {'database_id', 'filter', 'sorts', 'start_cursor', 'page_size'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        action_map = {
            'create_page': lambda: client.create_page(**args),
            'update_page': lambda: client.update_page(**args),
            'get_page': lambda: client.get_page(**args),
            'query_database': lambda: client.query_database(**args),
            'get_database': lambda: client.get_database(**args),
            'search': lambda: client.search(**args),
            'create_database': lambda: client.create_database(**args),
            'append_blocks': lambda: client.append_block_children(**args),
            'get_users': lambda: client.get_users(),
            'get_bot_info': lambda: client.get_bot_info(),
        }

        action = action_map.get(tool)
        if not action:
            return {'error': f'Unknown Notion action: {tool}'}

        result = action()
        if tool == 'query_database' and isinstance(result, dict) and 'results' in result:
            return {'success': True, 'data': result['results']}
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}


def get_ai_response_with_actions(user_message: str, model: str) -> Dict[str, Any]:
    """Get AI response with actions using the improved LLM agent."""
    if not st.session_state.openai_key:
        return {'error': 'OpenAI API key not configured'}

    try:
        # Use the improved propose_actions function
        result = propose_actions(
            openai_key=st.session_state.openai_key,
            model=model,
            user_message=user_message,
            gh_owner=st.session_state.gh_owner,
            gh_repo=st.session_state.gh_repo,
            jira_project_key=st.session_state.jira_project,
            notion_database_id=st.session_state.notion_database_id
        )

        return {
            "message": result.get("message", "Task processed"),
            "actions": result.get("actions", [])
        }

    except Exception as e:
        logger.error(f"AI response error: {e}")
        return {'error': f'AI response error: {str(e)}'}


def execute_actions(actions: List[Dict]) -> List[Dict]:
    """Execute actions and return results."""
    results = []

    for i, action in enumerate(actions):
        service = action.get('service', '').lower()
        action_name = action.get('action', '')
        args = action.get('args', {})
        description = action.get('description', f"{service} {action_name}")

        try:
            if service == 'github' and st.session_state.github_client:
                result = execute_github_action(st.session_state.github_client, action_name, args)
            elif service == 'jira' and st.session_state.jira_client:
                result = execute_jira_action(st.session_state.jira_client, action_name, args)
            elif service == 'notion' and st.session_state.notion_client:
                result = execute_notion_action(st.session_state.notion_client, action_name, args)
            else:
                result = {'error': f'{service} client not available'}

            results.append({
                'action': description,
                'service': service,
                'success': 'success' in result,
                'result': result,
                'index': i
            })

        except Exception as e:
            results.append({
                'action': description,
                'service': service,
                'success': False,
                'result': {'error': str(e)},
                'index': i
            })

    return results


def create_github_issues_dataframe(issues_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for GitHub issues."""
    if not issues_data:
        return pd.DataFrame()

    # Prepare data for DataFrame - same columns as before
    table_data = []
    for issue in issues_data:
        if isinstance(issue, dict):
            number = issue.get('number', 'N/A')
            title = issue.get('title', 'No title')
            state = issue.get('state', 'unknown').upper()
            created_at = issue.get('created_at', '')
            assignee = 'Unassigned'

            # Format date
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_at = date_obj.strftime('%Y-%m-%d')
                except:
                    created_at = created_at[:10]

            # Get assignee info
            if issue.get('assignee') and isinstance(issue['assignee'], dict):
                assignee = issue['assignee'].get('login', 'Unknown')
            elif issue.get('assignees') and len(issue['assignees']) > 0:
                assignee = issue['assignees'][0].get('login', 'Unknown')

            # Get labels
            labels = []
            if issue.get('labels'):
                labels = [label.get('name', '') for label in issue['labels'][:3]]
            labels_str = ', '.join(labels) if labels else '-'

            table_data.append({
                'Number': f"#{number}",
                'Title': title,
                'State': state,
                'Created': created_at,
                'Assignee': assignee,
                'Labels': labels_str
            })

    return pd.DataFrame(table_data)


def create_jira_issues_dataframe(issues_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for Jira issues with bulletproof data extraction."""
    if not issues_data:
        return pd.DataFrame()

    table_data = []
    for issue in issues_data:
        if isinstance(issue, dict):
            # Extract key
            key = issue.get('key', 'N/A')

            # Extract fields with multiple fallbacks
            fields = issue.get('fields', {})

            # Summary extraction with fallbacks
            summary = None
            summary_candidates = [
                fields.get('summary'),
                issue.get('summary'),
                fields.get('title'),
                issue.get('title'),
                fields.get('name'),
                issue.get('name'),
                fields.get('subject'),
                issue.get('subject')
            ]
            for candidate in summary_candidates:
                if candidate and isinstance(candidate, str) and candidate.strip():
                    summary = candidate.strip()
                    break

            if not summary:
                summary = 'No title'
                logger.debug(f"No summary found for {key}")

            # Status extraction with fallbacks
            status = 'Unknown'
            status_obj = fields.get('status')
            if status_obj and isinstance(status_obj, dict):
                status = status_obj.get('name', 'Unknown')
            elif isinstance(status_obj, str):
                status = status_obj
            else:
                # Try alternate status fields
                status_candidates = [
                    fields.get('statusCategory', {}).get('name') if isinstance(fields.get('statusCategory'),
                                                                               dict) else None,
                    fields.get('state'),
                    issue.get('status'),
                    issue.get('state')
                ]
                for candidate in status_candidates:
                    if candidate and isinstance(candidate, str):
                        status = candidate
                        break

            # Issue type extraction
            issue_type = '-'
            issuetype_obj = fields.get('issuetype') or fields.get('issueType') or fields.get('type')
            if issuetype_obj and isinstance(issuetype_obj, dict):
                issue_type = issuetype_obj.get('name', '-')
            elif isinstance(issuetype_obj, str):
                issue_type = issuetype_obj

            # Priority extraction
            priority = '-'
            priority_obj = fields.get('priority')
            if priority_obj and isinstance(priority_obj, dict):
                priority = priority_obj.get('name', '-')
            elif isinstance(priority_obj, str):
                priority = priority_obj

            # Assignee extraction with fallbacks
            assignee = 'Unassigned'
            assignee_obj = fields.get('assignee')
            if assignee_obj and isinstance(assignee_obj, dict):
                assignee_candidates = [
                    assignee_obj.get('displayName'),
                    assignee_obj.get('name'),
                    assignee_obj.get('emailAddress'),
                    assignee_obj.get('accountId')
                ]
                for candidate in assignee_candidates:
                    if candidate and isinstance(candidate, str):
                        assignee = candidate
                        if len(assignee) > 20:
                            assignee = assignee[:17] + "..."
                        break

            # Created date extraction with fallbacks
            created = ''
            created_candidates = [
                fields.get('created'),
                fields.get('createdDate'),
                issue.get('created'),
                issue.get('createdDate')
            ]
            for candidate in created_candidates:
                if candidate:
                    try:
                        date_obj = datetime.fromisoformat(str(candidate).replace('Z', '+00:00'))
                        created = date_obj.strftime('%Y-%m-%d')
                        break
                    except:
                        # Fallback to first 10 characters
                        created = str(candidate)[:10]
                        if len(created) >= 8:  # Reasonable date length
                            break

            table_data.append({
                'Key': key,
                'Summary': summary,
                'Status': status,
                'Type': issue_type,
                'Priority': priority,
                'Created': created,
                'Assignee': assignee
            })

    return pd.DataFrame(table_data)


def create_notion_pages_dataframe(pages_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for Notion pages."""
    if not pages_data:
        return pd.DataFrame()

    table_data = []
    for page in pages_data:
        if isinstance(page, dict):
            # Extract title from properties
            title = "Untitled"
            status = "-"
            created = ""

            properties = page.get('properties', {})
            for prop_name, prop_data in properties.items():
                if prop_data.get('type') == 'title':
                    title_texts = prop_data.get('title', [])
                    if title_texts:
                        title = ''.join([t.get('plain_text', '') for t in title_texts])
                elif 'status' in prop_name.lower() and prop_data.get('type') == 'select':
                    select_data = prop_data.get('select')
                    if select_data:
                        status = select_data.get('name', '-')

            # Get created date
            if page.get('created_time'):
                try:
                    date_obj = datetime.fromisoformat(page['created_time'].replace('Z', '+00:00'))
                    created = date_obj.strftime('%Y-%m-%d')
                except:
                    created = page['created_time'][:10]

            table_data.append({
                'Title': title,
                'Status': status,
                'Type': page.get('object', 'page').title(),
                'Created': created
            })

    return pd.DataFrame(table_data)


def format_execution_results_with_tables(results: List[Dict]) -> Tuple[str, List[pd.DataFrame], List[str]]:
    """Format results for display - return text summary, DataFrames, and table titles."""
    if not results:
        return "", [], []

    summary_parts = []
    dataframes = []
    table_titles = []

    for result in results:
        action = result['action']
        success = result['success']
        service = result.get('service', '').title()

        if success:
            data = result['result'].get('data', {})

            if isinstance(data, dict):
                if 'number' in data:  # GitHub issue/PR created
                    summary_parts.append(f"✅ {service}: Issue #{data['number']} created")
                    if 'html_url' in data:
                        summary_parts.append(f"   🔗 {data['html_url']}")
                elif 'key' in data:  # Jira issue created
                    summary_parts.append(f"✅ {service}: {data['key']} created")
                elif 'displayName' in data:  # Jira user info
                    summary_parts.append(f"✅ {service}: User - {data['displayName']}")
                elif 'name' in data and service == 'Notion':  # Notion bot
                    summary_parts.append(f"✅ {service}: Bot - {data['name']}")
                else:
                    summary_parts.append(f"✅ {service}: {action} completed")

            elif isinstance(data, list):  # List results
                summary_parts.append(f"✅ {service}: Found {len(data)} items")

                # Create DataFrames for different services
                if service.lower() == 'github' and data:
                    # Check if this looks like issues (has number and title)
                    if any('number' in item and 'title' in item for item in data[:3] if isinstance(item, dict)):
                        df = create_github_issues_dataframe(data)
                        if not df.empty:
                            dataframes.append(df)
                            table_titles.append("GitHub Issues")
                    else:
                        # Handle other GitHub list data generically
                        generic_data = []
                        for item in data[:50]:  # Limit to 50 items
                            if isinstance(item, dict):
                                row = {}
                                if 'name' in item:
                                    row['Name'] = item['name']
                                if 'login' in item:
                                    row['Login'] = item['login']
                                if 'created_at' in item:
                                    row['Created'] = item['created_at'][:10]
                                if row:  # Only add if we extracted some data
                                    generic_data.append(row)

                        if generic_data:
                            df = pd.DataFrame(generic_data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append(f"GitHub {action.title()}")

                elif service.lower() == 'jira' and data:
                    df = create_jira_issues_dataframe(data)
                    if not df.empty:
                        dataframes.append(df)
                        table_titles.append("Jira Issues")

                elif service.lower() == 'notion' and data:
                    df = create_notion_pages_dataframe(data)
                    if not df.empty:
                        dataframes.append(df)
                        table_titles.append("Notion Pages")

            else:
                summary_parts.append(f"✅ {service}: {action} completed")
        else:
            error = result['result'].get('error', 'Unknown error')
            summary_parts.append(f"❌ {service}: {action} failed - {error[:100]}")

    summary_text = "\n".join(summary_parts)
    return summary_text, dataframes, table_titles


def handle_chat_message(user_input: str, model: str):
    """Handle chat message with direct execution."""
    if not user_input.strip():
        st.warning("Please enter a message")
        return

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })

    with st.spinner("Processing..."):
        start_time = time.time()

        # Get AI response with actions
        ai_response = get_ai_response_with_actions(user_input, model)

        if 'error' in ai_response:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {ai_response['error']}",
                "timestamp": datetime.now(),
                "error": True
            })
            st.rerun()
            return

        message = ai_response.get('message', 'Task completed.')
        actions = ai_response.get('actions', [])

        # Execute actions if any
        execution_results = []
        dataframes = []
        table_titles = []

        if actions:
            execution_results = execute_actions(actions)
            results_text, dataframes, table_titles = format_execution_results_with_tables(execution_results)

            # Only add results text to message if it's not empty
            if results_text.strip():
                message += f"\n\n{results_text}"

        execution_time = time.time() - start_time

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now(),
            "actions_executed": len(actions),
            "execution_time": execution_time,
            "dataframes": dataframes,
            "table_titles": table_titles
        })

    st.rerun()


def display_chat_history():
    """Display clean chat history with proper DataFrame rendering."""
    if not st.session_state.chat_history:
        st.info("Start chatting! Try: 'Show me what I have in Jira' or 'List my GitHub issues'")
        return

    for message in st.session_state.chat_history:
        timestamp = message['timestamp'].strftime('%H:%M:%S')

        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You ({timestamp}):</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message
            execution_info = ""
            if message.get('actions_executed', 0) > 0:
                execution_info = f" | {message['actions_executed']} actions in {message.get('execution_time', 0):.1f}s"

            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Assistant ({timestamp}{execution_info}):</strong><br>
                {message['content'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Display DataFrames if present
            dataframes = message.get('dataframes', [])
            table_titles = message.get('table_titles', [])

            for i, df in enumerate(dataframes):
                title = table_titles[i] if i < len(table_titles) else f"Table {i + 1}"
                st.subheader(title)
                st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN APP
# ============================================================================

init_session_state()

# Header
st.markdown("""
<div class="main-header">
  <h1>Multi-Platform Assistant</h1>
  <p>Execute tasks across GitHub, Jira, and Notion</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")

    # Connection Status
    st.markdown("**Connection Status**")

    status_github = "Connected" if st.session_state.github_client else "Not Connected"
    status_jira = "Connected" if st.session_state.jira_client else "Not Connected"
    status_notion = "Connected" if st.session_state.notion_client else "Not Connected"
    status_openai = "Connected" if st.session_state.openai_key else "Not Connected"

    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.openai_key else "disconnected"}">OpenAI: {status_openai}</span>',
        unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.github_client else "disconnected"}">GitHub: {status_github}</span>',
        unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.jira_client else "disconnected"}">Jira: {status_jira}</span>',
        unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.notion_client else "disconnected"}">Notion: {status_notion}</span>',
        unsafe_allow_html=True)

    st.markdown("---")

    # OpenAI Configuration
    st.subheader("OpenAI")
    openai_key = st.text_input("API Key", type="password", key='sidebar_openai_key')
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"], index=0)

    # GitHub Configuration
    st.subheader("GitHub")
    github_token = st.text_input("Token", type="password", key='sidebar_github_token')
    gh_owner = st.text_input("Owner", key='sidebar_gh_owner')
    gh_repo = st.text_input("Repository", key='sidebar_gh_repo')

    # Jira Configuration
    st.subheader("Jira")
    jira_url = st.text_input("Base URL", placeholder="https://company.atlassian.net", key='sidebar_jira_url')
    jira_email = st.text_input("Email", key='sidebar_jira_email')
    jira_token = st.text_input("API Token", type="password", key='sidebar_jira_token')
    jira_project = st.text_input("Project Key", key='sidebar_jira_project')

    # Notion Configuration
    st.subheader("Notion")
    notion_token = st.text_input("Integration Token", type="password", key='sidebar_notion_token')
    notion_database_id = st.text_input("Database ID (optional)", key='sidebar_notion_database_id')

    st.markdown("---")

    # Connect Services Button
    if st.button("Connect Services", use_container_width=True, type="primary"):
        initialize_clients()

    # Clear Chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat cleared!")
        st.rerun()

# Main Chat Interface
st.subheader("Chat with Assistant")

# Display chat history
display_chat_history()

# Chat input
st.markdown("---")
user_input = st.text_area(
    "Your message:",
    height=100,
    value=st.session_state.get('current_input', ''),
    placeholder="Try: 'Show me what I have in Jira' or 'Create an issue in GitHub'",
    key="message_input"
)

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Send", type="primary", use_container_width=True):
        if user_input.strip():
            st.session_state.current_input = ""
            handle_chat_message(user_input, model)
            st.rerun()

# Footer
st.markdown("---")
connected_count = sum([
    1 for client in [st.session_state.github_client, st.session_state.jira_client, st.session_state.notion_client]
    if client is not None
])

st.markdown(f"**Status:** {connected_count}/3 platforms connected")

if connected_count == 0:
    st.info("Connect your services in the sidebar to start using the assistant.")

if __name__ == "__main__":
    if not IMPORTS_AVAILABLE:
        st.error("Required modules not available. Please install dependencies.")
        st.stop()