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
            client = GitHubClient(github_token, config)
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

        action_map = {
            'create_issue': lambda: client.create_issue(**args),
            'list_issues': lambda: client.list_issues(**args),
            'list_open_issues': lambda: client.list_issues(state="open", **args),
            'create_pull_request': lambda: client.create_pull_request(**args),
            'open_pull_request': lambda: client.create_pull_request(**args),
            'list_pull_requests': lambda: client.list_pull_requests(**args),
            'create_branch': lambda: client.create_branch(**args),
            'list_branches': lambda: client.list_branches(**args),
            'create_or_update_file': lambda: client.create_or_update_file(**args),
            'run_workflow': lambda: client.run_workflow(**args),
            'list_workflows': lambda: client.list_workflows(**args),
        }

        action = action_map.get(tool)
        if not action:
            return {'error': f'Unknown GitHub action: {tool}'}

        result = action()
        return {'success': True, 'data': result}
    except Exception as e:
        return {'error': str(e)}


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

        if tool == 'search' and 'jql' not in args and st.session_state.jira_project:
            args['jql'] = f'project = "{st.session_state.jira_project}" ORDER BY created DESC'

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
    """Get AI response with actions - simplified version."""
    if not st.session_state.openai_key:
        return {'error': 'OpenAI API key not configured'}

    client = get_openai_client()
    if not client:
        return {'error': 'Failed to initialize OpenAI client'}

    # Build context
    context_parts = []
    if st.session_state.github_client:
        context_parts.append(f"GitHub: {st.session_state.gh_owner}/{st.session_state.gh_repo}")
    if st.session_state.jira_client:
        context_parts.append(f"Jira: {st.session_state.jira_project}")
    if st.session_state.notion_client:
        context_parts.append(f"Notion: Connected")

    context = " | ".join(context_parts) if context_parts else "No services connected"

    system_prompt = f"""You are an AI assistant that executes GitHub, Jira, and Notion tasks.

Current Context: {context}

CAPABILITIES:
GitHub (if connected): create_issue, list_issues, create_pull_request, list_pull_requests, create_branch, list_branches
Jira (if connected): create_issue, search, add_comment, project_info, whoami
Notion (if connected): create_page, query_database, search, get_bot_info

CRITICAL PARAMETER RULES:
For Jira create_issue, ONLY use these parameters:
- "summary" (for the title/name)
- "description" (for the content/body)
- "project_key" (will be auto-filled if not provided)

For GitHub create_issue, use:
- "title" (for the issue title)
- "body" (for the content)

For Notion create_page, use:
- "properties" (formatted object)

RESPONSE FORMAT (JSON only):
{{
    "message": "Your response to the user",
    "actions": [
        {{
            "service": "github|jira|notion",
            "action": "function_name",
            "args": {{"param": "value"}},
            "description": "What this does"
        }}
    ]
}}

Examples:
- "Create Jira issue title Blue content Bottle" → {{"service": "jira", "action": "create_issue", "args": {{"summary": "Blue", "description": "Bottle"}}}}
- "Create GitHub issue title Bug content Fix login" → {{"service": "github", "action": "create_issue", "args": {{"title": "Bug", "body": "Fix login"}}}}
- "Show me Jira issues" → {{"service": "jira", "action": "search", "args": {{}}}}

Always execute requested tasks immediately using the EXACT parameter names specified above."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        response_text = response.choices[0].message.content

        # Parse JSON with fallbacks
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass

            # Fallback
            return {"message": response_text, "actions": []}

    except Exception as e:
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


def create_github_issues_table(issues_data: List[Dict]) -> str:
    """Create a formatted table for GitHub issues."""
    if not issues_data:
        return "No issues found."

    # Prepare data for table
    table_data = []
    for issue in issues_data[:25]:  # Show up to 25 issues
        # Handle both API response formats
        if isinstance(issue, dict):
            number = issue.get('number', 'N/A')
            title = issue.get('title', 'No title')
            # Truncate title but keep it readable
            if len(title) > 50:
                title = title[:47] + "..."

            state = issue.get('state', 'unknown').upper()
            created_at = issue.get('created_at', '')
            assignee = 'Unassigned'

            # Format date
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_at = date_obj.strftime('%m/%d/%y')
                except:
                    created_at = created_at[:10]  # Keep first 10 chars if parsing fails

            # Get assignee info
            if issue.get('assignee') and isinstance(issue['assignee'], dict):
                assignee = issue['assignee'].get('login', 'Unknown')
            elif issue.get('assignees') and len(issue['assignees']) > 0:
                assignee = issue['assignees'][0].get('login', 'Unknown')

            # Get labels
            labels = []
            if issue.get('labels'):
                labels = [label.get('name', '') for label in issue['labels'][:2]]  # Max 2 labels
            labels_str = ', '.join(labels) if labels else '-'

            table_data.append([
                f"#{number}",
                title,
                state,
                created_at,
                assignee,
                labels_str
            ])

    if table_data:
        # Create a clean table format
        headers = ['Number', 'Title', 'State', 'Created', 'Assignee', 'Labels']

        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]
        col_widths = [min(w, 50) for w in col_widths]  # Cap width at 50 chars

        # Create table
        table_lines = []

        # Header
        header_line = '| ' + ' | '.join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + ' |'
        separator_line = '|' + '|'.join('-' * (col_widths[i] + 2) for i in range(len(headers))) + '|'

        table_lines.append(header_line)
        table_lines.append(separator_line)

        # Data rows
        for row in table_data:
            row_line = '| ' + ' | '.join(
                str(row[i]).ljust(col_widths[i])[:col_widths[i]] for i in range(len(row))) + ' |'
            table_lines.append(row_line)

        return f"\n\n**GitHub Issues ({len(issues_data)} total, showing {len(table_data)}):**\n```\n" + '\n'.join(
            table_lines) + "\n```"

    return "No issues to display."


def create_jira_issues_table(issues_data: List[Dict]) -> str:
    """Create a formatted table for Jira issues."""
    if not issues_data:
        return "No issues found."

    table_data = []
    for issue in issues_data[:25]:  # Show up to 25 issues
        if isinstance(issue, dict):
            key = issue.get('key', 'N/A')
            fields = issue.get('fields', {})

            summary = fields.get('summary', 'No title')
            if len(summary) > 45:
                summary = summary[:42] + "..."

            status = 'Unknown'
            assignee = 'Unassigned'
            created = ''
            priority = '-'
            issue_type = '-'

            # Get status
            if fields.get('status') and isinstance(fields['status'], dict):
                status = fields['status'].get('name', 'Unknown')

            # Get assignee
            if fields.get('assignee') and isinstance(fields['assignee'], dict):
                assignee = fields['assignee'].get('displayName', 'Unknown')
                if len(assignee) > 15:
                    assignee = assignee[:12] + "..."

            # Get created date
            if fields.get('created'):
                try:
                    date_obj = datetime.fromisoformat(fields['created'].replace('Z', '+00:00'))
                    created = date_obj.strftime('%m/%d/%y')
                except:
                    created = fields['created'][:10]

            # Get priority
            if fields.get('priority') and isinstance(fields['priority'], dict):
                priority = fields['priority'].get('name', '-')

            # Get issue type
            if fields.get('issuetype') and isinstance(fields['issuetype'], dict):
                issue_type = fields['issuetype'].get('name', '-')

            table_data.append([
                key,
                summary,
                status,
                issue_type,
                priority,
                created,
                assignee
            ])

    if table_data:
        headers = ['Key', 'Summary', 'Status', 'Type', 'Priority', 'Created', 'Assignee']

        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]
        col_widths = [min(w, 45) for w in col_widths]  # Cap width

        # Create table
        table_lines = []

        # Header
        header_line = '| ' + ' | '.join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + ' |'
        separator_line = '|' + '|'.join('-' * (col_widths[i] + 2) for i in range(len(headers))) + '|'

        table_lines.append(header_line)
        table_lines.append(separator_line)

        # Data rows
        for row in table_data:
            row_line = '| ' + ' | '.join(
                str(row[i]).ljust(col_widths[i])[:col_widths[i]] for i in range(len(row))) + ' |'
            table_lines.append(row_line)

        return f"\n\n**Jira Issues ({len(issues_data)} total, showing {len(table_data)}):**\n```\n" + '\n'.join(
            table_lines) + "\n```"

    return "No issues to display."


def create_notion_pages_table(pages_data: List[Dict]) -> str:
    """Create a formatted table for Notion pages."""
    if not pages_data:
        return "No pages found."

    table_data = []
    for page in pages_data[:25]:  # Show up to 25 pages
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
                        if len(title) > 40:
                            title = title[:37] + "..."
                elif 'status' in prop_name.lower() and prop_data.get('type') == 'select':
                    select_data = prop_data.get('select')
                    if select_data:
                        status = select_data.get('name', '-')

            # Get created date
            if page.get('created_time'):
                try:
                    date_obj = datetime.fromisoformat(page['created_time'].replace('Z', '+00:00'))
                    created = date_obj.strftime('%m/%d/%y')
                except:
                    created = page['created_time'][:10]

            table_data.append([
                title,
                status,
                page.get('object', 'page').title(),
                created
            ])

    if table_data:
        headers = ['Title', 'Status', 'Type', 'Created']

        # Calculate column widths
        col_widths = [max(len(str(row[i])) for row in [headers] + table_data) for i in range(len(headers))]
        col_widths = [min(w, 50) for w in col_widths]

        # Create table
        table_lines = []

        # Header
        header_line = '| ' + ' | '.join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + ' |'
        separator_line = '|' + '|'.join('-' * (col_widths[i] + 2) for i in range(len(headers))) + '|'

        table_lines.append(header_line)
        table_lines.append(separator_line)

        # Data rows
        for row in table_data:
            row_line = '| ' + ' | '.join(
                str(row[i]).ljust(col_widths[i])[:col_widths[i]] for i in range(len(row))) + ' |'
            table_lines.append(row_line)

        return f"\n\n**Notion Pages ({len(pages_data)} total, showing {len(table_data)}):**\n```\n" + '\n'.join(
            table_lines) + "\n```"

    return "No pages to display."


def format_execution_results(results: List[Dict]) -> str:
    """Format results for display with improved table formatting."""
    if not results:
        return ""

    formatted = "\n\n**Results:**\n"

    for result in results:
        action = result['action']
        success = result['success']
        service = result.get('service', '').title()

        if success:
            data = result['result'].get('data', {})

            if isinstance(data, dict):
                if 'number' in data:  # GitHub issue/PR created
                    formatted += f"\n✅ {service}: Issue #{data['number']} created"
                    if 'html_url' in data:
                        formatted += f" - {data['html_url']}"
                elif 'key' in data:  # Jira issue created
                    formatted += f"\n✅ {service}: {data['key']} created"
                elif 'displayName' in data:  # Jira user info
                    formatted += f"\n✅ {service}: User - {data['displayName']}"
                elif 'name' in data and service == 'Notion':  # Notion bot
                    formatted += f"\n✅ {service}: Bot - {data['name']}"
                else:
                    formatted += f"\n✅ {service}: {action} completed"

            elif isinstance(data, list):  # List results
                formatted += f"\n✅ {service}: Found {len(data)} items"

                # Use specialized table formatters for different services
                if service.lower() == 'github' and data:
                    # Check if this looks like issues (has number and title)
                    if any('number' in item and 'title' in item for item in data[:3] if isinstance(item, dict)):
                        formatted += create_github_issues_table(data)
                    else:
                        # Fallback for other GitHub data
                        formatted += "\n"
                        for item in data[:20]:  # Show up to 20 items
                            if isinstance(item, dict):
                                if 'name' in item:
                                    formatted += f"\n  • {item['name']}"
                                elif 'number' in item and 'title' in item:
                                    formatted += f"\n  • #{item['number']}: {item['title'][:50]}..."

                        if len(data) > 20:
                            formatted += f"\n  ... and {len(data) - 20} more items"

                elif service.lower() == 'jira' and data:
                    formatted += create_jira_issues_table(data)

                elif service.lower() == 'notion' and data:
                    formatted += create_notion_pages_table(data)

                else:
                    # Generic list formatting
                    formatted += "\n"
                    for item in data[:20]:  # Show up to 20 items
                        if isinstance(item, dict):
                            if 'name' in item:
                                formatted += f"\n  • {item['name']}"
                            elif 'title' in item:
                                formatted += f"\n  • {item['title'][:50]}..."
                            elif 'summary' in item:
                                formatted += f"\n  • {item['summary'][:50]}..."
                            else:
                                formatted += f"\n  • {str(item)[:50]}..."

                    if len(data) > 20:
                        formatted += f"\n  ... and {len(data) - 20} more items"

            else:
                formatted += f"\n✅ {service}: {action} completed"
        else:
            error = result['result'].get('error', 'Unknown error')
            formatted += f"\n❌ {service}: {action} failed - {error[:100]}"

    return formatted


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
        if actions:
            execution_results = execute_actions(actions)
            results_text = format_execution_results(execution_results)
            message += results_text

        execution_time = time.time() - start_time

        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now(),
            "actions_executed": len(actions),
            "execution_time": execution_time
        })

    st.rerun()


def display_chat_history():
    """Display clean chat history."""
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