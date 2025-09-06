import streamlit as st
import json
import time
import logging
import base64
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
    from github_client import GitHubClient, GitHubError, GitHubConfig, normalize_issue_pr_input
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

    .error-details {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        word-wrap: break-word;
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
        'current_input': '',
        'jira_connection_status': 'Not Connected',
        'jira_connection_details': {}
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
    """
    Initialize clients with proper error handling and credential validation.

    FIXED: Enhanced Jira connection with better URL validation and error handling.
    """
    success_messages = []
    error_messages = []

    # Clear existing clients
    st.session_state.github_client = None
    st.session_state.jira_client = None
    st.session_state.notion_client = None
    st.session_state.jira_connection_status = 'Not Connected'
    st.session_state.jira_connection_details = {}

    # Get values from sidebar - FIXED: Ensure full strings are captured
    openai_key = st.session_state.get('sidebar_openai_key', '').strip()
    github_token = st.session_state.get('sidebar_github_token', '').strip()
    gh_owner = st.session_state.get('sidebar_gh_owner', '').strip()
    gh_repo = st.session_state.get('sidebar_gh_repo', '').strip()

    # FIXED: Enhanced Jira URL validation and normalization
    jira_url = st.session_state.get('sidebar_jira_url', '').strip()
    jira_email = st.session_state.get('sidebar_jira_email', '').strip()
    jira_token = st.session_state.get('sidebar_jira_token', '').strip()
    jira_project = st.session_state.get('sidebar_jira_project', '').strip()

    notion_token = st.session_state.get('sidebar_notion_token', '').strip()
    notion_database_id = st.session_state.get('sidebar_notion_database_id', '').strip()

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
            error_messages.append(f"GitHub: {str(e)[:200]}")

    # Initialize Jira client - FIXED: Enhanced validation and error handling
    if jira_url and jira_email and jira_token:
        try:
            # FIXED: Comprehensive URL validation and normalization
            original_url = jira_url

            # Add protocol if missing
            if not jira_url.startswith(('http://', 'https://')):
                jira_url = f'https://{jira_url}'

            # Handle common URL completion issues
            if 'atlassian' in jira_url and not jira_url.endswith('.net'):
                if jira_url.endswith('.atlassian.n'):
                    jira_url = jira_url + 'et'
                elif jira_url.endswith('.atlassian'):
                    jira_url = jira_url + '.net'
                elif jira_url.endswith('.'):
                    jira_url = jira_url + 'net'

            # Remove any trailing slash
            jira_url = jira_url.rstrip('/')

            # Store connection details for debugging
            st.session_state.jira_connection_details = {
                'original_url': original_url,
                'normalized_url': jira_url,
                'email': jira_email,
                'token_length': len(jira_token),
                'timestamp': datetime.now().isoformat()
            }

            # Log for debugging (without sensitive data)
            logger.info(f"Initializing Jira client:")
            logger.info(f"  Original URL: {original_url}")
            logger.info(f"  Normalized URL: {jira_url}")
            logger.info(f"  Email: {jira_email}")
            logger.info(f"  Token length: {len(jira_token)}")

            # Create Jira client with enhanced config
            config = JiraConfig(
                enable_caching=True,
                max_retries=3,
                timeout=30,
                verify_ssl=True  # Start with SSL verification
            )

            # Create client - this will test connection automatically
            client = JiraClient(jira_url, jira_email, jira_token, config)

            # If we get here, connection was successful
            user_info = client.whoami()
            st.session_state.jira_client = client
            st.session_state.jira_connection_status = 'Connected'

            display_name = user_info.get('displayName', 'User')
            email_address = user_info.get('emailAddress', jira_email)

            success_messages.append(f"Jira: Connected as {display_name} ({email_address})")

        except Exception as e:
            # FIXED: Comprehensive error handling without truncation
            error_str = str(e)
            st.session_state.jira_connection_status = 'Failed'

            # Create detailed error information
            error_details = {
                'error_type': type(e).__name__,
                'error_message': error_str,
                'url_used': jira_url,
                'troubleshooting': []
            }

            # Provide specific troubleshooting guidance
            if "Authentication failed" in error_str or "401" in error_str:
                error_details['troubleshooting'] = [
                    "1. Verify your Jira URL format: https://yourcompany.atlassian.net",
                    "2. Check your email address (must match Atlassian account)",
                    "3. Generate a new API token at: https://id.atlassian.com/manage-profile/security/api-tokens",
                    "4. Ensure your account has API access enabled"
                ]
                error_title = "Jira Authentication Failed"

            elif "Connection" in error_str or "timeout" in error_str.lower():
                error_details['troubleshooting'] = [
                    "1. Check your internet connection",
                    "2. Verify the Jira URL is accessible in your browser",
                    "3. Check if you're behind a corporate firewall",
                    "4. Try accessing Jira directly to confirm it's working"
                ]
                error_title = "Jira Connection Failed"

            elif "SSL" in error_str:
                error_details['troubleshooting'] = [
                    "1. SSL certificate issue detected",
                    "2. This may be due to corporate network security",
                    "3. Try accessing Jira in your browser first",
                    "4. Contact your IT administrator if the issue persists"
                ]
                error_title = "Jira SSL Error"

            else:
                error_details['troubleshooting'] = [
                    "1. Double-check all credentials",
                    "2. Ensure Jira is accessible via browser",
                    "3. Try generating a new API token",
                    "4. Contact your Jira administrator for assistance"
                ]
                error_title = "Jira Connection Error"

            # Display comprehensive error message
            error_msg = f"""
{error_title}

Connection Details:
• URL: {jira_url}
• Email: {jira_email}
• Token: {'*' * (len(jira_token) - 4)}{jira_token[-4:] if len(jira_token) > 4 else '****'}

Error: {error_str}

Troubleshooting Steps:
{chr(10).join(error_details['troubleshooting'])}
            """.strip()

            error_messages.append(error_msg)
            logger.error(f"Jira client initialization failed: {error_str}")

            # Store error details for debugging
            st.session_state.jira_connection_details.update({
                'error': error_details,
                'status': 'Failed'
            })

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
            error_messages.append(f"Notion: {str(e)[:200]}")

    # Display results with enhanced formatting
    if success_messages:
        st.success("✅ " + " | ".join(success_messages))

    if error_messages:
        for error_msg in error_messages:
            if "Jira" in error_msg and len(error_msg) > 200:
                # Display Jira errors in expandable section for better readability
                with st.expander("🔍 Jira Connection Error Details", expanded=True):
                    st.markdown(f'<div class="error-details">{error_msg}</div>', unsafe_allow_html=True)
            else:
                st.error(error_msg)


def normalize_github_issue_input(input_str: str) -> Optional[int]:
    """Normalize GitHub issue input to issue number."""
    if not input_str:
        return None

    try:
        return normalize_issue_pr_input(str(input_str))
    except ValueError:
        return None


def find_best_jira_transition(transitions: List[Dict], target_state: str) -> Optional[str]:
    """
    Find the best matching Jira transition for the target state with improved logic.

    FIXED: Enhanced pattern matching for more reliable transition detection.
    """
    if not transitions:
        return None

    # Normalize target state
    target_lower = target_state.lower().replace('_', ' ').strip()

    # Remove common prefixes like ":" from target state
    if target_lower.startswith(':'):
        target_lower = target_lower[1:]

    # Enhanced state mappings with regex patterns
    state_patterns = {
        'done': [
            r'^(done|close.*|closed.*|resolve.*|resolved.*|complete.*|completed.*)$'
        ],
        'closed': [
            r'^(close.*|closed.*|done|resolve.*|resolved.*|finish.*|finished.*)$'
        ],
        'open': [
            r'^(reopen.*|open|to\s*do.*|backlog.*)$'
        ],
        'in_progress': [
            r'^(in\s*progress.*|start.*|started.*|begin.*|doing.*|active.*)$'
        ],
    }

    patterns = state_patterns.get(target_lower, [f'^.*{re.escape(target_lower)}.*$'])

    # Look for pattern matches
    for pattern in patterns:
        for transition in transitions:
            transition_name = transition.get('name', '').lower()
            if re.match(pattern, transition_name):
                return transition.get('id')

    # Fallback: look for partial matches (more flexible)
    for transition in transitions:
        transition_name = transition.get('name', '').lower()
        if target_lower in transition_name or transition_name in target_lower:
            return transition.get('id')

    # Last resort: look for exact matches ignoring case
    for transition in transitions:
        transition_name = transition.get('name', '').lower()
        if transition_name == target_lower:
            return transition.get('id')

    return None


def execute_github_action(client, tool: str, args: dict) -> dict:
    """
    Execute GitHub action with proper argument mapping and normalization.

    FIXED: Improved parameter handling and error recovery for all GitHub operations.
    """
    try:
        # Add default owner/repo if not provided
        if 'owner' not in args and st.session_state.gh_owner:
            args['owner'] = st.session_state.gh_owner
        if 'repo' not in args and st.session_state.gh_repo:
            args['repo'] = st.session_state.gh_repo

        # Clean tool name
        if tool.startswith('gh_'):
            tool = tool[3:]

        # Normalize issue numbers in arguments
        if 'issue_number' in args:
            if isinstance(args['issue_number'], str):
                normalized_num = normalize_github_issue_input(args['issue_number'])
                if normalized_num:
                    args['issue_number'] = normalized_num
                else:
                    return {'error': f'Invalid issue number: {args["issue_number"]}'}

        # Parameter cleaning and validation based on action
        if tool == 'create_issue':
            valid_params = {'owner', 'repo', 'title', 'body', 'assignees', 'milestone', 'labels'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'update_issue':
            valid_params = {'owner', 'repo', 'issue_number', 'title', 'body', 'state', 'assignees', 'labels'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'get_issue':
            valid_params = {'owner', 'repo', 'issue_number'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'add_issue_comment':
            valid_params = {'owner', 'repo', 'issue_number', 'body'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'search_issues':
            if 'query' not in args:
                return {'error': 'GitHub search_issues requires query parameter'}
            valid_params = {'query', 'sort', 'order', 'per_page', 'page'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'search_pull_requests':
            if 'query' not in args:
                return {'error': 'GitHub search_pull_requests requires query parameter'}
            valid_params = {'query', 'sort', 'order', 'per_page', 'page'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'create_pull_request':
            # FIXED: Enhanced validation for pull request creation
            required_params = ['title', 'head', 'base']
            for param in required_params:
                if param not in args:
                    return {'error': f'create_pull_request requires {param} parameter'}

            valid_params = {'owner', 'repo', 'title', 'body', 'head', 'base', 'draft', 'maintainer_can_modify'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'get_file_contents':
            # FIXED: Ensure both path and ref are present with better defaults
            if 'path' not in args:
                return {'error': 'get_file_contents requires path parameter'}
            if 'ref' not in args:
                args['ref'] = 'main'  # default to main branch
            valid_params = {'owner', 'repo', 'path', 'ref'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'create_or_update_file':
            valid_params = {'owner', 'repo', 'path', 'content', 'message', 'branch'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'create_branch':
            valid_params = {'owner', 'repo', 'branch', 'from_branch'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        # FIXED: Added parameter cleaning for list_branches and list_tags
        elif tool == 'list_branches':
            valid_params = {'owner', 'repo', 'protected', 'per_page', 'page'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'list_tags':
            valid_params = {'owner', 'repo', 'per_page', 'page'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        # Comprehensive action mapping to client methods
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
            'list_repositories': lambda: client.list_repositories(**args),
            'list_repository_contents': lambda: client.list_repository_contents(**args),
            'search_file_across_repositories': lambda: client.search_file_across_repositories(**args),
            'get_repository': lambda: client.get_repository(**args),
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


def execute_jira_action(client, tool: str, args: dict) -> dict:
    """
    Execute Jira action with improved parameter handling and transition logic.

    FIXED: Enhanced Jira transition handling with better state matching and error recovery.
    """
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

            # Clean up parameters
            valid_params = {'project_key', 'summary', 'description', 'priority', 'assignee', 'labels', 'issuetype_name'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'search' and 'jql' not in args and st.session_state.jira_project:
            args['jql'] = f'project = "{st.session_state.jira_project}" ORDER BY created DESC'

        elif tool == 'transition_issue':
            # FIXED: Enhanced smart transition resolution using enhanced client methods
            if 'target_state' in args:
                target_state = args.pop('target_state')
                issue_key = args.get('issue_key')

                if issue_key:
                    try:
                        # Use the enhanced transition_to_status method from the new client if available
                        if hasattr(client, 'transition_to_status'):
                            # Use the new enhanced method
                            result = client.transition_to_status(issue_key, target_state)
                            return {'success': True, 'data': result}
                        else:
                            # Fallback to the improved transition handling
                            if hasattr(client, 'find_transition_by_name'):
                                transition_id = client.find_transition_by_name(issue_key, target_state)
                            else:
                                # Use local fallback method
                                transitions_result = client.list_transitions(issue_key)
                                transitions = transitions_result.get('transitions', [])
                                transition_id = find_best_jira_transition(transitions, target_state)

                            if transition_id:
                                args['transition_id'] = transition_id
                            else:
                                # Get available transitions for error message
                                transitions_result = client.list_transitions(issue_key)
                                available = [t.get('name', 'Unknown') for t in
                                             transitions_result.get('transitions', [])]
                                logger.warning(f"No transition found for '{target_state}'. Available: {available}")
                                return {
                                    'error': f'No suitable transition found for state "{target_state}". Available transitions: {", ".join(available)}'
                                }

                    except Exception as e:
                        logger.error(f"Failed to transition {issue_key}: {e}")
                        return {'error': f'Failed to transition issue: {str(e)}'}

        # Action mapping - Enhanced with new operations
        action_map = {
            # Core issue operations
            'create_issue': lambda: client.create_issue(**args),
            'update_issue': lambda: client.update_issue(**args),
            'delete_issue': lambda: client.delete_issue(**args),
            'get_issue': lambda: client.get_issue(**args),
            
            # Comments
            'add_comment': lambda: client.add_comment(**args),
            
            # Transitions and status  
            'transition_issue': lambda: client.transition_issue(**args),
            'list_transitions': lambda: client.list_transitions(**args),
            'transition_to_status': lambda: client.transition_to_status(**args),
            
            # Labels management
            'set_issue_labels': lambda: client.set_issue_labels(**args),
            'add_issue_labels': lambda: client.add_issue_labels(**args),
            'remove_issue_labels': lambda: client.remove_issue_labels(**args),
            
            # Priority and assignment
            'set_issue_priority': lambda: client.set_issue_priority(**args),
            'assign_issue': lambda: client.assign_issue(**args),
            'unassign_issue': lambda: client.unassign_issue(**args),
            
            # Search and listing
            'search': lambda: client.search(**args),
            
            # Project management
            'list_projects': lambda: client.list_projects(**args),
            'project_info': lambda: client.project_info(**args),
            'get_project_details': lambda: client.get_project_details(**args),
            'create_project': lambda: {'success': False, 'error': 'Project creation requires admin privileges and is typically done through Jira UI. Please contact your Jira administrator or create the project through the Jira web interface.'},
            'update_project': lambda: client.update_project(**args),
            
            # User info
            'whoami': lambda: client.whoami(),
        }

        action = action_map.get(tool)
        if not action:
            return {'error': f'Unknown Jira action: {tool}'}

        result = action()

        # FIXED: Improved result handling to ensure consistent format
        if tool == 'search':
            if isinstance(result, dict) and 'issues' in result:
                return {'success': True, 'data': result['issues']}
            elif isinstance(result, list):
                return {'success': True, 'data': result}

        return {'success': True, 'data': result}

    except Exception as e:
        logger.error(f"Jira action {tool} failed: {e}")
        return {'error': str(e)}


def execute_notion_action(client, tool: str, args: dict) -> dict:
    """Execute Notion action with preserved functionality."""
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

            # FIXED: Handle parent parameter properly - don't use 'parent', use direct parameters
            # Add default database_id if not provided
            if 'parent_database_id' not in args and 'parent_page_id' not in args:
                if st.session_state.notion_database_id:
                    args['parent_database_id'] = st.session_state.notion_database_id

            # FIXED: Improve property handling to handle different property types
            if 'properties' in args:
                props = args.get('properties', {})
                if isinstance(props, dict):
                    # First, get the database schema to understand property types
                    database_id = args.get('parent_database_id') or st.session_state.notion_database_id
                    database_schema = {}
                    
                    if database_id:
                        try:
                            db_info = client.get_database(database_id)
                            database_schema = db_info.get('properties', {})
                        except Exception as e:
                            # If we can't get the schema, proceed with basic formatting
                            pass
                    
                    formatted_props = {}
                    
                    # Special handling for 'title' property - find the actual title property in the database
                    if 'title' in props and database_schema:
                        title_value = props['title']
                        # Find the title property in the database schema
                        title_prop_name = None
                        for prop_name, prop_config in database_schema.items():
                            if prop_config.get('type') == 'title':
                                title_prop_name = prop_name
                                break
                        
                        if title_prop_name:
                            formatted_props[title_prop_name] = {"title": [{"type": "text", "text": {"content": title_value}}]}
                            logger.info(f"Mapped 'title' to database property '{title_prop_name}'")
                        else:
                            logger.warning(f"No title property found in database schema, cannot set page title")
                    
                    for key, value in props.items():
                        # Skip 'title' as it's handled above
                        if key == 'title':
                            continue
                            
                        # Check if the property exists in the database schema
                        if database_schema and key not in database_schema:
                            # Skip properties that don't exist in the database
                            logger.warning(f"Skipping property '{key}' - not found in database schema")
                            continue
                        
                        prop_schema = database_schema.get(key, {})
                        prop_type = prop_schema.get('type', 'rich_text')  # default to rich_text
                        
                        if isinstance(value, str):
                            if prop_type == 'title':
                                formatted_props[key] = {"title": [{"type": "text", "text": {"content": value}}]}
                            elif prop_type == 'select':
                                formatted_props[key] = {"select": {"name": value}}
                            elif prop_type == 'multi_select':
                                # Handle multi-select by splitting comma-separated values
                                options = [opt.strip() for opt in value.split(',')]
                                formatted_props[key] = {"multi_select": [{"name": opt} for opt in options]}
                            else:
                                formatted_props[key] = {"rich_text": [{"type": "text", "text": {"content": value}}]}
                        elif isinstance(value, (int, float)):
                            formatted_props[key] = {"number": value}
                        elif isinstance(value, bool):
                            formatted_props[key] = {"checkbox": value}
                        else:
                            formatted_props[key] = value
                    args['properties'] = formatted_props

            # Clean up parameters - keep parent_database_id and parent_page_id as direct parameters
            valid_params = {'parent_database_id', 'parent_page_id', 'properties', 'children', 'icon', 'cover'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'create_database':
            # FIXED: Handle database creation parameters properly
            # Ensure title is provided and not empty
            if not args.get('title'):
                return {'error': 'Title is required for database creation'}
                
            # Ensure we have properties - create default if not provided
            # Notion databases MUST have at least one property
            if 'properties' not in args or not args['properties']:
                args['properties'] = {
                    "Name": {"title": {}},  # Default title property - required
                    "Status": {"select": {"options": []}}  # Default status property
                }
                
            # Debug: Log the full arguments being passed
            logger.info(f"Streamlit create_database args: {args}")
            logger.info(f"Title value: '{args.get('title')}' (type: {type(args.get('title'))})")
            
            # Database creation requires a parent page ID - cannot be created in workspace root
            if 'parent_page_id' not in args or not args.get('parent_page_id'):
                return {'error': 'Database creation requires a parent page ID. Please create a page first, then create the database under that page.'}
            
            # Validate that we have a page ID, not a database ID
            parent_id = args.get('parent_page_id', '')
            if len(parent_id) == 32 and '-' not in parent_id:
                # This looks like a database ID - check if user meant to create a page instead
                return {'error': f'Cannot create database under a database. The ID "{parent_id}" appears to be a database ID. To create a database, you need to specify a page ID as the parent. Try creating a page first, then create the database under that page.'}
                    
            # Clean parameters for database creation
            valid_params = {'parent_page_id', 'title', 'properties', 'description'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'query_database':
            valid_params = {'database_id', 'filter', 'sorts', 'start_cursor', 'page_size'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'update_page_by_name':
            valid_params = {'page_name', 'properties'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'update_page_status_by_name':
            valid_params = {'page_name', 'status', 'database_id'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        elif tool == 'update_status_with_database':
            valid_params = {'page_name', 'status', 'database_id'}
            cleaned_args = {k: v for k, v in args.items() if k in valid_params}
            args = cleaned_args

        action_map = {
            'create_page': lambda: client.create_page(**args),
            'update_page': lambda: client.update_page(**args),
            'update_page_by_name': lambda: client.update_page_by_name(**args),
            'update_page_status_by_name': lambda: client.update_page_status_by_name(**args),
            'update_status_with_database': lambda: client.update_status_with_database(**args),
            'get_page': lambda: client.get_page(**args),
            'query_database': lambda: client.query_database(**args),
            'get_database': lambda: client.get_database(**args),
            'get_database_properties': lambda: client.get_database_properties(**args),
            'search': lambda: client.search(**args),
            'create_database': lambda: client.create_database(**args),
            'append_blocks': lambda: client.append_block_children(**args),
            'append_text_to_page': lambda: client.append_text_to_page(**args),
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


def get_conversation_context(max_messages: int = 5) -> str:
    """Extract relevant context from recent chat history for short-term memory."""
    if not st.session_state.chat_history:
        return ""
    
    # Get the last few messages for context
    recent_messages = st.session_state.chat_history[-max_messages:]
    context_parts = []
    
    for msg in recent_messages:
        if msg['role'] == 'user':
            context_parts.append(f"User: {msg['content']}")
        elif msg['role'] == 'assistant':
            # Extract key information from assistant responses
            content = msg.get('content', '')
            # Include responses that asked for information or mentioned missing data
            if any(keyword in content.lower() for keyword in ['need', 'require', 'missing', 'which', 'what', 'specify']):
                context_parts.append(f"Assistant: {content}")
    
    return "\n".join(context_parts) if context_parts else ""


def get_ai_response_with_actions(user_message: str, model: str) -> Dict[str, Any]:
    """Get AI response with actions using the improved LLM agent with short-term memory."""
    if not st.session_state.openai_key:
        return {'error': 'OpenAI API key not configured'}

    try:
        # FIXED: Ensure Jira project key is properly passed
        jira_project = st.session_state.get('jira_project', '').strip()

        # Add conversation context for short-term memory
        conversation_context = get_conversation_context()
        
        # Enhanced user message with context if available
        enhanced_message = user_message
        if conversation_context:
            enhanced_message = f"Recent conversation context:\n{conversation_context}\n\nCurrent request: {user_message}"

        # Use the improved propose_actions function
        result = propose_actions(
            openai_key=st.session_state.openai_key,
            model=model,
            user_message=enhanced_message,
            gh_owner=st.session_state.gh_owner,
            gh_repo=st.session_state.gh_repo,
            jira_project_key=jira_project,
            notion_database_id=st.session_state.notion_database_id
        )

        return {
            "message": result.get("message", "Task processed"),
            "actions": result.get("actions", []),
            "needs_info": result.get("needs_info", False)
        }

    except Exception as e:
        logger.error(f"AI response error: {e}")
        return {'error': f'AI response error: {str(e)}'}


def execute_actions(actions: List[Dict]) -> List[Dict]:
    """
    Execute actions and return results with enhanced tracking.

    FIXED: Improved error handling and result consistency across all platforms.
    """
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

            # Enhanced result logging for debugging
            if 'success' in result and result['success']:
                logger.info(f"Action {service}.{action_name} succeeded. Data type: {type(result.get('data'))}")
                if isinstance(result.get('data'), dict):
                    logger.info(f"Data keys: {list(result.get('data', {}).keys())}")
                elif isinstance(result.get('data'), list) and result.get('data'):
                    logger.info(f"Data list length: {len(result['data'])}")
                    if result['data']:
                        logger.info(
                            f"First item keys: {list(result['data'][0].keys()) if isinstance(result['data'][0], dict) else 'Not dict'}")
            else:
                logger.warning(f"Action {service}.{action_name} failed: {result.get('error', 'Unknown error')}")

            results.append({
                'action': description,
                'service': service,
                'raw_action': action_name,
                'raw_args': args,
                'success': 'success' in result,
                'result': result,
                'index': i
            })

        except Exception as e:
            logger.error(f"Action {service}.{action_name} failed with exception: {e}")
            results.append({
                'action': description,
                'service': service,
                'raw_action': action_name,
                'raw_args': args,
                'success': False,
                'result': {'error': str(e)},
                'index': i
            })

    return results


def create_github_issues_dataframe(issues_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for GitHub issues with enhanced detail display."""
    if not issues_data:
        return pd.DataFrame()

    # Prepare data for DataFrame
    table_data = []
    for issue in issues_data:
        if isinstance(issue, dict):
            number = issue.get('number', 'N/A')
            title = issue.get('title', 'No title')
            state = issue.get('state', 'unknown').upper()
            created_at = issue.get('created_at', '')
            updated_at = issue.get('updated_at', '')
            assignee = 'Unassigned'
            author = 'Unknown'
            milestone = '-'
            comments_count = issue.get('comments', 0)
            
            # Get the HTML URL for clickable links
            html_url = issue.get('html_url', '')

            # Format dates
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_at = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    created_at = created_at[:16].replace('T', ' ')
                    
            if updated_at:
                try:
                    date_obj = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    updated_at = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    updated_at = updated_at[:16].replace('T', ' ')

            # Get author info
            if issue.get('user') and isinstance(issue['user'], dict):
                author = issue['user'].get('login', 'Unknown')

            # Get assignee info
            if issue.get('assignee') and isinstance(issue['assignee'], dict):
                assignee = issue['assignee'].get('login', 'Unknown')
            elif issue.get('assignees') and len(issue['assignees']) > 0:
                assignee = ', '.join([a.get('login', 'Unknown') for a in issue['assignees'][:2]])
                if len(issue['assignees']) > 2:
                    assignee += f' +{len(issue["assignees"]) - 2} more'
                    
            # Get milestone
            if issue.get('milestone') and isinstance(issue['milestone'], dict):
                milestone = issue['milestone'].get('title', '-')

            # Get labels with colors
            labels = []
            if issue.get('labels'):
                for label in issue['labels'][:3]:
                    name = label.get('name', '')
                    color = label.get('color', 'gray')
                    labels.append(f'<span style="background-color: #{color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-right: 3px;">{name}</span>')
                if len(issue['labels']) > 3:
                    labels.append(f'<span style="background-color: #gray; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;">+{len(issue["labels"]) - 3} more</span>')
            labels_str = ''.join(labels) if labels else '-'

            # Create clickable number with link
            number_display = f"#{number}"
            if html_url:
                number_display = f'<a href="{html_url}" target="_blank" style="font-weight: bold; text-decoration: none;">#{number}</a>'

            # Get body preview
            body_preview = ''
            if issue.get('body'):
                body = issue['body'][:100].replace('\n', ' ').replace('\r', ' ')
                body_preview = body + '...' if len(issue['body']) > 100 else body

            table_data.append({
                'Number': number_display,
                'Title': title,
                'State': f'<span style="background-color: {"#28a745" if state.lower() == "open" else "#dc3545"}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">{state}</span>',
                'Author': author,
                'Assignee': assignee,
                'Created': created_at,
                'Updated': updated_at,
                'Comments': comments_count,
                'Labels': labels_str,
                'Milestone': milestone,
                'Preview': body_preview,
                'URL': html_url  # Keep URL for external use
            })

    return pd.DataFrame(table_data)


def create_github_repositories_dataframe(repos_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for GitHub repositories with enhanced detail display."""
    if not repos_data:
        logger.info("Repository dataframe: No data provided")
        return pd.DataFrame()
    
    logger.info(f"Creating repository dataframe with {len(repos_data)} repositories")
    logger.info(f"First repo sample: {repos_data[0] if repos_data else 'None'}")

    # Prepare data for DataFrame
    table_data = []
    for repo in repos_data:
        if isinstance(repo, dict):
            name = repo.get('name', 'No name')
            full_name = repo.get('full_name', 'N/A')
            description = repo.get('description', 'No description')
            private = repo.get('private', False)
            fork = repo.get('fork', False)
            archived = repo.get('archived', False)
            language = repo.get('language', '-')
            stars = repo.get('stargazers_count', 0)
            forks = repo.get('forks_count', 0)
            open_issues = repo.get('open_issues_count', 0)
            size = repo.get('size', 0)
            created_at = repo.get('created_at', '')
            updated_at = repo.get('updated_at', '')
            
            # Get the HTML URL for clickable links
            html_url = repo.get('html_url', '')
            clone_url = repo.get('clone_url', '')

            # Format dates
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_at = date_obj.strftime('%Y-%m-%d')
                except:
                    created_at = created_at[:10]
                    
            if updated_at:
                try:
                    date_obj = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    updated_at = date_obj.strftime('%Y-%m-%d')
                except:
                    updated_at = updated_at[:10]

            # Create visibility badge
            visibility = "Private" if private else "Public"
            visibility_color = "#dc3545" if private else "#28a745"
            visibility_display = f'<span style="background-color: {visibility_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">{visibility}</span>'
            
            # Create status badges
            status_badges = []
            if fork:
                status_badges.append('<span style="background-color: #6f42c1; color: white; padding: 1px 6px; border-radius: 8px; font-size: 11px;">FORK</span>')
            if archived:
                status_badges.append('<span style="background-color: #6c757d; color: white; padding: 1px 6px; border-radius: 8px; font-size: 11px;">ARCHIVED</span>')
            status_display = ' '.join(status_badges) if status_badges else '-'

            # Create clickable name with link
            name_display = name
            if html_url:
                name_display = f'<a href="{html_url}" target="_blank" style="font-weight: bold; text-decoration: none;">{name}</a>'

            # Format description
            if description and len(description) > 80:
                description = description[:80] + '...'
            elif not description:
                description = '-'

            # Format language with color if available
            language_display = language
            if language:
                # Common language colors
                lang_colors = {
                    'Python': '#3776ab',
                    'JavaScript': '#f1e05a',
                    'TypeScript': '#2b7489',
                    'Java': '#b07219',
                    'C++': '#f34b7d',
                    'C': '#555555',
                    'Go': '#00ADD8',
                    'Rust': '#dea584',
                    'PHP': '#4F5D95',
                    'Ruby': '#701516',
                    'Swift': '#ffac45',
                    'Kotlin': '#F18E33'
                }
                color = lang_colors.get(language, '#6c757d')
                language_display = f'<span style="background-color: {color}; color: white; padding: 1px 6px; border-radius: 8px; font-size: 11px;">{language}</span>'

            # Format size (KB to MB if large)
            size_display = f"{size} KB"
            if size > 1024:
                size_mb = size / 1024
                size_display = f"{size_mb:.1f} MB"

            table_data.append({
                'Name': name_display,
                'Full Name': full_name,
                'Description': description,
                'Visibility': visibility_display,
                'Language': language_display,
                'Stars': f"⭐ {stars:,}",
                'Forks': f"🍴 {forks:,}",
                'Issues': f"❗ {open_issues:,}" if open_issues > 0 else "-",
                'Size': size_display,
                'Status': status_display,
                'Created': created_at,
                'Updated': updated_at,
                'URL': html_url,  # Keep URL for external use
                'Clone': clone_url  # Keep clone URL for external use
            })

    return pd.DataFrame(table_data)


def create_github_contents_dataframe(contents_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for GitHub repository contents with enhanced detail display."""
    if not contents_data:
        return pd.DataFrame()

    # Prepare data for DataFrame
    table_data = []
    for item in contents_data:
        if isinstance(item, dict):
            name = item.get('name', 'Unknown')
            path = item.get('path', name)
            item_type = item.get('type', 'unknown')
            size = item.get('size', 0)
            sha = item.get('sha', '')[:7] if item.get('sha') else ''
            download_url = item.get('download_url', '')
            html_url = item.get('html_url', '')
            
            # Create type icon and badge
            type_icon = {
                'file': '📄',
                'dir': '📁',
                'symlink': '🔗',
                'submodule': '📦'
            }.get(item_type, '❓')
            
            type_color = {
                'file': '#6c757d',
                'dir': '#007bff', 
                'symlink': '#17a2b8',
                'submodule': '#6f42c1'
            }.get(item_type, '#6c757d')
            
            type_display = f'{type_icon} <span style="background-color: {type_color}; color: white; padding: 1px 6px; border-radius: 8px; font-size: 11px;">{item_type.upper()}</span>'
            
            # Create clickable name with link
            name_display = name
            if html_url:
                name_display = f'<a href="{html_url}" target="_blank" style="font-weight: bold; text-decoration: none;">{name}</a>'
            
            # Format size
            size_display = '-'
            if item_type == 'file' and size > 0:
                if size < 1024:
                    size_display = f"{size} B"
                elif size < 1024 * 1024:
                    size_display = f"{size / 1024:.1f} KB"
                else:
                    size_display = f"{size / (1024 * 1024):.1f} MB"
            elif item_type == 'dir':
                size_display = 'Directory'
            
            # Determine file extension and language
            language_display = '-'
            if item_type == 'file' and '.' in name:
                ext = name.split('.')[-1].lower()
                lang_map = {
                    'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript', 
                    'java': 'Java', 'cpp': 'C++', 'c': 'C', 'go': 'Go',
                    'rs': 'Rust', 'php': 'PHP', 'rb': 'Ruby', 'swift': 'Swift',
                    'kt': 'Kotlin', 'md': 'Markdown', 'json': 'JSON',
                    'yml': 'YAML', 'yaml': 'YAML', 'xml': 'XML', 'html': 'HTML',
                    'css': 'CSS', 'scss': 'SCSS', 'sh': 'Shell', 'sql': 'SQL'
                }
                language = lang_map.get(ext, ext.upper())
                
                # Color coding for languages
                lang_colors = {
                    'Python': '#3776ab', 'JavaScript': '#f1e05a', 'TypeScript': '#2b7489',
                    'Java': '#b07219', 'C++': '#f34b7d', 'C': '#555555', 'Go': '#00ADD8',
                    'Rust': '#dea584', 'PHP': '#4F5D95', 'Ruby': '#701516',
                    'Markdown': '#083fa1', 'JSON': '#292929', 'YAML': '#cb171e'
                }
                color = lang_colors.get(language, '#6c757d')
                language_display = f'<span style="background-color: {color}; color: white; padding: 1px 6px; border-radius: 8px; font-size: 11px;">{language}</span>'
            
            # Path display (truncate if too long)
            path_display = path
            if len(path) > 50:
                path_display = '...' + path[-47:]
            
            table_data.append({
                'Type': type_display,
                'Name': name_display,
                'Path': path_display,
                'Size': size_display,
                'Language': language_display,
                'SHA': sha if sha else '-',
                'Download': f'<a href="{download_url}" target="_blank">⬇️</a>' if download_url else '-',
                'URL': html_url,  # Keep URL for external use
                'Raw Path': path  # Keep original path for sorting
            })

    # Sort: directories first, then files, both alphabetically
    table_data.sort(key=lambda x: (0 if '📁' in x['Type'] else 1, x['Raw Path'].lower()))

    # Remove raw path from display
    for item in table_data:
        del item['Raw Path']

    return pd.DataFrame(table_data)


def create_jira_issues_dataframe(issues_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for Jira issues with enhanced detail display."""
    if not issues_data:
        return pd.DataFrame()

    table_data = []
    for issue in issues_data:
        if isinstance(issue, dict):
            # Extract key
            key = issue.get('key', 'N/A')

            # Extract fields with multiple fallbacks
            fields = issue.get('fields', {})

            # Summary extraction with comprehensive fallbacks
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

            # Status extraction with enhanced styling
            status = 'Unknown'
            status_color = '#6c757d'  # Default gray
            status_obj = fields.get('status')
            if status_obj and isinstance(status_obj, dict):
                status = status_obj.get('name', 'Unknown')
                # Color coding based on status category
                status_category = status_obj.get('statusCategory', {}).get('key', 'indeterminate')
                if status_category == 'new':
                    status_color = '#007bff'  # Blue for new
                elif status_category == 'indeterminate':
                    status_color = '#ffc107'  # Yellow for in progress
                elif status_category == 'done':
                    status_color = '#28a745'  # Green for done
            elif isinstance(status_obj, str):
                status = status_obj

            # Issue type extraction with icon
            issue_type = '-'
            issue_type_icon = '📄'
            issuetype_obj = fields.get('issuetype') or fields.get('issueType') or fields.get('type')
            if issuetype_obj and isinstance(issuetype_obj, dict):
                issue_type = issuetype_obj.get('name', '-')
                # Add icons based on issue type
                type_name = issue_type.lower()
                if 'bug' in type_name:
                    issue_type_icon = '🐛'
                elif 'story' in type_name or 'feature' in type_name:
                    issue_type_icon = '📖'
                elif 'task' in type_name:
                    issue_type_icon = '✅'
                elif 'epic' in type_name:
                    issue_type_icon = '🎯'
                elif 'improvement' in type_name:
                    issue_type_icon = '⚡'
            elif isinstance(issuetype_obj, str):
                issue_type = issuetype_obj

            # Priority extraction with enhanced display
            priority = '-'
            priority_color = '#6c757d'
            priority_obj = fields.get('priority')
            if priority_obj and isinstance(priority_obj, dict):
                priority = priority_obj.get('name', '-')
                priority_name = priority.lower()
                if 'highest' in priority_name or 'critical' in priority_name:
                    priority_color = '#dc3545'  # Red
                elif 'high' in priority_name:
                    priority_color = '#fd7e14'  # Orange
                elif 'medium' in priority_name:
                    priority_color = '#ffc107'  # Yellow
                elif 'low' in priority_name:
                    priority_color = '#28a745'  # Green
            elif isinstance(priority_obj, str):
                priority = priority_obj

            # Assignee extraction with enhanced info
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

            # Reporter info
            reporter = 'Unknown'
            reporter_obj = fields.get('reporter')
            if reporter_obj and isinstance(reporter_obj, dict):
                reporter = reporter_obj.get('displayName', reporter_obj.get('name', 'Unknown'))
                if len(reporter) > 15:
                    reporter = reporter[:12] + "..."

            # Date extractions with enhanced formatting
            created = ''
            updated = ''
            
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
                        created = date_obj.strftime('%Y-%m-%d %H:%M')
                        break
                    except:
                        created = str(candidate)[:16].replace('T', ' ')
                        if len(created) >= 10:
                            break
                            
            updated_candidates = [
                fields.get('updated'),
                fields.get('updatedDate'),
                issue.get('updated'),
                issue.get('updatedDate')
            ]
            for candidate in updated_candidates:
                if candidate:
                    try:
                        date_obj = datetime.fromisoformat(str(candidate).replace('Z', '+00:00'))
                        updated = date_obj.strftime('%Y-%m-%d %H:%M')
                        break
                    except:
                        updated = str(candidate)[:16].replace('T', ' ')
                        if len(updated) >= 10:
                            break

            # Labels extraction
            labels_str = '-'
            if fields.get('labels'):
                labels = fields['labels'][:3]  # Show max 3 labels
                label_list = []
                for label in labels:
                    if isinstance(label, str):
                        label_list.append(f'<span style="background-color: #007bff; color: white; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 2px;">{label}</span>')
                if len(fields['labels']) > 3:
                    label_list.append(f'<span style="background-color: #6c757d; color: white; padding: 1px 6px; border-radius: 3px; font-size: 10px;">+{len(fields["labels"]) - 3}</span>')
                labels_str = ''.join(label_list)

            # Components
            components_str = '-'
            if fields.get('components'):
                components = [comp.get('name', '') for comp in fields['components'][:2]]
                components_str = ', '.join(components)
                if len(fields['components']) > 2:
                    components_str += f' +{len(fields["components"]) - 2} more'

            # Description preview
            description_preview = ''
            if fields.get('description'):
                desc = str(fields['description'])[:80].replace('\n', ' ').replace('\r', ' ')
                description_preview = desc + '...' if len(str(fields['description'])) > 80 else desc

            # Generate Jira URL if we have the Jira base URL
            jira_url = ""
            if st.session_state.get('jira_client') and hasattr(st.session_state.jira_client, 'base'):
                jira_base = st.session_state.jira_client.base
                jira_url = f"{jira_base}/browse/{key}"

            # Create clickable key with link
            key_display = key
            if jira_url:
                key_display = f'<a href="{jira_url}" target="_blank" style="font-weight: bold; text-decoration: none;">{key}</a>'

            table_data.append({
                'Key': key_display,
                'Summary': summary,
                'Status': f'<span style="background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">{status}</span>',
                'Type': f'{issue_type_icon} {issue_type}',
                'Priority': f'<span style="background-color: {priority_color}; color: white; padding: 1px 6px; border-radius: 8px; font-size: 11px;">{priority}</span>',
                'Reporter': reporter,
                'Assignee': assignee,
                'Created': created,
                'Updated': updated,
                'Labels': labels_str,
                'Components': components_str,
                'Description': description_preview,
                'URL': jira_url  # Keep URL for external use
            })

    return pd.DataFrame(table_data)


def create_notion_pages_dataframe(pages_data: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame for Notion pages with enhanced detail display."""
    if not pages_data:
        return pd.DataFrame()

    table_data = []
    for page in pages_data:
        if isinstance(page, dict):
            # Extract title from properties
            title = "Untitled"
            status = "-"
            status_color = "#6c757d"
            created = ""
            updated = ""
            page_icon = "📄"
            tags = []
            author = "Unknown"
            
            # Extract page URL
            page_url = page.get('url', '')
            page_id = page.get('id', '')

            properties = page.get('properties', {})
            for prop_name, prop_data in properties.items():
                prop_type = prop_data.get('type', '')
                
                if prop_type == 'title':
                    title_texts = prop_data.get('title', [])
                    if title_texts:
                        title = ''.join([t.get('plain_text', '') for t in title_texts])
                        
                elif 'status' in prop_name.lower() and prop_type == 'select':
                    select_data = prop_data.get('select')
                    if select_data:
                        status = select_data.get('name', '-')
                        # Color based on status
                        status_lower = status.lower()
                        if status_lower in ['done', 'completed', 'finished']:
                            status_color = '#28a745'  # Green
                        elif status_lower in ['in progress', 'doing', 'working']:
                            status_color = '#ffc107'  # Yellow
                        elif status_lower in ['not started', 'todo', 'backlog']:
                            status_color = '#007bff'  # Blue
                        elif status_lower in ['blocked', 'on hold']:
                            status_color = '#dc3545'  # Red
                            
                elif prop_type == 'multi_select':
                    multi_select_data = prop_data.get('multi_select', [])
                    if multi_select_data and 'tag' in prop_name.lower():
                        tags = [item.get('name', '') for item in multi_select_data[:3]]
                        
                elif prop_type == 'people' and 'author' in prop_name.lower():
                    people_data = prop_data.get('people', [])
                    if people_data:
                        author = people_data[0].get('name', 'Unknown')

            # Get created and updated dates
            if page.get('created_time'):
                try:
                    date_obj = datetime.fromisoformat(page['created_time'].replace('Z', '+00:00'))
                    created = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    created = page['created_time'][:16].replace('T', ' ')
                    
            if page.get('last_edited_time'):
                try:
                    date_obj = datetime.fromisoformat(page['last_edited_time'].replace('Z', '+00:00'))
                    updated = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    updated = page['last_edited_time'][:16].replace('T', ' ')

            # Get page icon
            if page.get('icon'):
                if page['icon'].get('type') == 'emoji':
                    page_icon = page['icon'].get('emoji', '📄')
                elif page['icon'].get('type') == 'external':
                    page_icon = '🔗'
                elif page['icon'].get('type') == 'file':
                    page_icon = '🖼️'

            # Format tags
            tags_str = '-'
            if tags:
                tag_elements = []
                for tag in tags:
                    tag_elements.append(f'<span style="background-color: #e3f2fd; color: #1976d2; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-right: 2px;">{tag}</span>')
                tags_str = ''.join(tag_elements)

            # Get parent info
            parent_type = 'Workspace'
            parent_obj = page.get('parent', {})
            if parent_obj.get('type') == 'database_id':
                parent_type = 'Database'
            elif parent_obj.get('type') == 'page_id':
                parent_type = 'Page'

            # Create clickable title with link
            title_display = f'{page_icon} {title}'
            if page_url:
                title_display = f'<a href="{page_url}" target="_blank" style="text-decoration: none;">{page_icon} {title}</a>'

            table_data.append({
                'Title': title_display,
                'Status': f'<span style="background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">{status}</span>',
                'Type': parent_type,
                'Author': author,
                'Created': created,
                'Updated': updated,
                'Tags': tags_str,
                'ID': page_id[:8] + '...' if page_id else '-',
                'URL': page_url  # Keep URL for external use
            })

    return pd.DataFrame(table_data)


def format_execution_results_with_tables(results: List[Dict]) -> Tuple[str, List[pd.DataFrame], List[str]]:
    """
    Format results for display with accurate success messages matching the actual operations.

    FIXED: Enhanced result formatting with better handling of all GitHub/Jira operations
    and improved error messaging for failed searches.
    """
    if not results:
        return "", [], []

    summary_parts = []
    dataframes = []
    table_titles = []

    for result in results:
        action = result['action']
        success = result['success']
        service = result.get('service', '').title()
        raw_action = result.get('raw_action', '')
        raw_args = result.get('raw_args', {})

        if success:
            # For list operations, default to empty list instead of empty dict
            raw_action = result.get('raw_action', '')
            is_list_operation = any(action in raw_action for action in ['list_', 'search_'])
            default_data = [] if is_list_operation else {}
            
            # Debug the exact result structure
            logger.info(f"DEBUG - Full result structure: {result}")
            logger.info(f"DEBUG - result['result'] keys: {list(result['result'].keys()) if isinstance(result.get('result'), dict) else 'Not a dict'}")
            
            # Try multiple ways to extract the data
            result_data = result['result']
            
            # Method 1: Look for 'data' field
            data = result_data.get('data', default_data)
            
            # Method 2: If data is empty/default and result_data itself is a list, use it directly
            if data == default_data and isinstance(result_data, list):
                data = result_data
                logger.info("DEBUG - Using result_data directly as it's a list")
                logger.info(f"DEBUG - result_data length: {len(result_data)}, first item: {result_data[0] if result_data else 'Empty list'}")
            
            # Method 3: If result_data has 'success' field, the actual data might be the whole result_data minus metadata
            elif data == default_data and isinstance(result_data, dict) and 'success' in result_data:
                # Remove metadata fields to see if there's actual data
                metadata_fields = {'success', 'data', 'error', 'status_code'}
                potential_data = {k: v for k, v in result_data.items() if k not in metadata_fields}
                if potential_data:
                    logger.info(f"DEBUG - Found potential data fields: {list(potential_data.keys())}")
                    # If there's only one non-metadata field and it's a list, use it
                    if len(potential_data) == 1:
                        field_name, field_value = next(iter(potential_data.items()))
                        if isinstance(field_value, list):
                            data = field_value
                            logger.info(f"DEBUG - Using field '{field_name}' as data")
            
            logger.info(f"DEBUG - Final extracted data type: {type(data)}, Length: {len(data) if hasattr(data, '__len__') else 'No length'}")
            if isinstance(data, list) and data:
                logger.info(f"DEBUG - First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
            elif isinstance(data, dict):
                logger.info(f"DEBUG - Data dict keys: {list(data.keys())}")

            # GitHub actions with accurate messaging
            if service.lower() == 'github':
                if raw_action == 'create_issue':
                    if isinstance(data, dict) and 'number' in data:
                        html_url = data.get('html_url', '')
                        summary_parts.append(f"GitHub: Issue #{data['number']} created")
                        if html_url:
                            summary_parts.append(f"   Link: {html_url}")

                elif raw_action == 'get_issue':
                    issue_num = raw_args.get('issue_number', 'N/A')
                    summary_parts.append(f"GitHub: Issue #{issue_num} details")
                    # Add to dataframe for display
                    if isinstance(data, dict):
                        df = create_github_issues_dataframe([data])
                        if not df.empty:
                            dataframes.append(df)
                            table_titles.append("GitHub Issues")

                elif raw_action == 'update_issue':
                    issue_num = raw_args.get('issue_number', 'N/A')
                    state = raw_args.get('state', '')
                    html_url = ''
                    if isinstance(data, dict) and 'html_url' in data:
                        html_url = data['html_url']
                    elif st.session_state.gh_owner and st.session_state.gh_repo:
                        html_url = f"https://github.com/{st.session_state.gh_owner}/{st.session_state.gh_repo}/issues/{issue_num}"

                    if state == 'closed':
                        summary_parts.append(f"GitHub: Issue #{issue_num} closed")
                    elif state == 'open':
                        summary_parts.append(f"GitHub: Issue #{issue_num} reopened")
                    else:
                        summary_parts.append(f"GitHub: Issue #{issue_num} updated")

                    if html_url:
                        summary_parts.append(f"   Link: {html_url}")

                elif raw_action == 'add_issue_comment':
                    issue_num = raw_args.get('issue_number', 'N/A')
                    comment_url = ''
                    if isinstance(data, dict) and 'html_url' in data:
                        comment_url = data['html_url']
                    elif st.session_state.gh_owner and st.session_state.gh_repo:
                        comment_url = f"https://github.com/{st.session_state.gh_owner}/{st.session_state.gh_repo}/issues/{issue_num}"

                    summary_parts.append(f"GitHub: Comment added to #{issue_num}")
                    if comment_url:
                        summary_parts.append(f"   Link: {comment_url}")

                elif raw_action == 'create_branch':
                    branch = raw_args.get('branch', 'N/A')
                    from_branch = raw_args.get('from_branch', 'main')
                    summary_parts.append(f"GitHub: Branch '{branch}' created from '{from_branch}'")

                elif raw_action == 'get_file_contents':
                    # FIXED: Enhanced file content display with proper preview
                    path = raw_args.get('path', 'N/A')
                    ref = raw_args.get('ref', 'main')
                    summary_parts.append(f"GitHub: Retrieved {path} (ref={ref})")

                    # Show file preview
                    if isinstance(data, dict) and 'content' in data:
                        try:
                            content = base64.b64decode(data['content']).decode('utf-8')
                            # Show more content for better visibility
                            preview = content[:500] + "..." if len(content) > 500 else content
                            summary_parts.append(f"   File Content:\n```\n{preview}\n```")

                            # Also show file size and type
                            summary_parts.append(f"   Size: {data.get('size', 'Unknown')} bytes")

                        except Exception as e:
                            summary_parts.append(f"   Binary file or decode error: {str(e)}")
                            if 'download_url' in data:
                                summary_parts.append(f"   Download: {data['download_url']}")

                elif raw_action == 'create_or_update_file':
                    path = raw_args.get('path', 'N/A')
                    branch = raw_args.get('branch', 'main')
                    summary_parts.append(f"GitHub: File {path} updated on {branch}")

                elif raw_action == 'create_pull_request':
                    # FIXED: Enhanced PR creation success message
                    if isinstance(data, dict) and 'number' in data:
                        html_url = data.get('html_url', '')
                        title = data.get('title', 'N/A')
                        head = data.get('head', {}).get('ref', 'N/A')
                        base = data.get('base', {}).get('ref', 'N/A')

                        summary_parts.append(f"GitHub: PR #{data['number']} created")
                        summary_parts.append(f"   Title: {title}")
                        summary_parts.append(f"   Changes: {head} → {base}")
                        if html_url:
                            summary_parts.append(f"   Link: {html_url}")

                elif raw_action == 'list_branches':
                    # FIXED: Enhanced branch listing with better formatting
                    if isinstance(data, list):
                        summary_parts.append(f"GitHub: Found {len(data)} branches")

                        # Show branch list with protection status
                        branch_info = []
                        for branch in data[:10]:  # Show first 10
                            name = branch.get('name', 'Unknown')
                            protected = ' (protected)' if branch.get('protected', False) else ''
                            branch_info.append(f"• {name}{protected}")

                        if branch_info:
                            branch_list = '\n'.join(branch_info)
                            if len(data) > 10:
                                branch_list += f"\n... and {len(data) - 10} more"
                            summary_parts.append(f"\nBranches:\n{branch_list}")
                    else:
                        summary_parts.append(f"GitHub: Branch list retrieved")

                elif raw_action == 'list_tags':
                    # FIXED: Enhanced tag listing with better formatting
                    if isinstance(data, list):
                        summary_parts.append(f"GitHub: Found {len(data)} tags")

                        if len(data) == 0:
                            summary_parts.append("   No tags found in this repository")
                        else:
                            # Show tag list with commit info
                            tag_info = []
                            for tag in data[:10]:  # Show first 10
                                name = tag.get('name', 'Unknown')
                                commit_sha = tag.get('commit', {}).get('sha', '')[:7] if tag.get('commit') else ''
                                commit_info = f' ({commit_sha})' if commit_sha else ''
                                tag_info.append(f"• {name}{commit_info}")

                            if tag_info:
                                tag_list = '\n'.join(tag_info)
                                if len(data) > 10:
                                    tag_list += f"\n... and {len(data) - 10} more"
                                summary_parts.append(f"\nTags:\n{tag_list}")
                    else:
                        summary_parts.append(f"GitHub: Tag list retrieved")

                elif raw_action == 'search_issues':
                    # FIXED: Enhanced search results handling with proper "not found" messaging
                    if isinstance(data, dict):
                        # Handle GitHub search API response format
                        items = data.get('items', [])
                        total_count = data.get('total_count', len(items))

                        if total_count == 0 or len(items) == 0:
                            # FIXED: Proper "not found" message for empty search results
                            query = raw_args.get('query', '')
                            summary_parts.append(f"GitHub: Search completed - no issues found")
                            summary_parts.append(f"   Query: {query}")
                            summary_parts.append("   No issues matched your search criteria")
                        else:
                            summary_parts.append(f"GitHub: Found {total_count} issues")
                            if items:
                                df = create_github_issues_dataframe(items)
                                if not df.empty:
                                    dataframes.append(df)
                                    table_titles.append("GitHub Issues")

                                # Add quick summary of top results
                                issue_titles = [item.get('title', 'No title')[:50] for item in items[:3]]
                                if len(items) > 3:
                                    summary_parts.append(
                                        f"   Top results: {', '.join(issue_titles)}... and {len(items) - 3} more")
                                else:
                                    summary_parts.append(f"   Results: {', '.join(issue_titles)}")
                    else:
                        # Handle other response formats
                        summary_parts.append(f"GitHub: Search completed")
                        if isinstance(data, list):
                            if len(data) == 0:
                                summary_parts.append("   No issues found matching your criteria")
                            else:
                                df = create_github_issues_dataframe(data)
                                if not df.empty:
                                    dataframes.append(df)
                                    table_titles.append("GitHub Issues")

                elif raw_action == 'list_issues':
                    # Handle list results (preserve existing functionality)
                    if isinstance(data, list):
                        summary_parts.append(f"GitHub: Found {len(data)} issues")
                        if data:
                            df = create_github_issues_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("GitHub Issues")
                                # Show quick summary in chat
                                open_count = sum(1 for issue in data if issue.get('state', '').lower() == 'open')
                                closed_count = len(data) - open_count
                                summary_parts.append(f"   Summary: {open_count} open, {closed_count} closed")
                    else:
                        summary_parts.append(f"GitHub: Issues retrieved")
                        # Try to extract issues data anyway
                        if data:
                            try_data = [data] if isinstance(data, dict) else data
                            df = create_github_issues_dataframe(try_data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("GitHub Issues")

                elif raw_action == 'list_repositories':
                    # Handle repository listing with enhanced debugging
                    logger.info(f"Repository data type: {type(data)}")
                    logger.info(f"Repository data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    
                    if isinstance(data, list):
                        summary_parts.append(f"GitHub: Found {len(data)} repositories")
                        if data:
                            logger.info(f"First repo keys: {list(data[0].keys()) if data and isinstance(data[0], dict) else 'Not a dict'}")
                            df = create_github_repositories_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("GitHub Repositories")
                                # Show quick summary in chat
                                public_count = sum(1 for repo in data if not repo.get('private', True))
                                private_count = len(data) - public_count
                                summary_parts.append(f"   Summary: {public_count} public, {private_count} private")
                            else:
                                summary_parts.append("   Note: Repository data could not be formatted into table")
                        else:
                            summary_parts.append("   No repository data found")
                    else:
                        summary_parts.append(f"GitHub: Repositories retrieved (type: {type(data)})")
                        logger.info(f"Non-list data: {data}")
                        
                        # More aggressive data extraction for debugging
                        if data:
                            try_data = []
                            if isinstance(data, dict):
                                # Maybe repositories are nested in the dict
                                for key, value in data.items():
                                    if isinstance(value, list) and value and isinstance(value[0], dict):
                                        # Found a list of dicts, this might be repositories
                                        if any(repo_field in str(value[0]) for repo_field in ['name', 'full_name', 'owner', 'clone_url']):
                                            try_data = value
                                            logger.info(f"Found repositories in field '{key}'")
                                            break
                            elif isinstance(data, list):
                                try_data = data
                            
                            if try_data:
                                df = create_github_repositories_dataframe(try_data)
                                if not df.empty:
                                    dataframes.append(df)
                                    table_titles.append("GitHub Repositories")
                                    summary_parts.append(f"   Found {len(try_data)} repositories")
                                else:
                                    summary_parts.append("   Note: Repository data could not be formatted into table")
                                    # Show raw data structure for debugging
                                    if try_data:
                                        summary_parts.append(f"   Debug: Sample item keys: {list(try_data[0].keys()) if isinstance(try_data[0], dict) else 'Not a dict'}")
                            else:
                                summary_parts.append("   No recognizable repository data found")
                                summary_parts.append(f"   Debug: Data content: {str(data)[:200]}...")

                elif raw_action == 'list_repository_contents':
                    # Handle repository contents listing with enhanced debugging
                    owner = raw_args.get('owner', 'Repository')
                    repo = raw_args.get('repo', '')
                    path = raw_args.get('path', '')
                    ref = raw_args.get('ref', 'main')
                    
                    repo_display = f"{owner}/{repo}" if owner and repo else "Repository"
                    path_display = f" at path '{path}'" if path else ""
                    
                    logger.info(f"Repository contents data type: {type(data)}")
                    logger.info(f"Repository contents data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    logger.info(f"Repository contents data content: {str(data)[:500]}...")
                    logger.info(f"Is data a list? {isinstance(data, list)}")
                    logger.info(f"Is data equal to default_data? {data == default_data}")
                    logger.info(f"default_data value: {default_data}")
                    
                    if isinstance(data, list):
                        summary_parts.append(f"GitHub: Found {len(data)} items in {repo_display}{path_display}")
                        if data:
                            logger.info(f"First content item keys: {list(data[0].keys()) if data and isinstance(data[0], dict) else 'Not a dict'}")
                            df = create_github_contents_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append(f"Contents of {repo_display}{path_display}")
                                # Show quick summary
                                files_count = sum(1 for item in data if item.get('type') == 'file')
                                dirs_count = sum(1 for item in data if item.get('type') == 'dir')
                                summary_parts.append(f"   Summary: {files_count} files, {dirs_count} directories")
                            else:
                                summary_parts.append("   Note: Contents data could not be formatted into table")
                        else:
                            summary_parts.append("   No contents found")
                    else:
                        summary_parts.append(f"GitHub: Contents retrieved for {repo_display} (type: {type(data)})")
                        logger.info(f"Non-list contents data: {data}")
                        # Try to extract contents data anyway with more aggressive search
                        if data:
                            try_data = []
                            if isinstance(data, dict):
                                # Maybe contents are nested in the dict
                                for key, value in data.items():
                                    if isinstance(value, list) and value and isinstance(value[0], dict):
                                        # Found a list of dicts, this might be contents
                                        if any(content_field in str(value[0]) for content_field in ['name', 'path', 'type', 'size']):
                                            try_data = value
                                            logger.info(f"Found repository contents in field '{key}'")
                                            break
                            elif isinstance(data, list):
                                try_data = data
                            
                            if try_data:
                                df = create_github_contents_dataframe(try_data)
                                if not df.empty:
                                    dataframes.append(df)
                                    table_titles.append(f"Contents of {repo_display}{path_display}")
                                    summary_parts.append(f"   Found {len(try_data)} items")
                                else:
                                    summary_parts.append("   Note: Contents data could not be formatted into table")
                                    # Show debug info
                                    if try_data:
                                        summary_parts.append(f"   Debug: Sample item keys: {list(try_data[0].keys()) if isinstance(try_data[0], dict) else 'Not a dict'}")
                            else:
                                summary_parts.append("   No recognizable contents data found")
                                summary_parts.append(f"   Debug: Data content: {str(data)[:200]}...")

                elif raw_action == 'search_file_across_repositories':
                    # Handle file search across repositories
                    filename = raw_args.get('filename', 'Unknown file')
                    
                    if isinstance(data, dict):
                        total_found = data.get('total_found', 0)
                        found_files = data.get('found_in', [])
                        
                        if total_found > 0:
                            summary_parts.append(f"GitHub: Found '{filename}' in {total_found} location(s)")
                            
                            # Show where files were found
                            for i, file_info in enumerate(found_files[:3]):  # Show first 3
                                repo_name = file_info.get('repository', 'Unknown')
                                file_path = file_info.get('path', filename)
                                size = file_info.get('size', 0)
                                size_str = f" ({size} bytes)" if size else ""
                                summary_parts.append(f"   📄 {repo_name}/{file_path}{size_str}")
                            
                            if len(found_files) > 3:
                                summary_parts.append(f"   ... and {len(found_files) - 3} more")
                            
                            # Automatically get contents of the first file found
                            if total_found >= 1:
                                try:
                                    file_info = found_files[0]
                                    owner = file_info.get('owner')
                                    repo = file_info.get('repo')
                                    path = file_info.get('path')
                                    
                                    if owner and repo and path:
                                        summary_parts.append(f"\n📖 Contents of {filename}:")
                                        summary_parts.append("=" * 50)
                                        
                                        # Get file contents using the GitHub client
                                        github_client = st.session_state.get('github_client')
                                        if github_client:
                                            file_contents = github_client.get_file_contents(path=path, owner=owner, repo=repo)
                                            
                                            if isinstance(file_contents, dict) and 'content' in file_contents:
                                                # Decode base64 content
                                                import base64
                                                try:
                                                    content = base64.b64decode(file_contents['content']).decode('utf-8')
                                                    
                                                    # Show file contents with line numbers
                                                    lines = content.split('\n')
                                                    for line_num, line in enumerate(lines[:100], 1):  # Show first 100 lines
                                                        summary_parts.append(f"{line_num:3d}: {line}")
                                                    
                                                    if len(lines) > 100:
                                                        summary_parts.append(f"... ({len(lines) - 100} more lines)")
                                                    
                                                    summary_parts.append("=" * 50)
                                                except Exception as decode_error:
                                                    summary_parts.append(f"   Error decoding file: {str(decode_error)}")
                                                    summary_parts.append("   File might be binary or use different encoding")
                                            else:
                                                summary_parts.append("   Could not decode file contents")
                                                
                                except Exception as e:
                                    summary_parts.append(f"   Error retrieving contents: {str(e)}")
                        else:
                            summary_parts.append(f"GitHub: '{filename}' not found in any repositories")
                            summary_parts.append("   Try checking if the filename is correct or search with a different name")
                    else:
                        summary_parts.append(f"GitHub: File search completed for '{filename}'")

                else:
                    summary_parts.append(f"GitHub: {action} completed")
                    # Try to display any issues data even for unknown actions
                    if isinstance(data, list) and data:
                        # Check if it looks like issues data
                        first_item = data[0] if data else {}
                        if isinstance(first_item, dict) and ('number' in first_item or 'title' in first_item):
                            df = create_github_issues_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("GitHub Issues")
                    elif isinstance(data, dict):
                        # Check if it's a single issue
                        if 'number' in data and 'title' in data:
                            df = create_github_issues_dataframe([data])
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("GitHub Issues")

            # Jira actions with accurate messaging
            elif service.lower() == 'jira':
                if raw_action == 'create_issue':
                    if isinstance(data, dict) and 'key' in data:
                        issue_key = data['key']
                        summary_parts.append(f"Jira: {issue_key} created")
                        # Add Jira URL if available
                        if st.session_state.get('jira_client') and hasattr(st.session_state.jira_client, 'base'):
                            jira_url = f"{st.session_state.jira_client.base}/browse/{issue_key}"
                            summary_parts.append(f"   Link: {jira_url}")

                elif raw_action == 'transition_issue':
                    # FIXED: Enhanced transition success messaging
                    issue_key = raw_args.get('issue_key', 'N/A')
                    transition_id = raw_args.get('transition_id', 'N/A')

                    # Try to get the transition name for better messaging
                    transition_name = "new status"
                    if isinstance(data, dict) and 'transition' in data:
                        transition_name = data['transition'].get('name', transition_name)

                    summary_parts.append(f"Jira: {issue_key} transitioned to {transition_name}")
                    # Add link
                    if st.session_state.get('jira_client') and hasattr(st.session_state.jira_client, 'base'):
                        jira_url = f"{st.session_state.jira_client.base}/browse/{issue_key}"
                        summary_parts.append(f"   Link: {jira_url}")

                elif raw_action == 'add_comment':
                    issue_key = raw_args.get('issue_key', 'N/A')
                    summary_parts.append(f"Jira: Comment added to {issue_key}")
                    # Add link
                    if st.session_state.get('jira_client') and hasattr(st.session_state.jira_client, 'base'):
                        jira_url = f"{st.session_state.jira_client.base}/browse/{issue_key}"
                        summary_parts.append(f"   Link: {jira_url}")

                elif raw_action == 'get_issue':
                    # Handle individual issue retrieval
                    issue_key = raw_args.get('issue_key', 'N/A')
                    if isinstance(data, dict):
                        # Extract issue details for display
                        actual_key = data.get('key', issue_key)
                        summary_parts.append(f"Jira: Issue {actual_key} details retrieved")
                        
                        # Add link
                        if st.session_state.get('jira_client') and hasattr(st.session_state.jira_client, 'base'):
                            jira_url = f"{st.session_state.jira_client.base}/browse/{actual_key}"
                            summary_parts.append(f"   Link: {jira_url}")
                        
                        # Add to dataframe for detailed display
                        df = create_jira_issues_dataframe([data])
                        if not df.empty:
                            dataframes.append(df)
                            table_titles.append("Jira Issue Details")
                        
                        # Also show key issue info in summary
                        fields = data.get('fields', {})
                        if fields:
                            # Show summary/title
                            if 'summary' in fields:
                                summary_parts.append(f"   Summary: {fields['summary']}")
                            
                            # Show status
                            status_obj = fields.get('status', {})
                            if isinstance(status_obj, dict) and 'name' in status_obj:
                                status = status_obj['name']
                                summary_parts.append(f"   Status: {status}")
                            
                            # Show assignee
                            assignee_obj = fields.get('assignee')
                            if assignee_obj and isinstance(assignee_obj, dict):
                                assignee = assignee_obj.get('displayName', assignee_obj.get('name', 'Unknown'))
                                summary_parts.append(f"   Assignee: {assignee}")
                            elif not assignee_obj:
                                summary_parts.append(f"   Assignee: Unassigned")
                    else:
                        summary_parts.append(f"Jira: Issue {issue_key} details retrieved")

                elif raw_action == 'search':
                    if isinstance(data, list):
                        summary_parts.append(f"Jira: Found {len(data)} issues")
                        if data:
                            df = create_jira_issues_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Jira Issues")
                                # Add status breakdown
                                status_counts = {}
                                for issue in data:
                                    fields = issue.get('fields', {})
                                    status_obj = fields.get('status', {})
                                    status = status_obj.get('name', 'Unknown') if isinstance(status_obj, dict) else str(
                                        status_obj)
                                    status_counts[status] = status_counts.get(status, 0) + 1

                                status_summary = ', '.join(
                                    [f"{count} {status}" for status, count in status_counts.items()])
                                summary_parts.append(f"   Summary: {status_summary}")
                    else:
                        summary_parts.append(f"Jira: Search completed")
                        # Try to extract data anyway
                        if data:
                            # Handle case where data might be wrapped in another structure
                            if isinstance(data, dict) and 'issues' in data:
                                issues = data['issues']
                                if issues:
                                    df = create_jira_issues_dataframe(issues)
                                    if not df.empty:
                                        dataframes.append(df)
                                        table_titles.append("Jira Issues")
                            elif isinstance(data, dict):
                                # Single issue result
                                df = create_jira_issues_dataframe([data])
                                if not df.empty:
                                    dataframes.append(df)
                                    table_titles.append("Jira Issues")

                elif raw_action == 'whoami':
                    if isinstance(data, dict) and 'displayName' in data:
                        summary_parts.append(f"Jira: User - {data['displayName']}")
                    else:
                        summary_parts.append(f"Jira: User info retrieved")

                elif raw_action in ['project_info', 'get_project_details']:
                    # Handle project details results
                    if isinstance(data, dict):
                        project_key = data.get('key', raw_args.get('project_key', 'Unknown'))
                        project_name = data.get('name', project_key)
                        project_type = data.get('projectTypeKey', 'Unknown')
                        
                        summary_parts.append(f"Jira: Project '{project_name}' ({project_key}) details retrieved")
                        
                        # Add key project information
                        if 'description' in data and data['description']:
                            description = str(data['description'])
                            if len(description) > 100:
                                description = description[:100] + "..."
                            summary_parts.append(f"   Description: {description}")
                        
                        if 'lead' in data and data['lead']:
                            lead = data['lead']
                            if isinstance(lead, dict):
                                lead_name = lead.get('displayName', lead.get('name', 'Unknown'))
                            else:
                                lead_name = str(lead)
                            summary_parts.append(f"   Project Lead: {lead_name}")
                        
                        summary_parts.append(f"   Project Type: {project_type}")
                        
                        if 'url' in data:
                            summary_parts.append(f"   Project URL: {data['url']}")
                        elif st.session_state.get('jira_client') and hasattr(st.session_state.jira_client, 'base'):
                            project_url = f"{st.session_state.jira_client.base}/browse/{project_key}"
                            summary_parts.append(f"   Browse Project: {project_url}")
                            
                        # Show components if available
                        if 'components' in data and data['components']:
                            components = []
                            for comp in data['components'][:3]:
                                if isinstance(comp, dict):
                                    components.append(comp.get('name', 'Unknown'))
                                else:
                                    components.append(str(comp))
                            comp_text = ', '.join(components)
                            if len(data['components']) > 3:
                                comp_text += f" and {len(data['components']) - 3} more"
                            summary_parts.append(f"   Components: {comp_text}")
                        
                        # Show versions if available
                        if 'versions' in data and data['versions']:
                            versions = []
                            for ver in data['versions'][:3]:
                                if isinstance(ver, dict):
                                    versions.append(ver.get('name', 'Unknown'))
                                else:
                                    versions.append(str(ver))
                            ver_text = ', '.join(versions)
                            if len(data['versions']) > 3:
                                ver_text += f" and {len(data['versions']) - 3} more"
                            summary_parts.append(f"   Versions: {ver_text}")
                    else:
                        project_key = raw_args.get('project_key', 'Unknown')
                        summary_parts.append(f"Jira: Project '{project_key}' details retrieved")

                else:
                    summary_parts.append(f"Jira: {action} completed")
                    # Try to display any Jira data even for unknown actions
                    if isinstance(data, list) and data:
                        df = create_jira_issues_dataframe(data)
                        if not df.empty:
                            dataframes.append(df)
                            table_titles.append("Jira Issues")
                    elif isinstance(data, dict):
                        if 'issues' in data:
                            issues = data['issues']
                            df = create_jira_issues_dataframe(issues)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Jira Issues")
                        elif 'key' in data and 'fields' in data:
                            # Single issue
                            df = create_jira_issues_dataframe([data])
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Jira Issues")

            # Notion actions
            elif service.lower() == 'notion':
                if raw_action == 'create_page':
                    # Handle page creation results
                    if isinstance(data, dict):
                        page_id = data.get('id', 'Unknown')
                        page_url = data.get('url', '')
                        
                        # Extract title from properties if available
                        page_title = "New Page"
                        properties = data.get('properties', {})
                        for prop_name, prop_data in properties.items():
                            if prop_data.get('type') == 'title':
                                title_texts = prop_data.get('title', [])
                                if title_texts:
                                    page_title = ''.join([t.get('plain_text', '') for t in title_texts])
                                    break
                        
                        if not page_title or page_title.strip() == "":
                            page_title = "New Page"
                        
                        summary_parts.append(f"Notion: Page '{page_title}' created")
                        summary_parts.append(f"   Page ID: {page_id}")
                        if page_url:
                            summary_parts.append(f"   URL: {page_url}")
                        
                        # Show page properties if available
                        if properties:
                            prop_summary = []
                            for prop_name, prop_data in properties.items():
                                if prop_data.get('type') == 'title':
                                    continue  # Already shown as title
                                elif prop_data.get('type') == 'rich_text':
                                    text_content = prop_data.get('rich_text', [])
                                    if text_content:
                                        text_value = ''.join([t.get('plain_text', '') for t in text_content])
                                        if text_value.strip():
                                            prop_summary.append(f"{prop_name}: {text_value[:50]}{'...' if len(text_value) > 50 else ''}")
                                elif prop_data.get('type') == 'number':
                                    number_value = prop_data.get('number')
                                    if number_value is not None:
                                        prop_summary.append(f"{prop_name}: {number_value}")
                                elif prop_data.get('type') == 'select':
                                    select_value = prop_data.get('select')
                                    if select_value:
                                        prop_summary.append(f"{prop_name}: {select_value.get('name', '')}")
                            
                            if prop_summary:
                                summary_parts.append(f"   Properties: {', '.join(prop_summary[:3])}")
                    else:
                        summary_parts.append(f"Notion: Page created successfully")

                elif raw_action == 'query_database':
                    # Handle database query results
                    if isinstance(data, list):
                        summary_parts.append(f"Notion: Found {len(data)} pages in database")
                        if data:
                            df = create_notion_pages_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Notion Pages")
                    elif isinstance(data, dict) and 'results' in data:
                        results = data['results']
                        summary_parts.append(f"Notion: Found {len(results)} pages in database")
                        if results:
                            df = create_notion_pages_dataframe(results)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Notion Pages")
                    else:
                        summary_parts.append(f"Notion: Database query completed")

                elif raw_action == 'get_page':
                    # Handle get page results  
                    if isinstance(data, dict):
                        page_title = "Page"
                        properties = data.get('properties', {})
                        for prop_name, prop_data in properties.items():
                            if prop_data.get('type') == 'title':
                                title_texts = prop_data.get('title', [])
                                if title_texts:
                                    page_title = ''.join([t.get('plain_text', '') for t in title_texts])
                                    break
                        
                        summary_parts.append(f"Notion: Page '{page_title}' details retrieved")
                        
                        # Display as single-row table
                        df = create_notion_pages_dataframe([data])
                        if not df.empty:
                            dataframes.append(df)
                            table_titles.append("Notion Page Details")
                    else:
                        summary_parts.append(f"Notion: Page details retrieved")

                elif raw_action == 'search':
                    # Handle search results
                    if isinstance(data, dict) and 'results' in data:
                        results = data['results']
                        summary_parts.append(f"Notion: Found {len(results)} items in search")
                        if results:
                            df = create_notion_pages_dataframe(results)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Notion Search Results")
                    elif isinstance(data, list):
                        summary_parts.append(f"Notion: Found {len(data)} items in search")
                        if data:
                            df = create_notion_pages_dataframe(data)
                            if not df.empty:
                                dataframes.append(df)
                                table_titles.append("Notion Search Results")
                    else:
                        summary_parts.append(f"Notion: Search completed")

                elif isinstance(data, list):
                    summary_parts.append(f"Notion: Found {len(data)} items")
                    if data:
                        df = create_notion_pages_dataframe(data)
                        if not df.empty:
                            dataframes.append(df)
                            table_titles.append("Notion Pages")
                elif isinstance(data, dict):
                    if 'name' in data and raw_action == 'get_bot_info':
                        summary_parts.append(f"Notion: Bot - {data['name']}")
                    else:
                        summary_parts.append(f"Notion: {action} completed")
                else:
                    summary_parts.append(f"Notion: {action} completed")

        else:
            # FIXED: Enhanced error message handling with better user-friendly messages
            error = result['result'].get('error', 'Unknown error')
            error_str = str(error)

            # Handle specific GitHub API errors more gracefully
            if service.lower() == 'github':
                if '422' in error_str and 'Validation Failed' in error_str:
                    if raw_action == 'create_pull_request':
                        summary_parts.append(
                            f"GitHub: Pull request creation failed - check branch names and permissions")
                    else:
                        summary_parts.append(f"GitHub: Validation error - check required fields")
                elif '404' in error_str:
                    summary_parts.append(f"GitHub: Resource not found - check repository, branch, or file path")
                elif '403' in error_str:
                    summary_parts.append(f"GitHub: Permission denied - check token permissions")
                else:
                    # Extract first line of error for conciseness
                    error_line = error_str.split('\n')[0][:150]
                    summary_parts.append(f"GitHub: {action} failed - {error_line}")

            # Handle Jira errors
            elif service.lower() == 'jira':
                if raw_action == 'create_project' and 'admin privileges' in error_str.lower():
                    summary_parts.append(f"Jira: {error_str}")
                elif 'transition' in error_str.lower():
                    summary_parts.append(f"Jira: Status transition failed - {error_str[:100]}")
                elif '404' in error_str:
                    summary_parts.append(f"Jira: Issue not found - check issue key")
                elif '401' in error_str or '403' in error_str:
                    summary_parts.append(f"Jira: Authentication failed - check credentials")
                else:
                    error_line = error_str.split('\n')[0][:150]
                    summary_parts.append(f"Jira: {action} failed - {error_line}")

            else:
                # Generic error handling
                error_line = error_str.split('\n')[0][:150]
                summary_parts.append(f"{service}: {action} failed - {error_line}")

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
                # Display table with clickable links if URLs exist
                if 'URL' in df.columns and not df['URL'].isna().all():
                    # Create clickable links in the table
                    df_display = df.copy()
                    for idx, row in df_display.iterrows():
                        if pd.notna(row['URL']) and row['URL']:
                            # Handle different column names for GitHub vs Jira
                            if 'Title' in df_display.columns:
                                # GitHub format - make title clickable
                                df_display.at[
                                    idx, 'Title'] = f'<a href="{row["URL"]}" target="_blank">{row["Title"]}</a>'
                            elif 'Summary' in df_display.columns:
                                # Jira format - make summary clickable
                                df_display.at[
                                    idx, 'Summary'] = f'<a href="{row["URL"]}" target="_blank">{row["Summary"]}</a>'
                    # Drop the URL column for display
                    if 'URL' in df_display.columns:
                        df_display = df_display.drop('URL', axis=1)
                    st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
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

    # Connection Status with enhanced Jira details
    st.markdown("**Connection Status**")

    status_github = "Connected" if st.session_state.github_client else "Not Connected"
    status_jira = st.session_state.get('jira_connection_status', 'Not Connected')
    status_notion = "Connected" if st.session_state.notion_client else "Not Connected"
    status_openai = "Connected" if st.session_state.openai_key else "Not Connected"

    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.openai_key else "disconnected"}">OpenAI: {status_openai}</span>',
        unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.github_client else "disconnected"}">GitHub: {status_github}</span>',
        unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-indicator status-{"connected" if status_jira == "Connected" else "disconnected"}">Jira: {status_jira}</span>',
        unsafe_allow_html=True)
    st.markdown(
        f'<span class="status-indicator status-{"connected" if st.session_state.notion_client else "disconnected"}">Notion: {status_notion}</span>',
        unsafe_allow_html=True)

    # FIXED: Add Jira connection debugging info
    if status_jira == 'Failed':
        with st.expander("🔧 Jira Debug Info", expanded=False):
            connection_details = st.session_state.get('jira_connection_details', {})
            if connection_details:
                st.json(connection_details)

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

    # Jira Configuration - FIXED: Enhanced with validation hints
    st.subheader("Jira")
    jira_url = st.text_input(
        "Base URL",
        placeholder="https://yourcompany.atlassian.net",
        key='sidebar_jira_url',
        help="Format: https://yourcompany.atlassian.net (no trailing slash)"
    )
    jira_email = st.text_input(
        "Email",
        key='sidebar_jira_email',
        help="Your Atlassian account email address"
    )
    jira_token = st.text_input(
        "API Token",
        type="password",
        key='sidebar_jira_token',
        help="Generate at: https://id.atlassian.com/manage-profile/security/api-tokens"
    )
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

    # FIXED: Add test connection button for debugging
    if st.button("Test Jira Connection", use_container_width=True):
        if st.session_state.get('jira_client'):
            try:
                health = st.session_state.jira_client.health_check()
                if health.get('healthy'):
                    st.success(f"✅ Jira connection OK: {health.get('user', 'Connected')}")
                else:
                    st.error(f"❌ Jira connection failed: {health.get('error')}")
            except Exception as e:
                st.error(f"❌ Connection test failed: {str(e)}")
        else:
            st.warning("No Jira client available. Connect first.")

# Main Chat Interface
st.subheader("Chat with Assistant")

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

# FIXED: Add footer debug info for troubleshooting
if st.session_state.get('jira_connection_status') == 'Failed':
    st.warning(
        "⚠️ Jira connection failed. Check the sidebar debug info or ensure your URL format is: https://yourcompany.atlassian.net")

if __name__ == "__main__":
    if not IMPORTS_AVAILABLE:
        st.error("Required modules not available. Please install dependencies.")
        st.stop()