#!/usr/bin/env python3
"""
Intelligent Tool Base Class - Enhances all tools with LLM capabilities.
This class provides intelligent parameter inference, context understanding, and adaptive behavior.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from openai import OpenAI
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass 
class IntelligentToolConfig:
    """Configuration for intelligent tool behavior."""
    openai_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1000
    enable_parameter_inference: bool = True
    enable_context_awareness: bool = True
    enable_adaptive_responses: bool = True
    cache_responses: bool = True

class IntelligentToolBase(ABC):
    """
    Base class that adds LLM intelligence to any tool.
    
    Features:
    - Intelligent parameter inference
    - Context-aware task understanding  
    - Adaptive behavior based on user intent
    - Smart error handling and suggestions
    - Learning from user patterns
    """
    
    def __init__(self, config: IntelligentToolConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_key)
        self._context_cache = {}
        self._pattern_cache = {}
    
    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool (e.g., 'GitHub', 'Jira', 'Notion')."""
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """Return list of available actions for this tool."""
        pass
    
    @abstractmethod
    def execute_base_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the base/raw action without intelligence enhancement."""
        pass
    
    def intelligent_execute(
        self, 
        user_request: str, 
        suggested_action: str = None,
        suggested_params: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute an action with full LLM intelligence enhancement.
        
        Args:
            user_request: Original user request in natural language
            suggested_action: Pre-suggested action (optional)
            suggested_params: Pre-suggested parameters (optional) 
            context: Additional context information
            
        Returns:
            Enhanced execution result with intelligent feedback
        """
        
        try:
            # Step 1: Understand user intent and enhance parameters
            enhanced_plan = self._understand_and_plan(
                user_request, suggested_action, suggested_params, context
            )
            
            # Step 2: Execute the action with enhanced parameters
            base_result = self.execute_base_action(
                enhanced_plan['action'], 
                enhanced_plan['parameters']
            )
            
            # Step 3: Intelligently interpret and enhance the response  
            intelligent_response = self._enhance_response(
                base_result, user_request, enhanced_plan, context
            )
            
            # Step 4: Learn from this interaction for future improvement
            self._learn_from_interaction(user_request, enhanced_plan, intelligent_response)
            
            return intelligent_response
            
        except Exception as e:
            logger.error(f"Intelligent execution failed: {e}")
            # Fallback to base execution
            if suggested_action and suggested_params:
                return self.execute_base_action(suggested_action, suggested_params)
            else:
                return {'success': False, 'error': f'Intelligent execution failed: {str(e)}'}
    
    def _understand_and_plan(
        self, 
        user_request: str,
        suggested_action: str = None,
        suggested_params: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Use LLM to understand user intent and create optimal execution plan."""
        
        if not self.config.enable_parameter_inference:
            return {
                'action': suggested_action or 'unknown',
                'parameters': suggested_params or {},
                'confidence': 0.5
            }
        
        # Build intelligent planning prompt
        available_actions = self.get_available_actions()
        tool_name = self.get_tool_name()
        
        planning_prompt = f"""
You are an intelligent {tool_name} tool assistant. Analyze the user's request and create the optimal execution plan.

USER REQUEST: "{user_request}"

SUGGESTED ACTION: {suggested_action or "None"}
SUGGESTED PARAMETERS: {json.dumps(suggested_params or {}, indent=2)}

AVAILABLE ACTIONS: {json.dumps(available_actions, indent=2)}

CONTEXT: {json.dumps(context or {}, indent=2)}

TASK: Create the optimal execution plan. Consider:
1. What is the user really trying to accomplish?
2. Are there better actions to achieve their goal?
3. What parameters can be inferred or improved?
4. Are there missing parameters that should be prompted for?
5. What context clues can improve the execution?

RESPONSE FORMAT (JSON only):
{{
    "action": "optimal_action_name", 
    "parameters": {{"param1": "enhanced_value1", "param2": "inferred_value2"}},
    "confidence": 0.95,
    "reasoning": "Why this plan is optimal",
    "suggestions": ["Optional suggestions for the user"],
    "missing_info": ["Any critical missing information"],
    "alternatives": ["Alternative approaches if current plan fails"]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": planning_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Parse LLM response
            plan_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            plan = self._extract_json_from_response(plan_text)
            
            if not plan or 'action' not in plan:
                # Fallback to suggested action
                return {
                    'action': suggested_action or 'unknown',
                    'parameters': suggested_params or {},
                    'confidence': 0.3,
                    'reasoning': 'LLM planning failed, using fallback'
                }
            
            return plan
            
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}")
            return {
                'action': suggested_action or 'unknown', 
                'parameters': suggested_params or {},
                'confidence': 0.3,
                'reasoning': f'LLM planning error: {str(e)}'
            }
    
    def _enhance_response(
        self, 
        base_result: Dict[str, Any], 
        user_request: str,
        execution_plan: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Use LLM to enhance and interpret the raw tool response."""
        
        if not self.config.enable_adaptive_responses:
            return base_result
        
        tool_name = self.get_tool_name()
        
        enhancement_prompt = f"""
You are an intelligent {tool_name} response interpreter. Enhance the raw tool response to be more helpful and user-friendly.

USER REQUEST: "{user_request}"
EXECUTION PLAN: {json.dumps(execution_plan, indent=2)}
RAW RESULT: {json.dumps(base_result, indent=2)}
CONTEXT: {json.dumps(context or {}, indent=2)}

TASK: Enhance the response by:
1. Interpreting what actually happened
2. Explaining the result in user-friendly terms
3. Suggesting next steps if appropriate
4. Identifying any issues or improvements
5. Providing actionable insights

RESPONSE FORMAT (JSON only):
{{
    "success": true/false,
    "data": <original or enhanced data>,
    "user_message": "Clear explanation of what happened",
    "insights": ["Key insights from the result"],
    "next_steps": ["Suggested next actions"],
    "warnings": ["Any warnings or issues to note"],
    "enhanced_data": <any additional computed data>,
    "metadata": {{
        "execution_time": "estimate",
        "confidence": 0.95,
        "data_quality": "high/medium/low"
    }}
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": enhancement_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            enhanced_text = response.choices[0].message.content.strip()
            enhanced_result = self._extract_json_from_response(enhanced_text)
            
            if enhanced_result:
                # Merge with original result
                enhanced_result['raw_result'] = base_result
                enhanced_result['enhanced_by_llm'] = True
                return enhanced_result
            
        except Exception as e:
            logger.warning(f"Response enhancement failed: {e}")
        
        # Fallback to original result with basic enhancement
        base_result['enhanced_by_llm'] = False
        base_result['user_message'] = f"{tool_name} action completed"
        return base_result
    
    def _learn_from_interaction(
        self,
        user_request: str,
        execution_plan: Dict[str, Any], 
        result: Dict[str, Any]
    ):
        """Learn from this interaction to improve future performance."""
        
        # Store interaction patterns for future reference
        pattern_key = self._generate_pattern_key(user_request)
        
        if pattern_key not in self._pattern_cache:
            self._pattern_cache[pattern_key] = []
        
        self._pattern_cache[pattern_key].append({
            'request': user_request,
            'plan': execution_plan,
            'success': result.get('success', False),
            'timestamp': __import__('time').time()
        })
        
        # Keep only recent patterns (last 100 per pattern type)
        if len(self._pattern_cache[pattern_key]) > 100:
            self._pattern_cache[pattern_key] = self._pattern_cache[pattern_key][-100:]
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with multiple fallback strategies."""
        
        if not response_text:
            return {}
        
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(response_text)
        except:
            pass
        
        # Strategy 2: Extract from code blocks
        json_patterns = [
            r'```(?:json)?\s*(\{.*?\})\s*```',
            r'(\{(?:[^{}]|{[^{}]*})*\})',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        return {}
    
    def _generate_pattern_key(self, user_request: str) -> str:
        """Generate a pattern key for learning purposes.""" 
        
        # Simple pattern extraction based on keywords
        request_lower = user_request.lower()
        
        if any(word in request_lower for word in ['create', 'new', 'add']):
            return 'create_pattern'
        elif any(word in request_lower for word in ['update', 'edit', 'change']):
            return 'update_pattern'  
        elif any(word in request_lower for word in ['get', 'show', 'list', 'find']):
            return 'read_pattern'
        elif any(word in request_lower for word in ['delete', 'remove']):
            return 'delete_pattern'
        else:
            return 'general_pattern'

    def get_learned_patterns(self) -> Dict[str, Any]:
        """Get summary of learned patterns for debugging/analysis."""
        
        summary = {}
        for pattern_key, interactions in self._pattern_cache.items():
            if interactions:
                success_rate = sum(1 for i in interactions if i['success']) / len(interactions)
                summary[pattern_key] = {
                    'total_interactions': len(interactions),
                    'success_rate': success_rate,
                    'recent_interactions': len([i for i in interactions if __import__('time').time() - i['timestamp'] < 3600])  # Last hour
                }
        
        return summary