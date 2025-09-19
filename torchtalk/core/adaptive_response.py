#!/usr/bin/env python3
from typing import Dict, Tuple
from torchtalk.context.assembler import DynamicContextAssembler
from enum import Enum

class ResponseStyle(Enum):
    PYTORCH_EXPERT = "pytorch_expert"           # High relevance - detailed technical
    PYTORCH_AWARE = "pytorch_aware"             # Medium relevance - general with PyTorch context
    PROGRAMMING_HELPER = "programming_helper"   # Low relevance - general programming
    CASUAL_ASSISTANT = "casual_assistant"       # Very low relevance - conversational

class AdaptiveResponseManager:
    def __init__(self, context_assembler: DynamicContextAssembler):
        self.context_assembler = context_assembler

        # Relevance score thresholds for different response styles
        # Adjusted based on enhanced graph-based retrieval testing
        self.style_thresholds = {
            ResponseStyle.PYTORCH_EXPERT: 8.0,       # High confidence PyTorch questions
            ResponseStyle.PYTORCH_AWARE: 4.0,        # Medium confidence ML/DL questions
            ResponseStyle.PROGRAMMING_HELPER: 2.0,   # General programming
            ResponseStyle.CASUAL_ASSISTANT: 0.0      # Fallback for everything else
        }

    def analyze_question(self, question: str, max_context_chars: int = 96000) -> Dict:
        # Quick check for obvious casual conversation
        if self._is_obvious_casual(question):
            return {
                "context": "",
                "system_prompt": self._get_system_prompt(ResponseStyle.CASUAL_ASSISTANT, {}),
                "response_style": ResponseStyle.CASUAL_ASSISTANT.value,
                "relevance_metrics": {"max_relevance": 0, "reason": "casual_conversation"},
                "context_info": {"casual_detected": True},
                "max_tokens": self._get_max_tokens(ResponseStyle.CASUAL_ASSISTANT, question),
                "temperature": self._get_temperature(ResponseStyle.CASUAL_ASSISTANT)
            }

        # Check for general programming questions (not PyTorch-specific)
        if self._is_general_programming(question):
            # Still run context assembly but cap the response style
            try:
                context_info = self.context_assembler.get_context_info(question)
                context = self.context_assembler.assemble_context(question, max_context_chars)
                relevant_items = context_info.get('relevant_items', [])

                # Calculate metrics
                max_relevance = max((score for _, score in relevant_items), default=0.0) if relevant_items else 0.0

                # Cap at programming_helper even if high relevance
                return {
                    "context": context[:max_context_chars // 2],  # Use less context
                    "system_prompt": self._get_system_prompt(ResponseStyle.PROGRAMMING_HELPER, context_info),
                    "response_style": ResponseStyle.PROGRAMMING_HELPER.value,
                    "relevance_metrics": {"max_relevance": max_relevance, "capped": "general_programming"},
                    "context_info": context_info,
                    "max_tokens": self._get_max_tokens(ResponseStyle.PROGRAMMING_HELPER, question),
                    "temperature": self._get_temperature(ResponseStyle.PROGRAMMING_HELPER)
                }
            except Exception as e:
                pass  # Fall through to normal processing

        # Always run intelligent context assembly for non-casual questions
        try:
            context_info = self.context_assembler.get_context_info(question)
            context = self.context_assembler.assemble_context(question, max_context_chars)

            # Calculate relevance metrics
            relevant_items = context_info.get('relevant_items', [])

            if relevant_items:
                # Format should be: (item_type, score)
                max_relevance = max((score for _, score in relevant_items), default=0.0)
                total_relevance = sum(score for _, score in relevant_items)
                avg_relevance = total_relevance / len(relevant_items)
            else:
                max_relevance = 0.0
                total_relevance = 0.0
                avg_relevance = 0.0

            # Determine response style based on relevance
            response_style = self._determine_response_style(max_relevance, avg_relevance, relevant_items)

            # Get appropriate prompts for this style
            system_prompt = self._get_system_prompt(response_style, context_info)
            context_to_use = self._filter_context_by_style(context, response_style, max_relevance)

            return {
                "context": context_to_use,
                "system_prompt": system_prompt,
                "response_style": response_style.value,
                "relevance_metrics": {
                    "max_relevance": max_relevance,
                    "avg_relevance": avg_relevance,
                    "total_relevance": total_relevance,
                    "relevant_items_count": len(relevant_items)
                },
                "context_info": context_info,
                "max_tokens": self._get_max_tokens(response_style, question),
                "temperature": self._get_temperature(response_style)
            }

        except Exception as e:
            # Fallback to casual assistant if context assembly fails
            import traceback
            print(f"ERROR in analyze_question for '{question[:50]}': {e}")
            traceback.print_exc()
            return {
                "context": "",
                "system_prompt": self._get_system_prompt(ResponseStyle.CASUAL_ASSISTANT, {}),
                "response_style": ResponseStyle.CASUAL_ASSISTANT.value,
                "relevance_metrics": {"error": str(e)},
                "context_info": {"error": str(e)},
                "max_tokens": 100,
                "temperature": 0.7
            }

    def _determine_response_style(self, max_relevance: float, avg_relevance: float,
                                relevant_items: list) -> ResponseStyle:
        # Use max relevance as primary indicator
        if max_relevance >= self.style_thresholds[ResponseStyle.PYTORCH_EXPERT]:
            return ResponseStyle.PYTORCH_EXPERT
        elif max_relevance >= self.style_thresholds[ResponseStyle.PYTORCH_AWARE]:
            return ResponseStyle.PYTORCH_AWARE
        elif max_relevance >= self.style_thresholds[ResponseStyle.PROGRAMMING_HELPER]:
            return ResponseStyle.PROGRAMMING_HELPER
        else:
            return ResponseStyle.CASUAL_ASSISTANT

    def _get_system_prompt(self, style: ResponseStyle, context_info: Dict) -> str:

        prompts = {
            ResponseStyle.PYTORCH_EXPERT:
                "You are a PyTorch expert with deep knowledge of the codebase. Provide detailed, "
                "technical answers using the provided PyTorch documentation and code analysis. "
                "Include specific examples, code snippets, and references to relevant modules when helpful. "
                "Be precise and comprehensive in your explanations.",

            ResponseStyle.PYTORCH_AWARE:
                "You are a PyTorch expert with deep knowledge of machine learning. "
                "Answer questions clearly using the provided context when relevant. "
                "If the question relates to PyTorch, provide accurate information with examples. "
                "For general questions, respond naturally without forcing PyTorch references.",

            ResponseStyle.PROGRAMMING_HELPER:
                "You are a programming expert. Answer questions clearly and concisely. "
                "Use the provided context if it's relevant to the programming question. "
                "Focus on practical, actionable advice.",

            ResponseStyle.CASUAL_ASSISTANT:
                "You are a knowledgeable expert. Respond naturally and conversationally. "
                "Be friendly and helpful while staying focused on the user's question."
        }

        return prompts[style]

    def _filter_context_by_style(self, context: str, style: ResponseStyle, relevance: float) -> str:
        if style == ResponseStyle.CASUAL_ASSISTANT:
            # For casual conversation, provide minimal context
            return ""
        elif style == ResponseStyle.PROGRAMMING_HELPER and relevance < 5.0:
            # For low-relevance programming questions, provide summary only
            lines = context.split('\n')
            # Keep just the overview section
            overview_end = next((i for i, line in enumerate(lines)
                               if line.startswith('## ') and 'Module:' in line), 20)
            return '\n'.join(lines[:overview_end])
        else:
            # For PyTorch questions, provide full context
            return context

    def _get_max_tokens(self, style: ResponseStyle, question: str) -> int:
        question_length = len(question.split())

        if style == ResponseStyle.PYTORCH_EXPERT:
            # Detailed technical responses
            return min(400, 150 + question_length * 10)
        elif style == ResponseStyle.PYTORCH_AWARE:
            # Medium-length responses
            return min(250, 100 + question_length * 8)
        elif style == ResponseStyle.PROGRAMMING_HELPER:
            # Concise programming help
            return min(200, 80 + question_length * 6)
        else:  # CASUAL_ASSISTANT
            # Short conversational responses
            return min(100, 30 + question_length * 3)

    def _get_temperature(self, style: ResponseStyle) -> float:
        temperatures = {
            ResponseStyle.PYTORCH_EXPERT: 0.1,      # Very focused and precise
            ResponseStyle.PYTORCH_AWARE: 0.2,       # Mostly focused
            ResponseStyle.PROGRAMMING_HELPER: 0.3,  # Balanced
            ResponseStyle.CASUAL_ASSISTANT: 0.6     # More creative and conversational
        }

        return temperatures[style]

    def _is_obvious_casual(self, question: str) -> bool:
        question_lower = question.lower().strip()

        # Empty or too short
        if not question_lower or len(question_lower) <= 1:
            return True

        # Very short greetings
        if len(question_lower.split()) <= 2:
            casual_words = {'hi', 'hello', 'hey', 'thanks', 'thank', 'bye', 'goodbye', 'ok', 'okay'}
            if any(word in question_lower for word in casual_words):
                return True

        # Obvious casual patterns
        casual_patterns = [
            'how are you',
            'how\'s it going',
            'what\'s up',
            'tell me about yourself',
            'tell me a joke',
            'what\'s the weather',
            'how\'s the weather',
            'good morning',
            'good afternoon',
            'good evening',
            'thank you'
        ]

        return any(pattern in question_lower for pattern in casual_patterns)

    def _is_general_programming(self, question: str) -> bool:
        question_lower = question.lower().strip()

        # Explicitly PyTorch/ML terms mean it's NOT general programming
        pytorch_terms = ['torch', 'pytorch', 'tensor', 'autograd', 'nn.', 'neural', 'deep learning']
        if any(term in question_lower for term in pytorch_terms):
            return False

        # General programming indicators
        general_patterns = [
            'optimize.*code',
            'debug.*python',
            'python.*performance',
            'memory.*issue',
            'code.*optimization',
            'improve.*performance',
            'speed up.*code'
        ]

        import re
        return any(re.search(pattern, question_lower) for pattern in general_patterns)

    def get_debug_info(self, question: str) -> Dict:
        analysis = self.analyze_question(question)

        return {
            "question": question,
            "response_style": analysis["response_style"],
            "relevance_metrics": analysis["relevance_metrics"],
            "system_prompt": analysis["system_prompt"][:100] + "...",
            "context_length": len(analysis["context"]),
            "max_tokens": analysis["max_tokens"],
            "temperature": analysis["temperature"],
            "style_thresholds": {style.value: threshold for style, threshold in self.style_thresholds.items()},
            "relevant_items_preview": analysis["context_info"].get("relevant_items", [])[:3]
        }