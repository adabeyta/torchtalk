#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, Tuple
import tiktoken

@dataclass
class ContextBudget:
    total_tokens: int
    compendium_tokens: int
    compendium_chars: int
    history_tokens: int
    history_chars: int
    message_tokens: int
    message_chars: int
    response_tokens: int
    
    @property
    def total_chars(self) -> int:
        return self.compendium_chars + self.history_chars + self.message_chars

class ContextManager:
    # Predefined context budgets for different scenarios
    CONTEXT_BUDGETS = {
        'dev': ContextBudget(
            total_tokens=32000,      # 32k for development/testing
            compendium_tokens=24000,
            compendium_chars=96000,
            history_tokens=4000,
            history_chars=16000,
            message_tokens=2000,
            message_chars=8000,
            response_tokens=2000
        ),
        'production_128k': ContextBudget(
            total_tokens=128000,     # 128k context models  
            compendium_tokens=96000,
            compendium_chars=384000,
            history_tokens=20000,
            history_chars=80000,
            message_tokens=6000,
            message_chars=24000,
            response_tokens=6000
        ),
        'production_1m': ContextBudget(
            total_tokens=1000000,    # 1M context - your target
            compendium_tokens=800000,
            compendium_chars=3200000,
            history_tokens=150000,
            history_chars=600000,
            message_tokens=25000,
            message_chars=100000,
            response_tokens=25000
        )
    }
    
    def __init__(self, context_profile: str = 'production_1m'):
        if context_profile not in self.CONTEXT_BUDGETS:
            raise ValueError(f"Unknown context profile: {context_profile}")
        
        self.budget = self.CONTEXT_BUDGETS[context_profile]
        self.profile = context_profile
        
        # Try to initialize tiktoken for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except:
            self.tokenizer = None
            self.use_tiktoken = False
            print("Warning: tiktoken not available, using character-based estimation")
    
    def estimate_tokens(self, text: str) -> int:
        if self.use_tiktoken:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: ~4 chars per token for English text
            return len(text) // 4
    
    def truncate_compendium(self, compendium: str) -> str:
        if len(compendium) <= self.budget.compendium_chars:
            return compendium
        
        # Smart truncation
        lines = compendium.split('\n')
        essential_keywords = [
            '# PyTorch Codebase Overview',
            '## Architecture Map', 
            '## Core Layer Stack',
            '## AUTOGRAD Module',
            '## NN Module',
            '## Common PyTorch Patterns'
        ]
        
        # Keep essential sections
        essential_lines = []
        current_section = []
        keep_section = False
        
        for line in lines:
            # Check if this line starts an essential section
            if any(keyword in line for keyword in essential_keywords):
                keep_section = True
                current_section = [line]
            elif line.startswith('#') and current_section:
                # New section started
                if keep_section:
                    essential_lines.extend(current_section)
                current_section = [line]
                keep_section = any(keyword in line for keyword in essential_keywords)
            else:
                current_section.append(line)
        
        # Add last section if essential
        if keep_section and current_section:
            essential_lines.extend(current_section)
        
        truncated = '\n'.join(essential_lines)
        
        # If still too long, truncate more aggressively
        if len(truncated) > self.budget.compendium_chars:
            truncated = truncated[:self.budget.compendium_chars - 100]
            truncated += "\n\n[TRUNCATED - Content exceeds context budget]"
        
        return truncated
    
    def truncate_history(self, history: list) -> list:
        if not history:
            return []
        
        # Calculate total length of history
        total_chars = sum(len(msg.get('content', '')) for msg in history)
        
        if total_chars <= self.budget.history_chars:
            return history
        
        # Keep recent messages, truncate older ones
        truncated_history = []
        current_chars = 0
        
        # Start from most recent messages
        for msg in reversed(history):
            msg_chars = len(msg.get('content', ''))
            if current_chars + msg_chars <= self.budget.history_chars:
                truncated_history.append(msg)
                current_chars += msg_chars
            else:
                break
        
        return list(reversed(truncated_history))
    
    def validate_message(self, message: str) -> str:
        if len(message) <= self.budget.message_chars:
            return message
        
        # Truncate long messages
        truncated = message[:self.budget.message_chars - 50]
        truncated += "\n[MESSAGE TRUNCATED]"
        return truncated
    
    def get_context_info(self) -> Dict:
        return {
            'profile': self.profile,
            'total_tokens': self.budget.total_tokens,
            'compendium_limit_chars': self.budget.compendium_chars,
            'history_limit_chars': self.budget.history_chars,
            'message_limit_chars': self.budget.message_chars,
            'use_tiktoken': self.use_tiktoken
        }

