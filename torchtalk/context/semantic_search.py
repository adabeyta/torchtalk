#!/usr/bin/env python3
import re
import ast
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
from pathlib import Path
import math


@dataclass
class SemanticMatch:
    content_type: str
    content_data: Dict[str, Any]
    score: float
    match_reasons: List[str]
    context_snippet: Optional[str] = None


class SemanticContentFinder:
    def __init__(self, enhanced_analysis_path: str):
        self.analysis_path = enhanced_analysis_path
        self.analysis_data = self._load_analysis()

        # Build various indices for search
        self.code_patterns = self._build_code_patterns()
        self.semantic_signatures = self._build_semantic_signatures()
        self.docstring_index = self._build_docstring_index()
        self.ast_pattern_index = self._build_ast_pattern_index()

    def _load_analysis(self) -> Dict[str, Any]:
        with open(self.analysis_path, 'r') as f:
            return json.load(f)

    def _build_code_patterns(self) -> Dict[str, List[str]]:
        patterns = defaultdict(list)

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_data in modules.items():
            # Extract patterns from functions
            for func_name, func_data in module_data.get('functions', {}).items():
                if func_data.get('source_code') or func_data.get('source_preview'):
                    code = func_data.get('source_code') or func_data.get('source_preview', '')
                    extracted = self._extract_code_patterns(code)
                    for pattern_type, pattern_list in extracted.items():
                        patterns[pattern_type].extend(pattern_list)

            # Extract patterns from classes
            for class_name, class_data in module_data.get('classes', {}).items():
                if class_data.get('source_code') or class_data.get('source_preview'):
                    code = class_data.get('source_code') or class_data.get('source_preview', '')
                    extracted = self._extract_code_patterns(code)
                    for pattern_type, pattern_list in extracted.items():
                        patterns[pattern_type].extend(pattern_list)

        # Deduplicate patterns
        return {k: list(set(v)) for k, v in patterns.items()}

    def _extract_code_patterns(self, code: str) -> Dict[str, List[str]]:
        patterns = {
            'api_calls': [],
            'data_structures': [],
            'control_flow': [],
            'tensor_ops': [],
            'nn_patterns': []
        }

        if not code:
            return patterns

        # API calls (method calls with common patterns)
        api_calls = re.findall(r'\b(\w+)\s*\.\s*(\w+)\s*\(', code)
        patterns['api_calls'] = [f"{obj}.{method}" for obj, method in api_calls]

        # Data structures
        patterns['data_structures'].extend(re.findall(r'\b(list|dict|set|tuple|tensor|array)\b', code.lower()))

        # Control flow patterns
        if 'for ' in code:
            patterns['control_flow'].append('iteration')
        if 'while ' in code:
            patterns['control_flow'].append('while_loop')
        if 'if ' in code:
            patterns['control_flow'].append('conditional')
        if 'try:' in code:
            patterns['control_flow'].append('exception_handling')

        # PyTorch-specific patterns
        tensor_ops = re.findall(r'torch\.\w+|\.view|\.reshape|\.squeeze|\.unsqueeze|\.permute|\.transpose', code)
        patterns['tensor_ops'] = tensor_ops

        # Neural network patterns
        nn_patterns = re.findall(r'nn\.\w+|forward|backward|loss|optimizer|grad', code.lower())
        patterns['nn_patterns'] = nn_patterns

        return patterns

    def _build_semantic_signatures(self) -> Dict[str, Dict[str, float]]:
        signatures = {}

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_data in modules.items():
            module_name = module_data['name']

            # Build signatures for functions
            for func_name, func_data in module_data.get('functions', {}).items():
                signature_key = f"{module_name}.{func_name}"
                signatures[signature_key] = self._compute_semantic_signature(func_data)

            # Build signatures for classes
            for class_name, class_data in module_data.get('classes', {}).items():
                signature_key = f"{module_name}.{class_name}"
                signatures[signature_key] = self._compute_semantic_signature(class_data)

        return signatures

    def _compute_semantic_signature(self, entity_data: Dict[str, Any]) -> Dict[str, float]:
        signature = defaultdict(float)

        # Weight different aspects
        if entity_data.get('docstring'):
            doc_terms = self._extract_technical_terms(entity_data['docstring'])
            for term in doc_terms:
                signature[term] += 1.0

        if entity_data.get('signature'):
            sig_terms = self._extract_technical_terms(entity_data['signature'])
            for term in sig_terms:
                signature[term] += 0.5

        if entity_data.get('calls'):
            for call in entity_data['calls']:
                signature[f"calls_{call}"] += 0.3

        if entity_data.get('references'):
            for ref in entity_data['references']:
                signature[f"ref_{ref}"] += 0.4

        # Normalize
        total = sum(signature.values())
        if total > 0:
            signature = {k: v/total for k, v in signature.items()}

        return dict(signature)

    def _extract_technical_terms(self, text: str) -> List[str]:
        if not text:
            return []

        text_lower = text.lower()
        # Extract technical words
        terms = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', text_lower)

        # Filter common words
        common = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'can', 'will', 'not', 'get', 'set'}
        return [t for t in terms if t not in common and len(t) > 2]

    def _build_docstring_index(self) -> Dict[str, List[Tuple[str, str, str]]]:
        index = defaultdict(list)

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_data in modules.items():
            module_name = module_data['name']

            # Index function docstrings
            for func_name, func_data in module_data.get('functions', {}).items():
                if func_data.get('docstring'):
                    terms = self._extract_technical_terms(func_data['docstring'])
                    for term in terms:
                        index[term].append((module_path, 'function', func_name))

            # Index class docstrings
            for class_name, class_data in module_data.get('classes', {}).items():
                if class_data.get('docstring'):
                    terms = self._extract_technical_terms(class_data['docstring'])
                    for term in terms:
                        index[term].append((module_path, 'class', class_name))

        return dict(index)

    def _build_ast_pattern_index(self) -> Dict[str, List[str]]:
        patterns = defaultdict(list)

        modules = self.analysis_data.get('modules_with_code', {})
        for module_path, module_data in modules.items():
            # Analyze function code
            for func_name, func_data in module_data.get('functions', {}).items():
                code = func_data.get('source_code', '')
                if code:
                    ast_patterns = self._extract_ast_patterns(code)
                    for pattern in ast_patterns:
                        patterns[pattern].append(f"{module_data['path']}::{func_name}")

            # Analyze class code
            for class_name, class_data in module_data.get('classes', {}).items():
                code = class_data.get('source_code', '')
                if code:
                    ast_patterns = self._extract_ast_patterns(code)
                    for pattern in ast_patterns:
                        patterns[pattern].append(f"{module_data['path']}::{class_name}")

        return dict(patterns)

    def _extract_ast_patterns(self, code: str) -> List[str]:
        patterns = []

        try:
            tree = ast.parse(code)

            # Walk the AST and extract patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    patterns.append('for_loop')
                elif isinstance(node, ast.While):
                    patterns.append('while_loop')
                elif isinstance(node, ast.If):
                    patterns.append('conditional')
                elif isinstance(node, ast.Try):
                    patterns.append('try_except')
                elif isinstance(node, ast.With):
                    patterns.append('context_manager')
                elif isinstance(node, ast.Lambda):
                    patterns.append('lambda')
                elif isinstance(node, ast.ListComp):
                    patterns.append('list_comprehension')
                elif isinstance(node, ast.DictComp):
                    patterns.append('dict_comprehension')
                elif isinstance(node, ast.GeneratorExp):
                    patterns.append('generator_expression')
                elif isinstance(node, ast.Yield):
                    patterns.append('generator')
                elif isinstance(node, ast.AsyncFunctionDef):
                    patterns.append('async_function')
                elif isinstance(node, ast.ClassDef):
                    if any(base for base in node.bases if isinstance(base, ast.Name) and 'Module' in base.id):
                        patterns.append('nn_module')
        except:
            pass

        return patterns

    def find_semantic_matches(self, query: str, max_results: int = 20) -> List[SemanticMatch]:
        matches = []

        # Query understanding
        query_intent = self._understand_query_intent(query)
        query_terms = self._extract_technical_terms(query)

        # Direct keyword matching
        keyword_matches = self._find_keyword_matches(query_terms)
        matches.extend(keyword_matches)

        # Pattern-based matching
        pattern_matches = self._find_pattern_matches(query_intent)
        matches.extend(pattern_matches)

        # Semantic similarity matching
        similarity_matches = self._find_similarity_matches(query_terms, query_intent)
        matches.extend(similarity_matches)

        # Code pattern matching
        if query_intent.get('code_patterns'):
            code_matches = self._find_code_pattern_matches(query_intent['code_patterns'])
            matches.extend(code_matches)

        # Deduplicate and rank
        unique_matches = self._deduplicate_matches(matches)
        ranked = self._rank_matches(unique_matches, query_intent)

        return ranked[:max_results]

    def _understand_query_intent(self, query: str) -> Dict[str, Any]:
        intent = {
            'action': None,
            'target': None,
            'context': [],
            'code_patterns': []
        }

        query_lower = query.lower()

        # Detect action intent
        if any(word in query_lower for word in ['how', 'implement', 'create', 'build', 'make']):
            intent['action'] = 'implementation'
        elif any(word in query_lower for word in ['what', 'explain', 'describe']):
            intent['action'] = 'explanation'
        elif any(word in query_lower for word in ['why', 'reason']):
            intent['action'] = 'reasoning'
        elif any(word in query_lower for word in ['when', 'use']):
            intent['action'] = 'usage'
        elif any(word in query_lower for word in ['where', 'find', 'locate']):
            intent['action'] = 'location'

        # Detect target
        if 'class' in query_lower:
            intent['target'] = 'class'
        elif 'function' in query_lower or 'method' in query_lower:
            intent['target'] = 'function'
        elif 'module' in query_lower:
            intent['target'] = 'module'

        # Detect context
        if 'gradient' in query_lower or 'autograd' in query_lower:
            intent['context'].append('autograd')
        if 'tensor' in query_lower:
            intent['context'].append('tensor')
        if 'layer' in query_lower or 'nn' in query_lower or 'neural' in query_lower:
            intent['context'].append('neural_network')
        if 'optimize' in query_lower or 'optimizer' in query_lower:
            intent['context'].append('optimization')
        if 'loss' in query_lower:
            intent['context'].append('loss')
        if 'train' in query_lower:
            intent['context'].append('training')

        # Detect code patterns mentioned
        if 'loop' in query_lower or 'iterate' in query_lower:
            intent['code_patterns'].append('for_loop')
        if 'inherit' in query_lower or 'derive' in query_lower:
            intent['code_patterns'].append('inheritance')
        if 'override' in query_lower:
            intent['code_patterns'].append('override')

        return intent

    def _find_keyword_matches(self, query_terms: List[str]) -> List[SemanticMatch]:
        matches = []

        for term in query_terms:
            if term in self.docstring_index:
                for module_path, entity_type, entity_name in self.docstring_index[term]:
                    # Get the entity data
                    entity_data = self._get_entity_data(module_path, entity_type, entity_name)
                    if entity_data:
                        matches.append(SemanticMatch(
                            content_type=entity_type,
                            content_data=entity_data,
                            score=1.0,
                            match_reasons=[f"Keyword match: {term}"],
                            context_snippet=entity_data.get('docstring', '')[:200]
                        ))

        return matches

    def _find_pattern_matches(self, query_intent: Dict[str, Any]) -> List[SemanticMatch]:
        matches = []

        # Match based on target type
        if query_intent['target'] == 'class':
            # Find relevant classes
            modules = self.analysis_data.get('modules_with_code', {})
            for module_path, module_data in modules.items():
                for class_name, class_data in module_data.get('classes', {}).items():
                    # Check if class matches context
                    if self._matches_context(class_data, query_intent['context']):
                        matches.append(SemanticMatch(
                            content_type='class',
                            content_data=class_data,
                            score=0.8,
                            match_reasons=[f"Class matching context: {', '.join(query_intent['context'])}"],
                            context_snippet=class_data.get('signature', '')
                        ))

        return matches

    def _find_similarity_matches(self, query_terms: List[str], query_intent: Dict[str, Any]) -> List[SemanticMatch]:
        matches = []

        # Create query signature
        query_signature = {term: 1.0/len(query_terms) for term in query_terms if query_terms}

        # Compare with entity signatures
        for entity_key, entity_signature in self.semantic_signatures.items():
            similarity = self._compute_cosine_similarity(query_signature, entity_signature)
            if similarity > 0.3:  # Threshold
                # Get entity data
                parts = entity_key.split('.')
                if len(parts) >= 2:
                    module_name = parts[0]
                    entity_name = '.'.join(parts[1:])

                    # Find the entity
                    entity_data = self._find_entity_by_name(module_name, entity_name)
                    if entity_data:
                        matches.append(SemanticMatch(
                            content_type=entity_data['type'],
                            content_data=entity_data['data'],
                            score=similarity,
                            match_reasons=[f"Semantic similarity: {similarity:.2f}"],
                            context_snippet=None
                        ))

        return matches

    def _find_code_pattern_matches(self, patterns: List[str]) -> List[SemanticMatch]:
        matches = []

        for pattern in patterns:
            if pattern in self.ast_pattern_index:
                for entity_id in self.ast_pattern_index[pattern]:
                    # Parse entity ID
                    parts = entity_id.split('::')
                    if len(parts) == 2:
                        module_path, entity_name = parts

                        # Find the entity
                        entity_data = self._get_entity_by_path(module_path, entity_name)
                        if entity_data:
                            matches.append(SemanticMatch(
                                content_type=entity_data['type'],
                                content_data=entity_data['data'],
                                score=0.7,
                                match_reasons=[f"Code pattern: {pattern}"],
                                context_snippet=None
                            ))

        return matches

    def _matches_context(self, entity_data: Dict[str, Any], contexts: List[str]) -> bool:
        if not contexts:
            return True

        entity_text = ' '.join([
            entity_data.get('docstring', ''),
            entity_data.get('signature', ''),
            ' '.join(entity_data.get('calls', [])),
            ' '.join(entity_data.get('references', []))
        ]).lower()

        for context in contexts:
            if context in entity_text:
                return True

        return False

    def _compute_cosine_similarity(self, sig1: Dict[str, float], sig2: Dict[str, float]) -> float:
        if not sig1 or not sig2:
            return 0.0

        # Get common keys
        common_keys = set(sig1.keys()) & set(sig2.keys())
        if not common_keys:
            return 0.0

        # Compute dot product
        dot_product = sum(sig1[key] * sig2[key] for key in common_keys)

        # Compute magnitudes
        mag1 = math.sqrt(sum(v*v for v in sig1.values()))
        mag2 = math.sqrt(sum(v*v for v in sig2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _deduplicate_matches(self, matches: List[SemanticMatch]) -> List[SemanticMatch]:
        seen = set()
        unique = []

        for match in matches:
            # Create a unique key for the match
            key = f"{match.content_type}_{id(match.content_data)}"
            if key not in seen:
                seen.add(key)
                unique.append(match)
            else:
                # Merge match reasons if duplicate
                for existing in unique:
                    if f"{existing.content_type}_{id(existing.content_data)}" == key:
                        existing.match_reasons.extend(match.match_reasons)
                        existing.score = max(existing.score, match.score)
                        break

        return unique

    def _rank_matches(self, matches: List[SemanticMatch], query_intent: Dict[str, Any]) -> List[SemanticMatch]:
        for match in matches:
            # Boost score based on intent alignment
            if query_intent['action'] == 'implementation' and match.content_type == 'function':
                match.score *= 1.2
            elif query_intent['action'] == 'explanation' and match.content_data.get('docstring'):
                match.score *= 1.1

            # Boost if matches target type
            if query_intent['target'] == match.content_type:
                match.score *= 1.3

            # Boost for context matches
            for context in query_intent['context']:
                if context in str(match.content_data).lower():
                    match.score *= 1.1

        # Sort by score
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches

    def _get_entity_data(self, module_path: str, entity_type: str, entity_name: str) -> Optional[Dict[str, Any]]:
        modules = self.analysis_data.get('modules_with_code', {})

        for path, module_data in modules.items():
            if module_data.get('path') == module_path or path == module_path:
                if entity_type == 'function' and entity_name in module_data.get('functions', {}):
                    return module_data['functions'][entity_name]
                elif entity_type == 'class' and entity_name in module_data.get('classes', {}):
                    return module_data['classes'][entity_name]

        return None

    def _find_entity_by_name(self, module_name: str, entity_name: str) -> Optional[Dict[str, Any]]:
        modules = self.analysis_data.get('modules_with_code', {})

        for path, module_data in modules.items():
            if module_data.get('name') == module_name:
                # Check functions
                if entity_name in module_data.get('functions', {}):
                    return {
                        'type': 'function',
                        'data': module_data['functions'][entity_name]
                    }
                # Check classes
                elif entity_name in module_data.get('classes', {}):
                    return {
                        'type': 'class',
                        'data': module_data['classes'][entity_name]
                    }

        return None

    def _get_entity_by_path(self, module_path: str, entity_name: str) -> Optional[Dict[str, Any]]:
        modules = self.analysis_data.get('modules_with_code', {})

        for path, module_data in modules.items():
            if module_data.get('path') == module_path:
                # Check functions
                if entity_name in module_data.get('functions', {}):
                    return {
                        'type': 'function',
                        'data': module_data['functions'][entity_name]
                    }
                # Check classes
                elif entity_name in module_data.get('classes', {}):
                    return {
                        'type': 'class',
                        'data': module_data['classes'][entity_name]
                    }

        return None