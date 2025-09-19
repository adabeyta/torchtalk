#!/usr/bin/env python3
from pathlib import Path
from typing import Dict
from torchtalk.context.assembler_core import ContextAssemblerCore


class DynamicContextAssembler:
    def __init__(self, compendium_path: str = None, dynamic_analysis_path: str = None):

        # Auto-detect enhanced analysis file
        enhanced_files = list(Path("artifacts").glob("*_enhanced_analysis.json"))

        if enhanced_files:
            # Use the first enhanced analysis found
            enhanced_path = str(enhanced_files[0])
            self.assembler = ContextAssemblerCore(enhanced_path)
        else:
            raise FileNotFoundError(
                "No enhanced analysis found. Please run: python repo_analyzer.py <repo_path>"
            )

    def assemble_context(self, question: str, max_chars: int = 3200000) -> str:
        return self.assembler.assemble_context(question, max_chars)

    def get_context_info(self, question: str) -> Dict:
        return self.assembler.get_context_info(question) # Mostly for debug.