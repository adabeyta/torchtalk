#!/usr/bin/env python3
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TorchTalkConfig:
    # Context settings
    context_profile: str = "dev"  # dev, production_128k, production_1m

    # Repository settings
    repo_path: str = "pytorch"

    # Dynamic file paths (auto-generated based on repo)
    artifacts_dir: str = "artifacts"
    _analysis_file: Optional[str] = None

    @property
    def repo_name(self) -> str:
        return Path(self.repo_path).name

    @property
    def analysis_file(self) -> str:
        if self._analysis_file:
            return self._analysis_file
        return f"{self.artifacts_dir}/{self.repo_name}_enhanced_analysis.json"

    @analysis_file.setter
    def analysis_file(self, value: str):
        self._analysis_file = value
    
    # vLLM settings
    vllm_endpoint: str = "http://localhost:8000/v1/chat/completions"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Default model for dev
    
    # Service ports
    fastapi_port: int = 8001
    gradio_port: int = 7860
    vllm_port: int = 8000
    
    def save(self, config_file: str = "torchtalk_config.json"):
        with open(config_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, config_file: str = "torchtalk_config.json") -> 'TorchTalkConfig':
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()
    
    def update_from_env(self):
        env_mappings = {
            'CONTEXT_PROFILE': 'context_profile',
            'REPO_PATH': 'repo_path',
            'VLLM_ENDPOINT': 'vllm_endpoint',
            'MODEL_NAME': 'model_name',
            'FASTAPI_PORT': 'fastapi_port',
            'GRADIO_PORT': 'gradio_port',
            'VLLM_PORT': 'vllm_port',
            'ARTIFACTS_DIR': 'artifacts_dir'
        }
        
        for env_var, attr_name in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert ports to int
                if 'port' in attr_name:
                    value = int(value)
                setattr(self, attr_name, value)
    
    def to_env_vars(self) -> dict:
        return {
            'CONTEXT_PROFILE': self.context_profile,
            'REPO_PATH': self.repo_path,
            'ANALYSIS_FILE': self.analysis_file,
            'VLLM_ENDPOINT': self.vllm_endpoint,
            'MODEL_NAME': self.model_name,
            'FASTAPI_PORT': str(self.fastapi_port),
            'GRADIO_PORT': str(self.gradio_port),
            'VLLM_PORT': str(self.vllm_port),
            'ARTIFACTS_DIR': self.artifacts_dir
        }

# Global config instance
_config = None

def get_config() -> TorchTalkConfig:
    global _config
    if _config is None:
        _config = TorchTalkConfig.load()
        _config.update_from_env()
    return _config

def set_config(config: TorchTalkConfig):
    global _config
    _config = config

