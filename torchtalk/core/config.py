#!/usr/bin/env python3
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class TorchTalkConfig:
    # Context settings
    max_model_len: Optional[int] = None  # Auto-detected from vLLM, or set manually

    # Repository settings
    repo_path: str = "pytorch"

    # Dynamic file paths (auto-generated based on repo)
    artifacts_dir: str = "artifacts"
    _analysis_file: Optional[str] = None
    index_dir: Optional[str] = None  # v2.0 index location

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
    vllm_endpoint: str = "http://localhost:8080/v1/chat/completions"
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Default model for dev
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism

    # Service ports
    fastapi_port: int = 8001
    gradio_port: int = 7860
    vllm_port: int = 8080
    
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
            'MAX_MODEL_LEN': 'max_model_len',
            'REPO_PATH': 'repo_path',
            'VLLM_ENDPOINT': 'vllm_endpoint',
            'MODEL_NAME': 'model_name',
            'TENSOR_PARALLEL_SIZE': 'tensor_parallel_size',
            'FASTAPI_PORT': 'fastapi_port',
            'GRADIO_PORT': 'gradio_port',
            'VLLM_PORT': 'vllm_port',
            'ARTIFACTS_DIR': 'artifacts_dir',
            'INDEX_DIR': 'index_dir'
        }

        for env_var, attr_name in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert numeric values to int
                if 'port' in attr_name or attr_name in ['tensor_parallel_size', 'max_model_len']:
                    value = int(value)
                setattr(self, attr_name, value)
    
    def to_env_vars(self) -> dict:
        env_vars = {
            'REPO_PATH': self.repo_path,
            'ANALYSIS_FILE': self.analysis_file,
            'VLLM_ENDPOINT': self.vllm_endpoint,
            'MODEL_NAME': self.model_name,
            'TENSOR_PARALLEL_SIZE': str(self.tensor_parallel_size),
            'FASTAPI_PORT': str(self.fastapi_port),
            'GRADIO_PORT': str(self.gradio_port),
            'VLLM_PORT': str(self.vllm_port),
            'ARTIFACTS_DIR': self.artifacts_dir
        }
        if self.max_model_len:
            env_vars['MAX_MODEL_LEN'] = str(self.max_model_len)
        if self.index_dir:
            env_vars['INDEX_DIR'] = self.index_dir
        return env_vars

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

