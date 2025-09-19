#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="torchtalk",
    version="0.1.0",
    description="Streamlined PyTorch repository analysis and chat system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adrian Abeyta",
    author_email="aabeyta@redhat.com",
    url="https://github.com/adabeyta/torchtalk.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "web": ["fastapi", "uvicorn", "gradio"],
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "torchtalk=torchtalk.cli.main:main",
        ],
    },
)