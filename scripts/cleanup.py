#!/usr/bin/env python3
"""
Project cleanup script for quantalogic-pythonbox.

This script removes temporary files, cache directories, and build artifacts
to keep the project directory clean.
"""

import os
import shutil
from pathlib import Path


def remove_if_exists(path):
    """Remove a file or directory if it exists."""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else:
            os.remove(path)
            print(f"Removed file: {path}")


def cleanup_project():
    """Clean up the project directory."""
    print("Starting project cleanup...")
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Python cache and build artifacts
    patterns_to_remove = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/*.egg-info",
        "**/build",
        "**/dist",
        "**/.pytest_cache",
        "**/.mypy_cache",
        "**/.coverage",
        "**/htmlcov",
        "**/.tox",
        "**/.nox",
    ]
    
    for pattern in patterns_to_remove:
        for path in project_root.glob(pattern):
            remove_if_exists(path)
    
    # Specific files and directories
    specific_paths = [
        ".DS_Store",
        "*.log",
        "*.tmp",
        "*.temp",
        "*.bak",
        "*.backup",
    ]
    
    for pattern in specific_paths:
        for path in project_root.glob(pattern):
            remove_if_exists(path)
    
    print("Project cleanup completed!")


if __name__ == "__main__":
    cleanup_project()
