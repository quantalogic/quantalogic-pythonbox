"""
Utility functions for the PythonBox interpreter.
"""

import ast


def has_await(node: ast.AST) -> bool:
    """
    Check if an AST node contains an await expression.

    Args:
        node: The AST node to check.

    Returns:
        bool: True if the node contains an await expression, False otherwise.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Await):
            return True
    return False