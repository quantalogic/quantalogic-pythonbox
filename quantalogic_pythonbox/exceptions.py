import ast
from typing import Any, List

class ReturnException(Exception):
    def __init__(self, value: Any) -> None:
        self.value: Any = value

class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

class YieldException(Exception):
    """Exception used internally to signal generator yields."""
    def __init__(self, value=None, return_value=None):
        self.value = value  # The value being yielded to the caller
        self.return_value = return_value  # The value that the yield expression should return
        super().__init__()

class BaseExceptionGroup(Exception):
    def __init__(self, message: str, exceptions: List[Exception]):
        super().__init__(message)
        self.exceptions = exceptions
        self.message = message

    def __str__(self):
        return f"{self.message}: {', '.join(str(e) for e in self.exceptions)}"

class WrappedException(Exception):
    def __init__(self, message: str, original_exception: Exception, lineno: int, col: int, context_line: str):
        super().__init__(message)
        self.original_exception: Exception = original_exception
        self.lineno: int = lineno
        self.col: int = col
        self.context_line: str = context_line
        self.message = message  # Use the provided message directly for clarity

    def __str__(self):
        exc_type = type(self.original_exception).__name__
        return f"{exc_type} at line {self.lineno}, col {self.col}:\n{self.context_line}\nDescription: {self.message}"

def has_await(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Await):
            return True
    return False