from typing import Any, List


class ReturnException(Exception):
    def __init__(self, value: Any) -> None:
        self.value: Any = value


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


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


class GeneratorReturn(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class AsyncGeneratorReturn(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class StopAsyncIterationWithValue(StopAsyncIteration):
    """Custom StopAsyncIteration carrying return value via .value."""
    def __init__(self, value: Any):
        super().__init__()
        self.value = value