from typing import Any, Dict, List, Optional, Tuple
from .execution import execute_async


class PythonBox:
    """
    High-level interface for executing code in the Quantalogic Python sandbox.
    """
    def __init__(
        self,
        allowed_modules: Optional[List[str]] = None,
        namespace: Optional[Dict[str, Any]] = None,
        max_memory_mb: int = 1024,
        ignore_typing: bool = False,
    ) -> None:
        self.allowed_modules = allowed_modules
        self.namespace = namespace
        self.max_memory_mb = max_memory_mb
        self.ignore_typing = ignore_typing

    def execute_async(
        self,
        code: str,
        entry_point: Optional[str] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 30,
    ):
        """
        Execute the given code asynchronously. Returns an AsyncExecutionResult or a coroutine in async context.
        """
        return execute_async(
            code,
            entry_point=entry_point,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            allowed_modules=self.allowed_modules,
            namespace=self.namespace,
            max_memory_mb=self.max_memory_mb,
            ignore_typing=self.ignore_typing,
        )

    def execute(
        self,
        code: str,
        entry_point: Optional[str] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 30,
    ):
        """
        Synchronous wrapper for execute_async. Blocks until completion if not in async context.
        """
        return execute_async(
            code,
            entry_point=entry_point,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            allowed_modules=self.allowed_modules,
            namespace=self.namespace,
            max_memory_mb=self.max_memory_mb,
            ignore_typing=self.ignore_typing,
        )
