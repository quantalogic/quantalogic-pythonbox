from typing import Any, Dict, Optional


class AsyncFunction:
    async def __call__(self, *args: Any, _return_locals: bool = False, **kwargs: Any) -> Any:
        pass


class AsyncExecutionResult:
    def __init__(self, result: Any, error: Optional[str], execution_time: float, local_variables: Optional[Dict[str, Any]]):
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.local_variables = local_variables