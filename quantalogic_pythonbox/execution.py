import ast
import asyncio
import logging
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from .interpreter_core import ASTInterpreter
from .function_utils import Function, AsyncFunction, AsyncGeneratorFunction
from .exceptions import WrappedException
from .result_function import AsyncExecutionResult
from .generator_wrapper import GeneratorWrapper

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ControlledEventLoop:
    """Encapsulated event loop management to prevent unauthorized access"""
    def __init__(self):
        self._loop = None
        self._created = False
        self._lock = asyncio.Lock()

    async def get_loop(self) -> asyncio.AbstractEventLoop:
        async with self._lock:
            if self._loop is None:
                try:
                    # Prefer the current running event loop
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    # Create a new loop only if no running loop exists
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                    self._created = True
            return self._loop

    async def cleanup(self):
        async with self._lock:
            if self._created and self._loop and not self._loop.is_closed():
                for task in asyncio.all_tasks(self._loop):
                    task.cancel()
                await asyncio.gather(*asyncio.all_tasks(self._loop), return_exceptions=True)
                self._loop.close()
                self._loop = None
                self._created = False

    async def run_task(self, coro, timeout: float) -> Any:
        return await asyncio.wait_for(coro, timeout=timeout)

async def _async_execute_async(
    code: str,
    entry_point: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    timeout: float = 30,
    allowed_modules: Optional[List[str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    max_memory_mb: int = 1024,
    ignore_typing: bool = False
) -> AsyncExecutionResult:
    start_time = time.time()
    interpreter = None
    event_loop_manager = ControlledEventLoop()
    
    # Set default allowed modules
    if allowed_modules is None:
        allowed_modules = [
            'asyncio', 'json', 'math', 'random', 're', 'inspect', 'datetime', 'time',
            'collections', 'itertools', 'functools', 'operator', 'typing',
            'decimal', 'fractions', 'statistics', 'array', 'bisect', 'heapq',
            'copy', 'enum', 'uuid'
        ]
    
    try:
        dedented_code = textwrap.dedent(code).strip()
        ast_tree = ast.parse(dedented_code)
        # Detect default entry point 'main' if not specified
        func_names = [node.name for node in ast_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if entry_point is None and 'main' in func_names:
            entry_point = 'main'
        # Check if specified entry_point exists among defined functions
        if entry_point:
            # collect top-level function and async function names
            func_names = [node.name for node in ast_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if entry_point not in func_names:
                return AsyncExecutionResult(
                    result=None,
                    error=f"Entry point '{entry_point}' not found",
                    execution_time=time.time() - start_time,
                    local_variables={}
                )
        
        loop = await event_loop_manager.get_loop()
        
        # Prepare execution namespace
        safe_namespace = namespace.copy() if namespace else {}
        safe_namespace.pop('asyncio', None)
        safe_namespace['logging'] = logging
        
        interpreter = ASTInterpreter(
            allowed_modules=allowed_modules,
            restrict_os=True,
            namespace=safe_namespace,
            max_memory_mb=max_memory_mb,
            source=dedented_code,
            ignore_typing=ignore_typing
        )
        interpreter.loop = loop
        interpreter.initialize_visitors()  # Initialize visitor methods
        
        async def run_execution(entry_point=None):
            return await interpreter.execute_async(ast_tree, entry_point)

        module_result = None
        if entry_point:
            module_result = await event_loop_manager.run_task(run_execution(entry_point), timeout=timeout)
        else:
            module_result = await event_loop_manager.run_task(run_execution(), timeout=timeout)

        # module_result is a tuple (result, locals)
        result_raw, local_vars = module_result
        # Handle GeneratorWrapper for synchronous generators: consume yields into a list
        if isinstance(result_raw, GeneratorWrapper):
            result = list(result_raw)
        # Handle StopIteration return values carrying via args or value attribute
        elif isinstance(result_raw, tuple) and local_vars is None:
            # fallback for functions returning tuple directly
            result = result_raw
        else:
            if hasattr(result_raw, 'value'):
                result = result_raw.value
            elif hasattr(result_raw, '__iter__') and not isinstance(result_raw, Exception):
                result = result_raw
            else:
                result = result_raw
        # Filter out private vars
        filtered_locals = {k: v for k, v in local_vars.items() if not k.startswith('__')}
        return AsyncExecutionResult(
            result=result,
            error=None,
            execution_time=time.time() - start_time,
            local_variables=filtered_locals
        )
    except asyncio.TimeoutError as e:
        return AsyncExecutionResult(
            result=None,
            error=f'TimeoutError: Execution exceeded {timeout} seconds: {str(e)}',
            execution_time=time.time() - start_time,
            local_variables={}
        )
    except WrappedException as e:
        # Unwrap to report original exception
        orig_exc = getattr(e, 'original_exception', e)
        error_msg = f"{type(orig_exc).__name__}: {str(orig_exc)}"
        return AsyncExecutionResult(
            result=None,
            error=error_msg,
            execution_time=time.time() - start_time,
            local_variables={}
        )
    except Exception as exc:
        # Generic exception: report error with None result
        local_vars = {}
        return AsyncExecutionResult(
            result=None,
            error=f"{type(exc).__name__}: {str(exc)}",
            execution_time=time.time() - start_time,
            local_variables=local_vars
        )
    finally:
        await event_loop_manager.cleanup()

def execute_async(
    code: str,
    entry_point: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    timeout: float = 30,
    allowed_modules: Optional[List[str]] = None,
    namespace: Optional[Dict[str, Any]] = None,
    max_memory_mb: int = 1024,
    ignore_typing: bool = False
) -> AsyncExecutionResult:
    """
    Wrapper for execute_async supporting both sync and async usage.
    Returns coroutine if in async context, else runs via asyncio.run.
    """
    coro = _async_execute_async(
        code, entry_point, args, kwargs, timeout, allowed_modules, namespace, max_memory_mb, ignore_typing
    )
    try:
        asyncio.get_running_loop()
        return coro
    except RuntimeError:
        return asyncio.run(coro)

def interpret_ast(ast_tree: ast.AST, allowed_modules: List[str], source: str = "", restrict_os: bool = False, namespace: Optional[Dict[str, Any]] = None) -> Any:
    event_loop_manager = ControlledEventLoop()
    
    safe_namespace = namespace.copy() if namespace else {}
    safe_namespace.pop('asyncio', None)
    safe_namespace['logging'] = logging
    
    interpreter = ASTInterpreter(allowed_modules=allowed_modules, source=source, restrict_os=restrict_os, namespace=safe_namespace)
    interpreter.initialize_visitors()  # Initialize visitor methods
    
    async def run_interpreter():
        loop = await event_loop_manager.get_loop()
        interpreter.loop = loop
        result = await interpreter.visit(ast_tree, wrap_exceptions=True)
        return result

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        interpreter.loop = loop
        result = loop.run_until_complete(run_interpreter())
        return result
    finally:
        if not loop.is_closed():
            loop.close()