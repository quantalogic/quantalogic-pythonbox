# quantalogic_pythonbox/async_generator.py
"""
Async generator function handling for the PythonBox interpreter.
"""

import ast
import logging
from typing import Any, Dict, List, Optional

from .interpreter_core import ASTInterpreter
from .exceptions import ReturnException

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AsyncGeneratorFunction:
    def __init__(self, node: ast.AsyncFunctionDef, closure: List[Dict[str, Any]], interpreter: ASTInterpreter,
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        self.node: ast.AsyncFunctionDef = node
        self.closure: List[Dict[str, Any]] = closure[:]
        self.interpreter: ASTInterpreter = interpreter
        self.posonly_params = [arg.arg for arg in node.args.posonlyargs] if hasattr(node.args, 'posonlyargs') else []
        self.pos_kw_params = pos_kw_params
        self.vararg_name = vararg_name
        self.kwonly_params = kwonly_params
        self.kwarg_name = kwarg_name
        self.pos_defaults = pos_defaults
        self.kw_defaults = kw_defaults

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        logger.debug(f"Starting AsyncGeneratorFunction {self.node.name} with args: {args}, kwargs: {kwargs}")
        
        # Use instance variables for argument binding
        new_env_stack = self.closure[:]
        local_frame: Dict[str, Any] = {}
        
        # Bind positional arguments using self.pos_kw_params
        for name, value in zip(self.pos_kw_params, args):
            local_frame[name] = value
        
        # Bind keyword arguments using self.kwonly_params and self.kw_defaults
        for param in self.kwonly_params:
            if param in kwargs:
                local_frame[param] = kwargs[param]
            elif param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]
        
        # Handle vararg and kwarg if defined
        if self.vararg_name:
            remaining_pos_args = args[len(self.pos_kw_params):]
            local_frame[self.vararg_name] = remaining_pos_args
        if self.kwarg_name:
            local_frame[self.kwarg_name] = kwargs
        
        # Set up the interpreter with the new environment
        new_interp = self.interpreter.spawn_from_env(new_env_stack + [local_frame])
        
        async def gen():
            logger.debug("Async generator execution started")
            try:
                for stmt in self.node.body:
                    logger.debug(f"Executing statement: {stmt.__class__.__name__}")
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Yield):
                        yield_value = await new_interp.visit(stmt.value.value)
                        sent_value = yield yield_value
                        target = stmt.targets[0]
                        if isinstance(target, ast.Name):
                            await new_interp.set_variable(target.id, sent_value)
                            logger.debug(f"Assigned sent value to variable '{target.id}': {sent_value}, type: {type(sent_value).__name__}")
                        else:
                            raise Exception(f"Unsupported target type for yield assignment: {target.__class__.__name__}")
                    elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.YieldFrom):
                        yield_from_value = await new_interp.visit(stmt.value.value)
                        async for item in yield_from_value:
                            sent_value = yield item
                            logger.debug(f"Yielding from item and received sent value: {item}, sent: {sent_value}")
                    elif isinstance(stmt, ast.Yield):
                        yield_value = await new_interp.visit(stmt.value)
                        sent_value = yield yield_value
                        logger.debug(f"Yielding value and received sent value: {yield_value}, sent: {sent_value}")
                    elif isinstance(stmt, ast.YieldFrom):
                        yield_from_value = await new_interp.visit(stmt.value)
                        async for item in yield_from_value:
                            sent_value = yield item
                            logger.debug(f"Yielding from item and received sent value: {item}, sent: {sent_value}")
                    else:
                        await new_interp.visit(stmt, wrap_exceptions=True)
            except ReturnException as ret:
                logger.debug(f"Caught ReturnException with value: {ret.value}, raising StopAsyncIteration")
                raise StopAsyncIteration(ret.value or None)
            except Exception as e:
                logger.error(f"General exception in async generator: {str(e)}, type: {type(e).__name__}")
                raise
        return gen()