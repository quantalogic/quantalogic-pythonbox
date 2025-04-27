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
        self.gen_coroutine = self.gen()  # Store the coroutine object
        self.logger = logging.getLogger(__name__)

    async def gen(self):
        self.logger.debug(f"Async generator gen started for {self.node.name}")
        try:
            for stmt in self.node.body:
                self.logger.debug(f"Executing statement in gen: {stmt.__class__.__name__}")
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Yield):
                    yield_value = await self.interpreter.visit(stmt.value.value)
                    self.logger.debug(f"Yielding in gen: {yield_value}")
                    sent_value = yield yield_value
                    self.logger.debug(f"Received in gen: {sent_value}")
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        await self.interpreter.set_variable(target.id, sent_value)
                        self.logger.debug(f"Assigned sent value to variable '{target.id}': {sent_value}, type: {type(sent_value).__name__}")
                    else:
                        raise Exception(f"Unsupported target type for yield assignment: {target.__class__.__name__}")
                elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.YieldFrom):
                    yield_from_value = await self.interpreter.visit(stmt.value.value)
                    async for item in yield_from_value:
                        self.logger.debug(f"Yielding from item: {item}")
                        sent_value = yield item
                        self.logger.debug(f"Received sent value: {sent_value}")
                elif isinstance(stmt, ast.Yield):
                    yield_value = await self.interpreter.visit(stmt.value)
                    self.logger.debug(f"Yielding in gen: {yield_value}")
                    sent_value = yield yield_value
                    self.logger.debug(f"Received in gen: {sent_value}")
                elif isinstance(stmt, ast.YieldFrom):
                    yield_from_value = await self.interpreter.visit(stmt.value)
                    async for item in yield_from_value:
                        self.logger.debug(f"Yielding from item: {item}")
                        sent_value = yield item
                        self.logger.debug(f"Received sent value: {sent_value}")
                else:
                    await self.interpreter.visit(stmt, wrap_exceptions=True)
        except ReturnException as ret:
            self.logger.debug(f"Gen caught ReturnException: {ret.value}")
            raise StopAsyncIteration(ret.value or None)
        except Exception as e:
            self.logger.error(f"Gen exception: {str(e)}")
            raise

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.logger.debug(f"__call__ invoked for {self.node.name} with args: {args}, kwargs: {kwargs}")
        logger.debug(f"Starting AsyncGeneratorFunction {self.node.name} with args: {args}, kwargs: {kwargs}")
        
        # Use instance variables for argument binding
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
        new_interp = self.interpreter.spawn_from_env(self.closure + [local_frame])
        self.interpreter = new_interp  # Update interpreter if needed

        class AsyncGenerator:
            def __init__(self, gen_coroutine):
                self.gen_coroutine = gen_coroutine
                self.logger = logging.getLogger(__name__)

            async def __anext__(self):
                self.logger.debug(f"__anext__ called for generator {self.gen_coroutine.cr_code.co_name}, attempting asend(None)")
                try:
                    value = await self.gen_coroutine.asend(None)
                    self.logger.debug(f"__anext__ yielded value: {value}")
                    self.logger.debug(f"__anext__ call details: coroutine={self.gen_coroutine.cr_code.co_name}, value={value}, type={type(value).__name__}")
                    return value
                except StopAsyncIteration as e:
                    self.logger.debug(f"__anext__ raised StopAsyncIteration with value: {getattr(e, 'value', 'No value')}")
                    self.logger.debug(f"StopAsyncIteration details: type={type(e).__name__}, value={getattr(e, 'value', 'No value')}")
                    raise
                except Exception as exc:
                    self.logger.error(f"Exception in __anext__: {str(exc)}, type: {type(exc).__name__}")
                    self.logger.error(f"__anext__ exception details: coroutine={self.gen_coroutine.cr_code.co_name}, exception={str(exc)}, type={type(exc).__name__}")
                    raise
            def __aiter__(self):
                return self

        gen_coroutine = self.gen()
        async_gen_instance = AsyncGenerator(gen_coroutine)
        self.logger.debug(f"Returning AsyncGenerator instance for {self.node.name}")
        return async_gen_instance