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
        logger.debug(f"Starting AsyncGeneratorFunction {self.node.name}")
        
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}
        local_frame[self.node.name] = self

        num_posonly = len(self.posonly_params)
        num_pos_kw = len(self.pos_kw_params)
        total_pos = num_posonly + num_pos_kw
        pos_args_used = 0

        for i, arg in enumerate(args):
            if i < num_posonly:
                local_frame[self.posonly_params[i]] = arg
                pos_args_used += 1
            elif i < total_pos:
                local_frame[self.pos_kw_params[i - num_posonly]] = arg
                pos_args_used += 1
            elif self.vararg_name:
                if self.vararg_name not in local_frame:
                    local_frame[self.vararg_name] = []
                local_frame[self.vararg_name].append(arg)
            else:
                raise TypeError(f"Async generator '{self.node.name}' takes {total_pos} positional arguments but {len(args)} were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.posonly_params:
                raise TypeError(f"Async generator '{self.node.name}' got an unexpected keyword argument '{kwarg_name}' (positional-only)")
            elif kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError(f"Async generator '{self.node.name}' got multiple values for argument '{kwarg_name}'")
                local_frame[kwarg_name] = kwarg_value
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
            else:
                raise TypeError(f"Async generator '{self.node.name}' got an unexpected keyword argument '{kwarg_name}'")

        for param in self.pos_kw_params:
            if param not in local_frame and param in self.pos_defaults:
                local_frame[param] = self.pos_defaults[param]
        for param in self.kwonly_params:
            if param not in local_frame and param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]

        missing_args = [param for param in self.posonly_params if param not in local_frame]
        missing_args += [param for param in self.pos_kw_params if param not in local_frame and param not in self.pos_defaults]
        missing_args += [param for param in self.kwonly_params if param not in local_frame and param not in self.kw_defaults]
        if missing_args:
            raise TypeError(f"Async generator '{self.node.name}' missing required arguments: {', '.join(missing_args)}")

        new_env_stack.append(local_frame)
        new_interp: ASTInterpreter = self.interpreter.spawn_from_env(new_env_stack)

        async def execute():
            logger.debug(f"Starting execution of {self.node.name}")
            new_interp.generator_context['active'] = True
            try:
                for stmt in self.node.body:
                    if isinstance(stmt, ast.Try):
                        try:
                            for body_stmt in stmt.body:
                                if isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Yield):
                                    value = await new_interp.visit(body_stmt.value.value, wrap_exceptions=True) if body_stmt.value.value else None
                                    sent_value = yield value
                                    if isinstance(sent_value, BaseException):
                                        raise sent_value
                                elif isinstance(body_stmt, ast.Assign) and len(body_stmt.targets) == 1 and isinstance(body_stmt.value, ast.Yield):
                                    value = await new_interp.visit(body_stmt.value.value, wrap_exceptions=True) if body_stmt.value.value else None
                                    sent_value = yield value
                                    if isinstance(sent_value, BaseException):
                                        raise sent_value
                                    await new_interp.assign(body_stmt.targets[0], sent_value)
                                else:
                                    await new_interp.visit(body_stmt, wrap_exceptions=True)
                        except Exception as e:
                            handled = False
                            for handler in stmt.handlers:
                                exc_type = await new_interp._resolve_exception_type(handler.type)
                                if exc_type and isinstance(e, exc_type):
                                    if handler.name:
                                        new_interp.set_variable(handler.name, e)
                                    for handler_stmt in handler.body:
                                        if isinstance(handler_stmt, ast.Expr) and isinstance(handler_stmt.value, ast.Yield):
                                            value = await new_interp.visit(handler_stmt.value.value, wrap_exceptions=True) if handler_stmt.value.value else None
                                            sent_value = yield value
                                            if isinstance(sent_value, BaseException):
                                                raise sent_value
                                        elif isinstance(handler_stmt, ast.Assign) and len(handler_stmt.targets) == 1 and isinstance(handler_stmt.value, ast.Yield):
                                            value = await new_interp.visit(handler_stmt.value.value, wrap_exceptions=True) if body_stmt.value.value else None
                                            sent_value = yield value
                                            if isinstance(sent_value, BaseException):
                                                raise sent_value
                                            await new_interp.assign(handler_stmt.targets[0], sent_value)
                                        else:
                                            await new_interp.visit(handler_stmt, wrap_exceptions=True)
                                    handled = True
                                    break
                            if not handled:
                                raise e
                    else:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                            value = await new_interp.visit(stmt.value.value, wrap_exceptions=True) if stmt.value.value else None
                            sent_value = yield value
                            if isinstance(sent_value, BaseException):
                                raise sent_value
                        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.value, ast.Yield):
                            value = await new_interp.visit(stmt.value.value, wrap_exceptions=True) if stmt.value.value else None
                            sent_value = yield value
                            if isinstance(sent_value, BaseException):
                                raise sent_value
                            await new_interp.assign(stmt.targets[0], sent_value)
                        else:
                            await new_interp.visit(stmt, wrap_exceptions=True)
                raise StopAsyncIteration
            except ReturnException as ret:
                logger.debug(f"Caught ReturnException with value: {ret.value}")
                raise StopAsyncIteration(ret.value or None)
            except Exception as e:
                logger.debug(f"Caught exception in generator: {type(e).__name__}")
                raise e
            finally:
                logger.debug("Execution finished, setting active to False")
                new_interp.generator_context['active'] = False
                new_interp.generator_context['finished'] = True

        class AsyncGenerator:
            def __init__(self):
                self.execute_gen = execute()
                self.return_value = None
                self.result = None
                self.yielded_values = []

            def __aiter__(self):
                logger.debug("AsyncGenerator.__aiter__ called")
                return self

            async def __anext__(self):
                logger.debug("Entering AsyncGenerator.__anext__")
                try:
                    value = await self.execute_gen.asend(None)
                    logger.debug(f"Returning value from __anext__: {value}")
                    self.yielded_values.append(value)
                    return value
                except StopAsyncIteration as e:
                    logger.debug(f"StopAsyncIteration in __anext__ with args: {e.args}")
                    self.return_value = e.args[0] if e.args else None
                    self.result = self.yielded_values if self.yielded_values else "Empty generator"
                    raise
                except Exception as e:
                    logger.debug(f"Exception in __anext__: {type(e).__name__}")
                    raise

            async def asend(self, value):
                logger.debug(f"Entering AsyncGenerator.asend with value: {value}")
                try:
                    value = await self.execute_gen.asend(value)
                    logger.debug(f"Returning value from asend: {value}")
                    self.yielded_values.append(value)
                    return value
                except StopAsyncIteration as e:
                    logger.debug(f"StopAsyncIteration in asend with args: {e.args}")
                    self.return_value = e.args[0] if e.args else None
                    self.result = self.yielded_values if self.yielded_values else "Empty generator"
                    raise
                except Exception as e:
                    logger.debug(f"Exception in asend: {type(e).__name__}")
                    raise

            async def athrow(self, exc_type, exc_val=None, exc_tb=None):
                logger.debug(f"Entering AsyncGenerator.athrow with exc_type: {exc_type}")
                try:
                    if exc_val is None:
                        if isinstance(exc_type, type):
                            exc_val = exc_type()
                        else:
                            exc_val = exc_type
                    value = await self.execute_gen.athrow(exc_val)
                    logger.debug(f"Returning value from athrow: {value}")
                    self.yielded_values.append(value)
                    return value
                except StopAsyncIteration as e:
                    logger.debug(f"StopAsyncIteration in athrow with args: {e.args}")
                    self.return_value = e.args[0] if e.args else None
                    self.result = self.yielded_values if self.yielded_values else "Empty generator"
                    raise
                except Exception as e:
                    logger.debug(f"Exception in athrow: {type(e).__name__}")
                    self.result = "caught"
                    raise e

            async def aclose(self):
                logger.debug("Entering AsyncGenerator.aclose")
                try:
                    await self.execute_gen.aclose()
                    logger.debug("Generator closed")
                except Exception as e:
                    logger.debug(f"Exception during aclose: {e}")
                return self.return_value

        logger.debug("Returning AsyncGenerator instance")
        return AsyncGenerator()