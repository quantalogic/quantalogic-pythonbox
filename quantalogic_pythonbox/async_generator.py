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
        logger.debug("Starting AsyncGeneratorFunction " + self.node.name)
        
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
                raise TypeError("Async generator '" + self.node.name + "' takes " + str(total_pos) + " positional arguments but " + str(len(args)) + " were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.posonly_params:
                raise TypeError("Async generator '" + self.node.name + "' got an unexpected keyword argument '" + kwarg_name + "' (positional-only)")
            elif kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError("Async generator '" + self.node.name + "' got multiple values for argument '" + kwarg_name + "'")
                local_frame[kwarg_name] = kwarg_value
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
            else:
                raise TypeError("Async generator '" + self.node.name + "' got an unexpected keyword argument '" + kwarg_name + "'")

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
            raise TypeError("Async generator '" + self.node.name + "' missing required arguments: " + ', '.join(missing_args))

        new_env_stack.append(local_frame)
        new_interp: ASTInterpreter = self.interpreter.spawn_from_env(new_env_stack)

        async def execute():
            logger.debug("Execute method started for async generator")
            new_interp.generator_context['active'] = True
            try:
                for stmt in self.node.body:
                    logger.debug("Executing statement: " + stmt.__class__.__name__)
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Yield):
                        # Handle yield expression in assignment
                        yield_value = await new_interp.visit(stmt.value.value)
                        logger.debug("Yielding value from Yield expression: " + str(yield_value) + ", type: " + type(yield_value).__name__)
                        sent_value = yield yield_value  # Yield the value and receive sent value
                        target = stmt.targets[0]  # Assume simple target for now, e.g., ast.Name
                        if isinstance(target, ast.Name):
                            await new_interp.set_variable(target.id, sent_value)
                            logger.debug("Assigned sent value to variable '" + target.id + "': " + str(sent_value) + ", type: " + type(sent_value).__name__)
                        else:
                            raise Exception("Unsupported target type for yield assignment: " + target.__class__.__name__)
                    elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.YieldFrom):
                        # Handle YieldFrom expression in assignment
                        yield_from_value = await new_interp.visit(stmt.value.value)
                        async for item in yield_from_value:
                            logger.debug("Yielding from item in YieldFrom expression: " + str(item) + ", type: " + type(item).__name__)
                            sent_value = yield item  # Yield each item and receive sent value
                            # For simplicity, ignore sent value in YieldFrom loop; can be extended if needed
                        # After YieldFrom, assign any final value if applicable (not standard, so skip for now)
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                        yield_value = await new_interp.visit(stmt.value.value)
                        logger.debug("Yielding value from Yield expression: " + str(yield_value) + ", type: " + type(yield_value).__name__)
                        sent_value = yield yield_value  # Yield and receive sent value, but no assignment
                    elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.YieldFrom):
                        yield_from_value = await new_interp.visit(stmt.value.value)
                        async for item in yield_from_value:
                            logger.debug("Yielding from item in YieldFrom expression: " + str(item) + ", type: " + type(item).__name__)
                            sent_value = yield item  # Yield and receive sent value
                    elif isinstance(stmt, ast.Yield):
                        yield_value = await new_interp.visit(stmt.value)
                        logger.debug("Yielding value from direct Yield node: " + str(yield_value) + ", type: " + type(yield_value).__name__)
                        sent_value = yield yield_value  # Yield and receive sent value for direct Yield
                    elif isinstance(stmt, ast.YieldFrom):
                        yield_from_value = await new_interp.visit(stmt.value)
                        async for item in yield_from_value:
                            logger.debug("Yielding from item in direct YieldFrom node: " + str(item) + ", type: " + type(item).__name__)
                            sent_value = yield item  # Yield and receive sent value
                    else:
                        await new_interp.visit(stmt, wrap_exceptions=True)
            except ReturnException as ret:
                logger.debug("Caught ReturnException with value: " + str(ret.value) + ", raising StopAsyncIteration")
                logger.error("ReturnException caught in execute: " + str(ret.value) + ", type: " + type(ret).__name__)
                raise StopAsyncIteration(ret.value or None)
            except StopAsyncIteration:
                logger.debug("Caught StopAsyncIteration in execute")
                logger.error("StopAsyncIteration caught in execute")
                raise
            except Exception as e:
                logger.error("General exception in execute: " + str(e) + ", type: " + type(e).__name__)
                raise
            finally:
                logger.debug("Execution finished, setting active to False")
                new_interp.generator_context['active'] = False
                new_interp.generator_context['finished'] = True

        class AsyncGenerator:
            def __init__(self):
                logger.debug("Initializing AsyncGenerator, setting self.gen to execute(). Type of self.gen: " + type(execute()).__name__)
                self.gen = execute()
                self.return_value = None
                self.result = None
                self.yielded_values = []

            def __aiter__(self):
                logger.debug("__aiter__ called on AsyncGenerator")
                return self

            async def __anext__(self):
                logger.debug("__anext__ called on AsyncGenerator")  
                try:
                    result = await self.gen.__anext__()
                    logger.debug("__anext__ yielded result: " + str(result) + ", type: " + type(result).__name__)
                    return result
                except StopAsyncIteration as e:
                    logger.debug("__anext__ raising StopAsyncIteration with value: " + str(e.value if hasattr(e, 'value') else 'None'))
                    raise
                except Exception as e:
                    logger.error("Exception in __anext__: " + str(e) + ", type: " + type(e).__name__)
                    raise

            async def _send(self, value):
                logger.debug(f"Type of self.gen: {type(self.gen).__name__}, value being sent: {value}")
                try:
                    return_value = self.gen.asend(value)
                    logger.debug(f"Type of return_value from self.gen.asend: {type(return_value).__name__}")
                    return await return_value
                except StopAsyncIteration as e:
                    logger.debug("StopAsyncIteration in asend with args: " + str(e.args))
                    self.return_value = e.value if e.args else None
                    self.result = self.yielded_values if self.yielded_values else []
                    self.execution_finished = True
                    self.active = False
                    # propagate async generator completion
                    raise StopAsyncIteration(self.result)

            async def asend(self, value):
                return await self._send(value)

            async def _throw(self, typ, val=None, tb=None):
                try:
                    return await self.gen.athrow(val)
                except StopAsyncIteration as e:
                    logger.debug("StopAsyncIteration in athrow with args: " + str(e.args))
                    self.return_value = e.value if e.args else None
                    self.result = self.yielded_values if self.yielded_values else []
                    self.execution_finished = True
                    self.active = False
                    # propagate async generator completion
                    raise StopAsyncIteration(self.result)

            async def athrow(self, typ, val=None, tb=None):
                return await self._throw(typ, val, tb)

            async def aclose(self):
                logger.debug("Entering AsyncGenerator.aclose")
                try:
                    await self.gen.aclose()
                    logger.debug("Generator closed")
                except Exception as e:
                    logger.debug("Exception during aclose: " + str(e))
                return self.return_value

        logger.debug("Returning AsyncGenerator instance")
        return AsyncGenerator()