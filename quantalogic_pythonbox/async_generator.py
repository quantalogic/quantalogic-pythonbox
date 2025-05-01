# quantalogic_pythonbox/async_generator.py
"""
Async generator function handling for the PythonBox interpreter.
"""

import ast
import asyncio
import inspect as _inspect_module
import logging
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Patch inspect to recognize our async generator types
_orig_isasyncgenfunction = _inspect_module.isasyncgenfunction
def isasyncgenfunction(obj):
    from .async_generator import AsyncGeneratorFunction
    return isinstance(obj, AsyncGeneratorFunction) or _orig_isasyncgenfunction(obj)
_inspect_module.isasyncgenfunction = isasyncgenfunction

_orig_isasyncgen = _inspect_module.isasyncgen
def isasyncgen(obj):
    from .async_generator import AsyncGenerator
    return isinstance(obj, AsyncGenerator) or _orig_isasyncgen(obj)
_inspect_module.isasyncgen = isasyncgen

from .interpreter_core import ASTInterpreter
from .exceptions import ReturnException

class StopAsyncIterationWithValue(StopAsyncIteration):
    """Custom StopAsyncIteration carrying return value via .value."""
    def __init__(self, value):
        super().__init__()
        self.value = value  # Directly set value as an attribute

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
        self.logger.debug(f"Entering gen method for {self.node.name}")
        for idx, stmt in enumerate(self.node.body):
            self.logger.debug(f"Gen stmt #{idx}: {stmt.__class__.__name__}")
        try:
            for stmt in self.node.body:
                if isinstance(stmt, ast.Return):
                    ret_val = await self.interpreter.visit(stmt.value, wrap_exceptions=False) if stmt.value else None
                    if ret_val is not None:
                        self.logger.debug(f"Raising StopAsyncIterationWithValue with value: {ret_val}")
                        raise StopAsyncIterationWithValue(ret_val)
                    else:
                        self.logger.debug("Raising StopAsyncIteration without value")
                        raise StopAsyncIteration()
                if isinstance(stmt, ast.For):
                    iterable = await self.interpreter.visit(stmt.iter, wrap_exceptions=False)
                    for item in iterable:
                        await self.interpreter.assign(stmt.target, item)
                        for inner in stmt.body:
                            if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Yield):
                                yield_val = await self.interpreter.visit(inner.value.value, wrap_exceptions=False)
                                yield yield_val
                            else:
                                result = await self.interpreter.visit(inner, wrap_exceptions=False)
                                if result is not None:
                                    yield result
                    continue
                if isinstance(stmt, ast.AsyncFor):
                    iterable = await self.interpreter.visit(stmt.iter, wrap_exceptions=False)
                    if not hasattr(iterable, '__aiter__'):
                        raise TypeError(f"Object {iterable} is not an async iterable")
                    iterator = iterable.__aiter__()
                    while True:
                        try:
                            value = await iterator.__anext__()
                        except StopAsyncIteration:
                            break
                        await self.interpreter.assign(stmt.target, value)
                        for inner in stmt.body:
                            if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Yield):
                                yield_val = await self.interpreter.visit(inner.value.value, wrap_exceptions=False)
                                yield yield_val
                            else:
                                result = await self.interpreter.visit(inner, wrap_exceptions=False)
                                if result is not None:
                                    yield result
                    continue
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Yield):
                    yield_val = await self.interpreter.visit(stmt.value.value, wrap_exceptions=False)
                    self.logger.debug(f"Yielding value from assign: {yield_val}")
                    try:
                        sent = yield yield_val
                    except Exception as e:
                        self.logger.debug(f"Exception at yield, delegating to AST handler: {e}")
                        handler_res = await self.interpreter.visit(stmt, wrap_exceptions=True)
                        if handler_res is not None:
                            yield handler_res
                        continue
                    await self.interpreter.assign(stmt.targets[0], sent)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                    yield_val = await self.interpreter.visit(stmt.value.value, wrap_exceptions=False)
                    self.logger.debug(f"Yielding value from expr: {yield_val}")
                    yield yield_val
                elif isinstance(stmt, ast.Try):
                    try:
                        for inner in stmt.body:
                            if isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Yield):
                                yield_val = await self.interpreter.visit(inner.value.value, wrap_exceptions=False)
                                try:
                                    sent = yield yield_val
                                except Exception as e:
                                    self.logger.debug(f"Exception at yield, delegating to AST handler: {e}")
                                    handler_res = await self.interpreter.visit(inner, wrap_exceptions=True)
                                    if handler_res is not None:
                                        yield handler_res
                                    continue
                                await self.interpreter.assign(inner.targets[0], sent)
                            elif isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Yield):
                                yield_val = await self.interpreter.visit(inner.value.value, wrap_exceptions=False)
                                yield yield_val
                            else:
                                result = await self.interpreter.visit(inner, wrap_exceptions=False)
                                if result is not None:
                                    yield result
                    except ReturnException as re:
                        if re.value is not None:
                            self.logger.debug(f"Raising StopAsyncIterationWithValue from ReturnException with value: {re.value}")
                            raise StopAsyncIterationWithValue(re.value)
                        else:
                            self.logger.debug("Raising StopAsyncIteration from ReturnException without value")
                            raise StopAsyncIteration()
                    except GeneratorExit:
                        for fstmt in stmt.finalbody:
                            if isinstance(fstmt, ast.Expr) and isinstance(fstmt.value, ast.Yield):
                                yield_val = await self.interpreter.visit(fstmt.value.value, wrap_exceptions=False)
                                yield yield_val
                            else:
                                result = await self.interpreter.visit(fstmt, wrap_exceptions=False)
                                if result is not None:
                                    yield result
                        raise
                    except Exception as e:
                        for handler in stmt.handlers:
                            exc_type = await self.interpreter._resolve_exception_type(handler.type) if handler.type else Exception
                            if isinstance(e, exc_type):
                                if handler.name:
                                    self.interpreter.set_variable(handler.name, e)
                                for hstmt in handler.body:
                                    if isinstance(hstmt, ast.Expr) and isinstance(hstmt.value, ast.Yield):
                                        yield_val = await self.interpreter.visit(hstmt.value.value, wrap_exceptions=False)
                                        yield yield_val
                                    else:
                                        result = await self.interpreter.visit(hstmt, wrap_exceptions=False)
                                        if result is not None:
                                            yield result
                                break
                        else:
                            raise
                    for fstmt in stmt.finalbody:
                        if isinstance(fstmt, ast.Expr) and isinstance(fstmt.value, ast.Yield):
                            yield_val = await self.interpreter.visit(fstmt.value.value, wrap_exceptions=False)
                            yield yield_val
                        else:
                            result = await self.interpreter.visit(fstmt, wrap_exceptions=False)
                            if result is not None:
                                yield result
                else:
                    try:
                        result = await self.interpreter.visit(stmt, wrap_exceptions=False)
                    except ReturnException as ret:
                        if ret.value is not None:
                            self.logger.debug(f"Raising StopAsyncIterationWithValue from ReturnException with value: {ret.value}")
                            raise StopAsyncIterationWithValue(ret.value)
                        else:
                            self.logger.debug("Raising StopAsyncIteration from ReturnException without value")
                            raise StopAsyncIteration()
                    if result is not None:
                        yield result
        except ReturnException as re:
            if re.value is not None:
                self.logger.debug(f"Raising StopAsyncIterationWithValue with value: {re.value}")
                raise StopAsyncIterationWithValue(re.value)
            else:
                self.logger.debug("Raising StopAsyncIteration without value")
                raise StopAsyncIteration()
        self.logger.debug(f"Raising StopAsyncIteration at end of generator {self.node.name}")
        raise StopAsyncIteration()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.logger.debug(f"__call__ invoked for {self.node.name} with args: {args}, kwargs: {kwargs}")
        logger.debug(f"Starting AsyncGeneratorFunction {self.node.name} with args: {args}, kwargs: {kwargs}")
        
        local_frame: Dict[str, Any] = {}
        
        for name, value in zip(self.pos_kw_params, args):
            local_frame[name] = value
        
        for param, default in self.pos_defaults.items():
            if param not in local_frame:
                local_frame[param] = default
        
        for param in self.kwonly_params:
            if param in kwargs:
                local_frame[param] = kwargs[param]
            elif param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]
        
        if self.vararg_name:
            remaining_pos_args = args[len(self.pos_kw_params):]
            local_frame[self.vararg_name] = remaining_pos_args
        if self.kwarg_name:
            local_frame[self.kwarg_name] = kwargs
        
        new_interp = self.interpreter.spawn_from_env(self.closure + [local_frame])
        self.interpreter = new_interp

        self.interpreter.generator_context = {'yield_queue': asyncio.Queue(), 'sent_queue': asyncio.Queue(), 'active': True}
        gen_coroutine = self.gen()
        self.logger.debug(f"gen_coroutine type in __call__: {type(gen_coroutine).__name__}")
        async_gen_instance = AsyncGenerator(gen_coroutine, self.node.name, self.interpreter)
        self.logger.debug(f"Returning AsyncGenerator instance for {self.node.name}")
        return async_gen_instance

class AsyncGenerator:
    def __init__(self, gen_coroutine, gen_name, interpreter):
        self.gen_coroutine = gen_coroutine
        self.gen_name = gen_name
        self.interpreter = interpreter
        self.logger = logging.getLogger(__name__)

    async def __anext__(self):
        self.logger.debug(f'__anext__ called for generator {self.gen_name}, sending None')
        try:
            value = await self.gen_coroutine.asend(None)
            self.logger.debug(f'Value yielded in __anext__: {value}, type: {type(value)}')
            if value is None:
                self.logger.warning(f'Warning: Yielding None in __anext__ for generator {self.gen_name}')
            return value
        except StopAsyncIterationWithValue as e:
            self.logger.debug(f'Raising StopAsyncIterationWithValue with return value: {e.value} in __anext__')
            raise StopAsyncIterationWithValue(e.value)
        except StopAsyncIteration:
            self.logger.debug(f'Raising StopAsyncIteration without value in __anext__')
            raise
        except RuntimeError as re:
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIterationWithValue):
                self.logger.debug(f'Unwrapped StopAsyncIterationWithValue with return value: {ctx.value} in __anext__')
                raise StopAsyncIterationWithValue(ctx.value)
            elif isinstance(ctx, StopAsyncIteration):
                self.logger.debug(f'Unwrapped StopAsyncIteration without value in __anext__')
                raise StopAsyncIteration()
            self.logger.error(f'Unexpected RuntimeError in __anext__ for generator {self.gen_name}: {str(re)}')
            raise
        except Exception as exc:
            self.logger.error(f'Exception in __anext__ for generator {self.gen_name}: {type(exc).__name__}: {str(exc)}')
            raise
    
    async def asend(self, value):
        self.interpreter.generator_context['active'] = True
        self.logger.debug(f"asend called for generator {self.gen_name}, sending value: {value}")
        try:
            result = await self.gen_coroutine.asend(value)
            self.logger.debug(f"asend yielded value: {result}")
            return result
        except StopAsyncIterationWithValue as e:
            self.logger.debug(f"asend received StopAsyncIterationWithValue, return value: {e.value}")
            return e.value
        except StopAsyncIteration:
            self.logger.debug(f"asend received StopAsyncIteration without value")
            raise
        except RuntimeError as re:
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIterationWithValue):
                self.logger.debug(f"asend unwrapped StopAsyncIterationWithValue, return value: {ctx.value}")
                return ctx.value
            elif isinstance(ctx, StopAsyncIteration):
                self.logger.debug(f"asend unwrapped StopAsyncIteration without value")
                raise StopAsyncIteration()
            self.logger.error(f"Unexpected RuntimeError in asend for generator {self.gen_name}: {str(re)}")
            raise
        except Exception as exc:
            self.logger.error(f"Exception in asend for generator {self.gen_name}: {str(exc)}")
            raise
        finally:
            self.interpreter.generator_context['active'] = False

    async def athrow(self, exc):
        self.interpreter.generator_context['active'] = True
        self.logger.debug(f"athrow called for generator {self.gen_name}, throwing exception: {exc}")
        try:
            result = await self.gen_coroutine.athrow(exc)
            self.logger.debug(f"athrow yielded value: {result}")
            return result
        except StopAsyncIterationWithValue as e:
            self.logger.debug(f"athrow received StopAsyncIterationWithValue for generator {self.gen_name}, return value: {e.value}")
            raise StopAsyncIterationWithValue(e.value)
        except StopAsyncIteration:
            self.logger.debug(f"athrow received StopAsyncIteration without value for generator {self.gen_name}")
            raise
        except Exception as err:
            self.logger.debug(f"athrow exception {err}, re-raising for caller to handle")
            raise
        finally:
            self.interpreter.generator_context['active'] = False

    async def aclose(self):
        self.logger.debug(f"aclose called for generator {self.gen_name}")
        try:
            await self.gen_coroutine.athrow(GeneratorExit())
        except (GeneratorExit, StopAsyncIteration):
            return
        except Exception as err:
            self.logger.debug(f"Exception during aclose: {err}")
            return

    def __aiter__(self):
        return self