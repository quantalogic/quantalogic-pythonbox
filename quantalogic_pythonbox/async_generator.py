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

from .exceptions import ReturnException, StopAsyncIterationWithValue

class AsyncGeneratorFunction:
    def __init__(self, node: ast.AsyncFunctionDef, closure: List[Dict[str, Any]], interpreter,  # Local import of ASTInterpreter inside methods to avoid circular import
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        from .interpreter_core import ASTInterpreter  # Local import
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
                                await self.interpreter.visit(inner, wrap_exceptions=False)
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
                            self.logger.debug(f"Yielding nested generator return from ReturnException: {re.value}")
                            yield re.value
                        # End generator after yielding nested return
                        return
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
        from .interpreter_core import ASTInterpreter  # Local import
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
        # track return yield for async generators
        self._return_yielded = False
        self._last_return = None
        # Create a mock agenerator attribute that mimics Python's internal structure
        self.agenerator = _MockGeneratorWithFrame(self)

    def __getattr__(self, name):
        # Fallback for any attributes not explicitly defined
        if name == 'generator':
            # For backward compatibility with old code
            class FrameView:
                def __init__(self, locals_):
                    self.f_locals = locals_
            class GenView:
                def __init__(self, locals_):
                    self.gi_frame = FrameView(locals_)
            return GenView({'__return__': self._last_return})
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

    async def __anext__(self):
        self.logger.debug(f'__anext__ called for generator {self.gen_name}, sending None')
        try:
            value = await self.gen_coroutine.asend(None)
            self.logger.debug(f'Value yielded in __anext__: {value}, type: {type(value)}')
            if value is None:
                self.logger.warning(f'Warning: Yielding None in __anext__ for generator {self.gen_name}')
            return value
        except StopAsyncIterationWithValue as e:
            # Store return value and propagate via StopAsyncIteration
            self._last_return = e.value
            self.agenerator._update_return(e.value)  # Update the return value in the mock generator
            self.logger.debug(f"Return value for generator {self.gen_name}: {e.value}")
            # Instead of raising a plain StopAsyncIteration, create one with a value attribute
            exc = StopAsyncIteration()
            exc.value = e.value  # Add the value attribute to match Python's behavior
            self.logger.debug(f"Raising StopAsyncIteration with value attribute: {e.value}")
            raise exc
        except StopAsyncIteration:
            self.logger.debug(f'Raising StopAsyncIteration without value in __anext__')
            raise
        except RuntimeError as re:
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIterationWithValue):
                self.logger.debug(f'Unwrapped StopAsyncIterationWithValue with return value: {ctx.value} in __anext__')
                self._last_return = ctx.value
                self.agenerator._update_return(ctx.value)  # Update the return value in the mock generator
                # Create StopAsyncIteration with value attribute
                exc = StopAsyncIteration()
                exc.value = ctx.value
                raise exc
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
            raise StopAsyncIterationWithValue(e.value)
        except StopAsyncIteration:
            self.logger.debug(f"asend received StopAsyncIteration without value")
            raise
        except RuntimeError as re:
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIterationWithValue):
                self.logger.debug(f"asend unwrapped StopAsyncIterationWithValue, return value: {ctx.value}")
                raise StopAsyncIterationWithValue(ctx.value)
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

class _MockGeneratorWithFrame:
    """
    Mock class that mimics Python's internal generator structure for async generators.
    This allows accessing the return value through g.agenerator.gi_frame.f_locals.get('__return__')
    """
    def __init__(self, generator):
        self.generator = generator
        self._return_value = None
        # Create a frame with locals to store return value
        self.gi_frame = _MockFrame({'__return__': None})
    
    def _update_return(self, value):
        """Update the return value in the frame locals"""
        self._return_value = value
        self.gi_frame.f_locals['__return__'] = value

class _MockFrame:
    """
    Mock class that mimics Python's frame object with locals.
    """
    def __init__(self, locals_dict):
        self.f_locals = locals_dict