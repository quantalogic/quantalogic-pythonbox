# quantalogic_pythonbox/async_generator.py
"""
Async generator function handling for the PythonBox interpreter.
"""

import ast
import logging
from typing import Any, Dict, List, Optional
import asyncio
import inspect as _inspect_module

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
        self.logger.debug(f"Entering gen method for {self.node.name}")
        # Instrument: Log AST statements in generator body
        for idx, stmt in enumerate(self.node.body):
            self.logger.debug(f"Gen stmt #{idx}: {stmt.__class__.__name__}")
        try:
            for stmt in self.node.body:
                # Handle explicit return statements in async generator
                if isinstance(stmt, ast.Return):
                    ret_val = await self.interpreter.visit(stmt.value, wrap_exceptions=False) if stmt.value else None
                    raise ReturnException(ret_val)
                # Handle async for loops
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
                # Handle assignment with yield (e.g., received = yield x)
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Yield):
                    # Evaluate the yield expression argument
                    yield_val = await self.interpreter.visit(stmt.value.value, wrap_exceptions=False)
                    self.logger.debug(f"Yielding value from assign: {yield_val}")
                    # Wrap yield to catch exceptions thrown via athrow
                    try:
                        sent = yield yield_val
                    except Exception as e:
                        self.logger.debug(f"Exception at yield, delegating to AST handler: {e}")
                        handler_res = await self.interpreter.visit(stmt, wrap_exceptions=True)
                        if handler_res is not None:
                            yield handler_res
                        continue
                    # Assign the sent value to the target variable
                    await self.interpreter.assign(stmt.targets[0], sent)
                # Handle bare yield expressions (e.g., yield x)
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                    yield_val = await self.interpreter.visit(stmt.value.value, wrap_exceptions=False)
                    self.logger.debug(f"Yielding value from expr: {yield_val}")
                    yield yield_val
                # Handle Try blocks with exception handling and yields
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
                        # Finish async generator with return value
                        raise StopAsyncIteration(getattr(re, 'value', None))
                    except GeneratorExit:
                        # Handle GeneratorExit: run finally block before closing
                        for fstmt in stmt.finalbody:
                            if isinstance(fstmt, ast.Expr) and isinstance(fstmt.value, ast.Yield):
                                yield_val = await self.interpreter.visit(fstmt.value.value, wrap_exceptions=False)
                                yield yield_val
                            else:
                                result = await self.interpreter.visit(fstmt, wrap_exceptions=False)
                                if result is not None:
                                    yield result
                        # Propagate GeneratorExit to signal closure
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
                    # Execute finally block if present
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
                        # Properly end async generator with return value
                        self.logger.debug(f"Gen caught ReturnException with return value: {ret.value}")
                        raise StopAsyncIteration(ret.value)
                    if result is not None:
                        yield result
        except ReturnException as re:
            # Finish async generator with return value
            raise StopAsyncIteration(getattr(re, 'value', None))
        # No explicit return: exhaustion without a return statement
        # End of generator
        raise StopAsyncIteration()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.logger.debug(f"__call__ invoked for {self.node.name} with args: {args}, kwargs: {kwargs}")
        logger.debug(f"Starting AsyncGeneratorFunction {self.node.name} with args: {args}, kwargs: {kwargs}")
        
        # Use instance variables for argument binding
        local_frame: Dict[str, Any] = {}
        
        # Bind positional arguments using self.pos_kw_params
        for name, value in zip(self.pos_kw_params, args):
            local_frame[name] = value
        
        # Apply defaults for parameters with default values
        for param, default in self.pos_defaults.items():
            if param not in local_frame:
                local_frame[param] = default
        
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

        # Set up generator_context for proper yield and send handling in async generators
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
            if value is None:
                self.logger.warning(f'Warning: Yielding None value in __anext__ for generator {self.gen_name}')
            self.logger.debug(f'Value received and yielding: {value}')
            return value
        except StopAsyncIteration as e:
            ret_val = e.args[0] if e.args else None
            self.logger.debug(f'Raising StopAsyncIteration with return value: {ret_val}')
            raise e
        except RuntimeError as re:
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIteration):
                ret_val = ctx.args[0] if ctx.args else None
                self.logger.debug(f'Unwrapped StopAsyncIteration with return value: {ret_val}')
                raise StopAsyncIteration(ret_val)
            raise
        except Exception as exc:
            self.logger.error(f'Exception in __anext__ for generator {self.gen_name}: {type(exc).__name__}: {str(exc)}')
            raise

    async def asend(self, value):
        # Mark generator as active
        self.interpreter.generator_context['active'] = True
        self.logger.debug(f"asend called for generator {self.gen_name}, sending value: {value}")
        try:
            result = await self.gen_coroutine.asend(value)
            self.logger.debug(f"asend yielded value: {result}")
            return result
        except StopAsyncIteration as e:
            # Generator completed; return its return value
            ret_val = getattr(e, 'value', e.args[0] if e.args else None)
            self.logger.debug(f"asend received StopAsyncIteration for generator {self.gen_name}, return value: {ret_val}")
            return ret_val
        except RuntimeError as re:
            # Unwrap StopAsyncIteration wrapped by Python runtime
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIteration):
                ret_val = getattr(ctx, 'value', ctx.args[0] if ctx.args else None)
                self.logger.debug(f"asend unwrapped StopAsyncIteration, return value: {ret_val}")
                return ret_val
            raise
        except Exception as exc:
            self.logger.error(f"Exception in asend for generator {self.gen_name}: {str(exc)}")
            raise
        finally:
            self.interpreter.generator_context['active'] = False

    async def athrow(self, exc):
        # Mark generator as active
        self.interpreter.generator_context['active'] = True
        self.logger.debug(f"athrow called for generator {self.gen_name}, throwing exception: {exc}")
        try:
            result = await self.gen_coroutine.athrow(exc)
            self.logger.debug(f"athrow yielded value: {result}")
            return result
        except StopAsyncIteration as e:
            # Generator completed; return its return value
            ret_val = e.args[0] if e.args else None
            self.logger.debug(f"athrow received StopAsyncIteration for generator {self.gen_name}, return value: {ret_val}")
            return ret_val
        except Exception as err:
            self.logger.debug(f"athrow exception {err}, re-raising for caller to handle")
            raise
        finally:
            self.interpreter.generator_context['active'] = False

    async def aclose(self):
        """Close the async generator, running any finally blocks."""
        self.logger.debug(f"aclose called for generator {self.gen_name}")
        try:
            # Throw GeneratorExit into generator to trigger finally
            await self.gen_coroutine.athrow(GeneratorExit())
        except (GeneratorExit, StopAsyncIteration):
            # Generator has closed normally
            return
        except Exception as err:
            # Ignore other exceptions on close
            self.logger.debug(f"Exception during aclose: {err}")
            return

    def __aiter__(self):
        return self