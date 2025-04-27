# quantalogic_pythonbox/async_generator.py
"""
Async generator function handling for the PythonBox interpreter.
"""

import ast
import logging
from typing import Any, Dict, List, Optional
import asyncio

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
        try:
            for stmt in self.node.body:
                result = await self.interpreter.visit(stmt, wrap_exceptions=False)
                if result is not None:
                    yield result
        except ReturnException as ret:
            # Properly end async generator with return value
            self.logger.debug(f"Gen caught ReturnException with return value: {ret.value}")
            raise StopAsyncIteration(ret.value)
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
        # Mark generator as active
        self.interpreter.generator_context['active'] = True
        self.logger.debug(f"gen_coroutine type: {type(self.gen_coroutine).__name__}")
        self.logger.debug(f"__anext__ called for generator {self.gen_name}, attempting asend(None)")
        try:
            # Attempt to get next yield
            value = await self.gen_coroutine.asend(None)
            self.logger.debug(f"__anext__ yielded value: {value}")
            return value
        except StopAsyncIteration as e:
            # Normal completion with optional return value
            ret_val = e.args[0] if e.args else None
            setattr(e, 'value', ret_val)
            self.logger.debug(f"__anext__ raised StopAsyncIteration for generator {self.gen_name}, return value: {ret_val}")
            raise e
        except RuntimeError as re:
            # Unwrap StopAsyncIteration wrapped by Python runtime
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIteration):
                ret_val = ctx.args[0] if ctx.args else None
                new_exc = StopAsyncIteration(ret_val)
                setattr(new_exc, 'value', ret_val)
                self.logger.debug(f"__anext__ unwrapped StopAsyncIteration, return value: {ret_val}")
                raise new_exc
            raise
        except Exception as exc:
            self.logger.error(f"Exception in __anext__ for generator {self.gen_name}: {str(exc)}, type: {type(exc).__name__}")
            raise
        finally:
            self.interpreter.generator_context['active'] = False

    async def asend(self, value):
        # Mark generator as active
        self.interpreter.generator_context['active'] = True
        self.logger.debug(f"asend called for generator {self.gen_name}, sending value: {value}")
        try:
            result = await self.gen_coroutine.asend(value)
            self.logger.debug(f"asend yielded value: {result}")
            return result
        except StopAsyncIteration as e:
            # Completion: propagate return value
            ret_val = e.args[0] if e.args else None
            self.logger.debug(f"asend received StopAsyncIteration for generator {self.gen_name}, return value: {ret_val}")
            return ret_val
        except RuntimeError as re:
            # Unwrap StopAsyncIteration wrapped by Python runtime
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIteration):
                ret_val = ctx.args[0] if ctx.args else None
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
            # Completion: propagate return value
            ret_val = e.args[0] if e.args else None
            self.logger.debug(f"athrow received StopAsyncIteration for generator {self.gen_name}, return value: {ret_val}")
            return ret_val
        except RuntimeError as re:
            # Unwrap StopAsyncIteration wrapped by Python runtime
            ctx = re.__context__ or re.__cause__
            if isinstance(ctx, StopAsyncIteration):
                ret_val = ctx.args[0] if ctx.args else None
                self.logger.debug(f"athrow unwrapped StopAsyncIteration, return value: {ret_val}")
                return ret_val
            raise
        except Exception as err:
            self.logger.error(f"Exception in athrow for generator {self.gen_name}: {str(err)}")
            raise
        finally:
            self.interpreter.generator_context['active'] = False

    def __aiter__(self):
        return self