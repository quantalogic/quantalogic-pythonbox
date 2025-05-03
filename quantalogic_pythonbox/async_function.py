# quantalogic_pythonbox/async_function.py
"""
Async function handling for the PythonBox interpreter.
"""

import ast
import logging
from typing import Any, Dict, List, Optional

from .interpreter_core import ASTInterpreter
from .exceptions import ReturnException

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AsyncFunction:
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
        self.interpreter.env_stack[0]['logger'].debug(f"AsyncFunction __init__ for {self.node.name}, kwonly_params set to {self.kwonly_params}")

    async def __call__(self, *args: Any, _return_locals: bool = False, **kwargs: Any) -> Any:
        logger.debug(f"Entering AsyncFunction.__call__ for {self.node.name}, kwonly_params: {self.kwonly_params if hasattr(self, 'kwonly_params') else 'not set'}, args: {args}, kwargs: {kwargs}")
        logger.debug("Instance variables: posonly_params={}, pos_kw_params={}, kwonly_params={}, vararg_name={}, kwarg_name={}, pos_defaults={}, kw_defaults={}".format(self.posonly_params, self.pos_kw_params, self.kwonly_params, self.vararg_name, self.kwarg_name, self.pos_defaults, self.kw_defaults))
        logger.debug(f"Debug in __call__: received args: {args}, kwargs: {kwargs}")
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}
        
        num_posonly = len(self.posonly_params)
        num_pos_kw = len(self.pos_kw_params)
        total_pos = num_posonly + num_pos_kw
        pos_args_used = 0

        # Bind positional arguments with logging
        logger.debug(f"Binding positional arguments: args length={len(args)}, total_pos={total_pos}")
        for i, arg in enumerate(args):
            if i < num_posonly:
                local_frame[self.posonly_params[i]] = arg
                logger.debug(f"Bound posonly arg {self.posonly_params[i]} to {arg}")
                pos_args_used += 1
            elif i < total_pos:
                local_frame[self.pos_kw_params[i - num_posonly]] = arg
                logger.debug(f"Bound pos_kw arg {self.pos_kw_params[i - num_posonly]} to {arg}")
                pos_args_used += 1
            elif self.vararg_name:
                if self.vararg_name not in local_frame:
                    local_frame[self.vararg_name] = []
                local_frame[self.vararg_name].append(arg)
                logger.debug(f"Appended to vararg {self.vararg_name}: {arg}")
            else:
                raise TypeError(f"Async function '{self.node.name}' takes {total_pos} positional arguments but {len(args)} were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        # Ensure kwonly_params is a list to avoid scoping issues
        if not isinstance(self.kwonly_params, list):
            raise ValueError(f"kwonly_params should be a list, got {type(self.kwonly_params)}")
        # Bind keyword-only parameters
        logger.debug(f"Debug: kwonly_params value - {self.kwonly_params}")
        for param_name in self.kwonly_params:
            if param_name in kwargs:
                local_frame[param_name] = kwargs[param_name]
                # Remove from kwargs to avoid conflict in general keyword binding
                del kwargs[param_name]
            elif param_name in self.kw_defaults:
                local_frame[param_name] = self.kw_defaults[param_name]
            else:
                raise TypeError(f"Missing required keyword-only argument '{param_name}'")

        # Bind keyword arguments with logging
        logger.debug(f"Binding keyword arguments: kwargs={kwargs}")
        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.posonly_params:
                raise TypeError(f"Async function '{self.node.name}' got an unexpected keyword argument '{kwarg_name}' (positional-only)")
            elif kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError(f"Async function '{self.node.name}' got multiple values for argument '{kwarg_name}'")
                local_frame[kwarg_name] = kwarg_value
                logger.debug(f"Bound keyword arg {kwarg_name} to {kwarg_value}")
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
                logger.debug(f"Bound to kwarg {self.kwarg_name}: {kwarg_name} = {kwarg_value}")
            else:
                raise TypeError(f"Async function '{self.node.name}' got an unexpected keyword argument '{kwarg_name}'")

        # Apply defaults for missing optional arguments with logging
        logger.debug("Applying defaults for pos_kw_params and kwonly_params")
        for param in self.pos_kw_params:
            if param not in local_frame and param in self.pos_defaults:
                local_frame[param] = self.pos_defaults[param]
                logger.debug(f"Applied default to pos_kw_param {param}: {self.pos_defaults[param]}")

        # Check for missing required arguments with logging
        missing_args = [param for param in self.posonly_params if param not in local_frame] + [param for param in self.pos_kw_params if param not in local_frame and param not in self.pos_defaults] + [param for param in self.kwonly_params if param not in local_frame and param not in self.kw_defaults]
        if missing_args:
            logger.error(f"Missing required arguments: {missing_args}")
            missing_pos_args = [param for param in missing_args if param in self.posonly_params or param in self.pos_kw_params]
            raise TypeError(f"Async function '{self.node.name}' missing {len(missing_pos_args)} required positional argument{'s' if len(missing_pos_args) != 1 else ''}")

        # Set up the interpreter with the new environment
        new_interp = self.interpreter.spawn_from_env(new_env_stack + [local_frame])
        
        last_value = None
        try:
            logger.debug(f"Debug: local_frame before execution in {self.node.name}: {local_frame}")
            logger.debug("Beginning statement execution in {}".format(self.node.name))
            for stmt in self.node.body:
                # Propagate StopAsyncIteration out of async function for user try/except
                last_value = await new_interp.visit(stmt, wrap_exceptions=False)
            logger.debug(f"{self.node.name} completed all statements, last_value: {last_value}")
            if _return_locals:
                local_vars = {k: v for k, v in local_frame.items() if not k.startswith('__')}
                return last_value, local_vars
            return last_value
        except ReturnException as ret:
            logger.debug(f"{self.node.name} caught ReturnException with value: {ret.value}")
            if _return_locals:
                local_vars = {k: v for k, v in local_frame.items() if not k.startswith('__')}
                return ret.value, local_vars
            return ret.value
        except RuntimeError as e:
            # Handle StopIteration raised from coroutine per PEP 479
            if str(e) == "coroutine raised StopIteration":
                orig = e.__cause__ if hasattr(e, '__cause__') else None
                ret_val = getattr(orig, 'value', None)
                logger.debug(f"{self.node.name} caught coroutine StopIteration with value: {ret_val}")
                return ret_val
            raise
        except StopIteration as stop:
            # Treat StopIteration as a return from async function
            logger.debug(f"{self.node.name} caught StopIteration with value: {getattr(stop, 'value', None)}")
            ret_val = getattr(stop, 'value', None)
            if _return_locals:
                local_vars = {k: v for k, v in local_frame.items() if not k.startswith('__')}
                return ret_val, local_vars
            return ret_val
        except Exception as e:
            logger.error("Exception in {}: {}, type: {}".format(self.node.name, str(e), type(e).__name__))
            raise
        finally:
            if not _return_locals:
                new_env_stack.pop()

    def __get__(self, instance: Any, owner: Any):
        if instance is None:
            return self
        async def bound_method(*args: Any, **kwargs: Any) -> Any:
            return await self(instance, *args, **kwargs)
        return bound_method