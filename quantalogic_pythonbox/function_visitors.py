import ast
import asyncio
import inspect
from typing import Any, Dict, List
import logging
import traceback
from quantalogic_pythonbox.exceptions import WrappedException, ReturnException

logger = logging.getLogger(__name__)

def contains_yield(node):
    logger.debug("Checking for yields in body of node of type {}".format(type(node).__name__))
    for stmt in node.body:
        logger.debug("Statement type: {}".format(type(stmt).__name__))
        for n in ast.walk(stmt):
            if isinstance(n, (ast.Yield, ast.YieldFrom)):
                logger.debug("Yield found in statement of type {}".format(type(stmt).__name__))
                return True
    logger.debug('No yield found in body')  
    return False


async def visit_FunctionDef(interpreter, node: ast.FunctionDef, wrap_exceptions: bool = True) -> None:
    from .function_utils import Function
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    logger.debug("kwonly_params defined in visit_FunctionDef: {}, node line {}".format(kwonly_params, node.lineno if hasattr(node, 'lineno') else 'unknown'))
    logger.debug("kwonly_params after definition: {}, type: {}, node line {}".format(kwonly_params, type(kwonly_params), node.lineno if hasattr(node, 'lineno') else 'unknown'))
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:] if num_pos_defaults else [], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
    logger.debug("Debug: Before kw_defaults assignment - kwonly_params: {}, type: {}, node line {}".format(kwonly_params, type(kwonly_params), node.lineno if hasattr(node, 'lineno') else 'unknown'))
    kw_defaults = dict(zip(kwonly_params, kw_defaults_values))
    kw_defaults = {k: v for k, v in kw_defaults.items() if v is not None}

    func = Function(node, closure, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults)
    decorated_func = func
    for decorator in reversed(node.decorator_list):
        dec = await interpreter.visit(decorator, wrap_exceptions=True)
        if asyncio.iscoroutine(dec):
            dec = await dec
        if dec in (staticmethod, classmethod, property):
            decorated_func = dec(func)
        else:
            from .execution_utils import execute_function
            decorated_func = await execute_function(dec, [decorated_func], {})
    interpreter.set_variable(node.name, decorated_func)
    if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node)):
        interpreter.env_stack[0]['logger'].debug(f"Function {node.name} is a generator")
    else:
        interpreter.env_stack[0]['logger'].debug(f"Function {node.name} is not a generator")


async def visit_AsyncFunctionDef(interpreter, node: ast.AsyncFunctionDef, wrap_exceptions: bool = True) -> Any:
    try:
        logger.debug("Visiting AsyncFunctionDef for node at line {}".format(node.lineno if hasattr(node, 'lineno') else 'unknown'))

        # Extract parameter information with proper handling of optional posonlyargs
        if hasattr(node.args, 'posonlyargs') and node.args.posonlyargs:
            posonly_arg_names = [p.arg for p in node.args.posonlyargs]
        else:
            posonly_arg_names = []
        pos_kw_params = [arg.arg for arg in node.args.args if arg.arg not in posonly_arg_names]
        vararg_name = node.args.vararg.arg if node.args.vararg else None
        kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
        kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
        pos_defaults = {param.arg: await interpreter.visit(default, wrap_exceptions=True) for param, default in zip(reversed(node.args.args), reversed(node.args.defaults)) if default is not None}
        kw_defaults = {param.arg: await interpreter.visit(default, wrap_exceptions=True) if default else None for param, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is not None}

        # Check for yield or yield from statements to determine ONLY if it's an async generator
        logger.debug("Statements in {}: {}".format(node.name, [stmt.__class__.__name__ for stmt in ast.walk(node)]))
        has_yield = contains_yield(node)
        if has_yield:
            logger.debug("Detected async generator function: {}".format(node.name))
            logger.debug("Creating AsyncGeneratorFunction for {}, generator_context active: {}".format(node.name, interpreter.generator_context.get('active', False)))
            from .async_generator import AsyncGeneratorFunction
            func = AsyncGeneratorFunction(node, interpreter.env_stack, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults)
        else:
            logger.debug("No yield detected, creating standard AsyncFunction: {}".format(node.name))
            logger.debug("Creating standard AsyncFunction for {}".format(node.name))
            from .async_function import AsyncFunction
            func = AsyncFunction(node, interpreter.env_stack, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults)
        
        for decorator in reversed(node.decorator_list):
            dec = await interpreter.visit(decorator, wrap_exceptions=wrap_exceptions)
            if asyncio.iscoroutine(dec):
                dec = await dec
            from .execution_utils import execute_function
            func = await execute_function(dec, [func], {})
        interpreter.set_variable(node.name, func)
    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.error("Exception in visit_AsyncFunctionDef at node line {}: {}, type: {}, traceback: {}, env_stack depths: {}".format(node.lineno if hasattr(node, 'lineno') else 'unknown', str(e), type(e).__name__, full_traceback, [len(env) for env in interpreter.env_stack]))
        if wrap_exceptions:
            raise WrappedException("Error during AsyncFunctionDef visit: {}".format(str(e)), e, node.lineno, node.col_offset, 'async def {}(...) at line {}'.format(node.name, node.lineno))
        else:
            raise


async def visit_Call(interpreter, node: ast.Call, is_await_context: bool = False, wrap_exceptions: bool = True) -> Any:
    from .function_utils import Function, AsyncFunction, AsyncGeneratorFunction
    from .execution_utils import create_class_instance
    func = await interpreter.visit(node.func, wrap_exceptions=wrap_exceptions)
    # Handle bare super() calls to bind __class__ context
    if isinstance(node.func, ast.Name) and node.func.id == 'super' and not node.args and not node.keywords:
        # zero-arg super: bind to local 'self'
        instance = None
        for frame in reversed(interpreter.env_stack):
            if 'self' in frame:
                instance = frame['self']
                break
        if instance is None:
            # fallback to default super
            func = super
        else:
            func = super(type(instance), instance)
        return func
    interpreter.env_stack[0]['logger'].debug("Debug: node.func type in visit_Call: {}".format(type(node.func).__name__))
    interpreter.env_stack[0]['logger'].debug("Calling function: {}{}".format(func.__name__ if hasattr(func, '__name__') else str(func), ' (async)' if asyncio.iscoroutinefunction(func) else ''))
    
    if isinstance(node.func, ast.Attribute) and node.func.attr == 'throw':
        interpreter.env_stack[0]['logger'].debug("Debug: Attempting to call throw method on object")
    
    evaluated_args: List[Any] = []
    for arg in node.args:
        arg_value = await interpreter.visit(arg, wrap_exceptions=wrap_exceptions)
        if isinstance(arg, ast.Starred):
            evaluated_args.extend(arg_value)
        else:
            evaluated_args.append(arg_value)

    kwargs: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            unpacked_kwargs = await interpreter.visit(kw.value, wrap_exceptions=wrap_exceptions)
            if not isinstance(unpacked_kwargs, dict):
                raise TypeError("** argument must be a mapping, not {}".format(type(unpacked_kwargs).__name__))
            kwargs.update(unpacked_kwargs)
        else:
            kwargs[kw.arg] = await interpreter.visit(kw.value, wrap_exceptions=wrap_exceptions)

    if func and hasattr(func, '__name__') and func.__name__ == 'next':
        iterator = evaluated_args[0]
        try:
            if inspect.isasyncgen(iterator):
                logging.debug("Debug: Iterator is async generator, calling __anext__")
                result = await iterator.__anext__()
            else:
                logging.debug("Debug: Calling __next__ on sync iterator")
                result = iterator.__next__()
            return result
        except StopIteration as stop_e:
            interpreter.env_stack[0]['logger'].debug("Debug: Caught StopIteration in 'next' call with value: {}, async context: {}".format(
                getattr(stop_e, 'value', stop_e.args[0] if stop_e.args else 'No value'), interpreter.sync_mode))
            if len(evaluated_args) > 1:
                return evaluated_args[1]  # Return default if provided
            else:
                orig_val = getattr(stop_e, 'value', None)
                if orig_val is None and stop_e.args:
                    orig_val = stop_e.args[0]
                raise StopIteration(orig_val)
        except StopAsyncIteration as stop_async_e:
            interpreter.env_stack[0]['logger'].debug("Debug: Caught StopAsyncIteration in 'next' call with value: {}, async context: {}".format(stop_async_e.value if hasattr(stop_async_e, 'value') else 'No value', interpreter.sync_mode))
            if len(evaluated_args) > 1:
                return evaluated_args[1]  # Return default if provided
            else:
                raise stop_async_e
        except RuntimeError as gen_err:
            # Handle PEP-479: generator raised StopIteration => convert back to StopIteration with original value
            interpreter.env_stack[0]['logger'].debug(f"Debug: Caught RuntimeError in 'next' call: {str(gen_err)}. Checking for StopIteration cause.")
            cause = gen_err.__cause__ if hasattr(gen_err, '__cause__') else None
            if cause and isinstance(cause, StopIteration):
                orig_val = getattr(cause, 'value', None)
                if orig_val is None and cause.args:
                    orig_val = cause.args[0]
                raise StopIteration(orig_val)
            # Not a PEP-479 StopIteration, re-raise
            raise
        except AttributeError:
            raise TypeError("'{}' object is not an iterator".format(type(iterator).__name__))
    # Handle type() calls explicitly
    if func is type:
        if len(evaluated_args) == 1:
            obj = evaluated_args[0]
            return obj.__class__  # Use __class__ for single-argument type calls
        elif len(evaluated_args) == 3:
            return type(*evaluated_args)
        raise TypeError("type() takes 1 or 3 arguments, got {}".format(len(evaluated_args)))

    # Special handling for method calls on super() objects
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call) \
       and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == 'super':
        from .execution_utils import execute_function
        call_node = node.func.value
        # bind zero-arg super to local 'self'
        if not call_node.args and not call_node.keywords:
            instance = None
            for frame in reversed(interpreter.env_stack):
                if 'self' in frame:
                    instance = frame['self']
                    break
            if instance is None:
                # fallback to dynamic super evaluation
                super_call = await interpreter.visit(call_node, wrap_exceptions=wrap_exceptions)
            else:
                super_call = super(type(instance), instance)
        else:
            super_call = await interpreter.visit(call_node, wrap_exceptions=wrap_exceptions)
        method = getattr(super_call, node.func.attr)
        return await execute_function(method, evaluated_args, kwargs)

    if func is list and len(evaluated_args) == 1 and hasattr(evaluated_args[0], '__aiter__'):
        # Convert async iterable to list
        return [val async for val in evaluated_args[0]]

    # Special handling for heapq.nlargest with async key functions
    if hasattr(func, '__module__') and func.__module__ == 'heapq' and func.__name__ == 'nlargest':
        n = evaluated_args[0]
        iterable = evaluated_args[1]
        keyfunc = kwargs.get('key', None)
        reverse = kwargs.get('reverse', False)
        if not callable(keyfunc):
            return func(*evaluated_args, **kwargs)
        new_pairs = []
        for item in iterable:
            if keyfunc:
                k = keyfunc(item)
                if asyncio.iscoroutine(k):
                    k = await k
            else:
                k = item
            new_pairs.append((k, item))
        new_pairs.sort(key=lambda pair: pair[0], reverse=reverse)
        return [elem for _, elem in new_pairs][:n]

    if isinstance(node.func, ast.Attribute) and node.func.attr == "sort" and hasattr(func, "__self__") and isinstance(func.__self__, list):
        lst = func.__self__
        keyfunc = kwargs.pop("key", None)
        reverse = kwargs.pop("reverse", False)
        if not callable(keyfunc):
            return func(*evaluated_args, **kwargs)
        new_pairs = []
        for item in lst:
            key_val = keyfunc(item)
            if asyncio.iscoroutine(key_val):
                key_val = await key_val
            new_pairs.append((key_val, item))
        new_pairs.sort(key=lambda pair: pair[0], reverse=reverse)
        lst[:] = [item for _, item in new_pairs]
        return None

    if func is sorted:
        # Always use async_sorted to handle possible async key functions
        from .async_builtins import async_sorted
        result = await async_sorted(
            evaluated_args[0] if evaluated_args else [],
            key=kwargs.get('key'),
            reverse=kwargs.get('reverse', False)
        )
        return result

    if func in (range, dict, set, tuple, frozenset):
        return func(*evaluated_args, **kwargs)

    if inspect.isclass(func):
        instance = await create_class_instance(func, *evaluated_args, **kwargs)
        return instance

    if isinstance(func, (staticmethod, classmethod, property)):
        if isinstance(func, property):
            result = func.fget(*evaluated_args, **kwargs)
        else:
            result = func(*evaluated_args, **kwargs)
    elif asyncio.iscoroutinefunction(func) or isinstance(func, (AsyncFunction, AsyncGeneratorFunction)):
        result = func(*evaluated_args, **kwargs)
        if not is_await_context:
            result = await result
    elif isinstance(func, Function):
        if func.node.name == "__init__":
            await func(*evaluated_args, **kwargs)
            return None
        if func.is_generator:
            result = await func(*evaluated_args, **kwargs)
        else:
            result = await func(*evaluated_args, **kwargs)
    else:
        result = func(*evaluated_args, **kwargs)
        if asyncio.iscoroutine(result) and not is_await_context:
            result = await result
    return result


async def visit_Await(interpreter, node: ast.Await, wrap_exceptions: bool = True) -> Any:
    from .async_generator import AsyncGenerator  # Import here to avoid circular dependency
    coro = await interpreter.visit(node.value, is_await_context=True, wrap_exceptions=wrap_exceptions)
    interpreter.env_stack[0]['logger'].debug("Attempting to await object of type '{}' in visit_Await".format(type(coro).__name__))
    if not asyncio.iscoroutine(coro):
        raise TypeError("attempt to await a non-coroutine object of type '{}'".format(type(coro).__name__))
    try:
        awaited_result = await asyncio.wait_for(coro, timeout=60)
        return awaited_result
    except StopAsyncIteration:
        # Propagate StopAsyncIteration directly so user code can catch it
        raise
    except Exception as e:
        # Avoid wrapping exceptions for async generator methods
        if wrap_exceptions and not (hasattr(coro, '__self__') and isinstance(coro.__self__, AsyncGenerator)):
            raise RuntimeError("Error awaiting expression: {}".format(str(e))) from e
        else:
            raise


async def visit_Lambda(interpreter, node: ast.Lambda, wrap_exceptions: bool = True) -> Any:
    from .function_utils import LambdaFunction
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    logger.debug("Debug: node.args.kwonlyargs type: {}, value: {}, node line {}".format(type(node.args.kwonlyargs), node.args.kwonlyargs, node.lineno if hasattr(node, 'lineno') else 'unknown'))
    logger.debug("kwonly_params defined in visit_Lambda: {}, node line {}".format(kwonly_params, node.lineno if hasattr(node, 'lineno') else 'unknown'))
    logger.debug("kwonly_params after definition: {}, type: {}, node line {}".format(kwonly_params, type(kwonly_params), node.lineno if hasattr(node, 'lineno') else 'unknown'))
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:] if num_pos_defaults else [], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
    logger.debug("Debug: Before kw_defaults assignment - kwonly_params: {}, type: {}, node line {}".format(kwonly_params, type(kwonly_params), node.lineno if hasattr(node, 'lineno') else 'unknown'))
    kw_defaults = dict(zip(kwonly_params, kw_defaults_values))
    kw_defaults = {k: v for k, v in kw_defaults.items() if v is not None}

    lambda_func = LambdaFunction(
        node, closure, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults
    )
    
    def lambda_wrapper(*args, **kwargs):
        coro = lambda_func(*args, **kwargs)
        if not asyncio.iscoroutine(coro):
            return coro
            
        try:
            if hasattr(interpreter, 'loop') and interpreter.loop:
                try:
                    return interpreter.loop.run_until_complete(coro)
                except RuntimeError:
                    pass
                    
            loop = asyncio.get_event_loop()
            try:
                return loop.run_until_complete(coro)
            except RuntimeError:
                pass
                
            return coro
        except Exception:
            return coro
    
    return lambda_wrapper