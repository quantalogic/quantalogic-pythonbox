import ast
import asyncio
import inspect
from typing import Any, Dict, List
import logging
import traceback
from quantalogic_pythonbox.exceptions import WrappedException

logger = logging.getLogger(__name__)

def contains_yield(node):
    logger.debug(f"Checking for yields in body of node of type {type(node).__name__}")
    for stmt in node.body:
        logger.debug(f"Statement type: {type(stmt).__name__}")
        for n in ast.walk(stmt):
            if isinstance(n, (ast.Yield, ast.YieldFrom)):
                logger.debug(f"Yield found in statement of type {type(stmt).__name__}")
                return True
    logger.debug('No yield found in body')  
    return False


async def visit_FunctionDef(interpreter, node: ast.FunctionDef, wrap_exceptions: bool = True) -> None:
    from .function_utils import Function
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    logger.debug(f"kwonly_params defined in visit_FunctionDef: {kwonly_params}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
    logger.debug(f"kwonly_params after definition: {kwonly_params}, type: {type(kwonly_params)}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:] if num_pos_defaults else [], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
    logger.debug(f"Debug: Before kw_defaults assignment - kwonly_params: {kwonly_params}, type: {type(kwonly_params)}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
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


async def visit_AsyncFunctionDef(interpreter, node: ast.AsyncFunctionDef, wrap_exceptions: bool = True) -> Any:
    try:
        logger.debug(f"Visiting AsyncFunctionDef for node at line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")

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
        logger.debug(f"Statements in {node.name}: {[stmt.__class__.__name__ for stmt in ast.walk(node)]}")
        has_yield = contains_yield(node)
        if has_yield:
            logger.debug(f"Detected async generator function: {node.name}")
            logger.debug(f"Creating AsyncGeneratorFunction for {node.name}, generator_context active: {interpreter.generator_context.get('active', False)}")
            from .async_generator import AsyncGeneratorFunction
            func = AsyncGeneratorFunction(node, interpreter.env_stack, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults)
        else:
            logger.debug(f"No yield detected, creating standard AsyncFunction: {node.name}")
            logger.debug(f"Creating standard AsyncFunction for {node.name}")
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
        logger.error(f"Exception in visit_AsyncFunctionDef at node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}: {str(e)}, type: {type(e).__name__}, traceback: {full_traceback}, env_stack depths: {[len(env) for env in interpreter.env_stack]}")
        if wrap_exceptions:
            raise WrappedException(f"Error during AsyncFunctionDef visit: {str(e)}", e, node.lineno, node.col_offset, f'async def {node.name}(...) at line {node.lineno}')
        else:
            raise


async def visit_Call(interpreter, node: ast.Call, is_await_context: bool = False, wrap_exceptions: bool = True) -> Any:
    from .function_utils import Function, AsyncFunction, AsyncGeneratorFunction
    from .execution_utils import create_class_instance, execute_function
    func = await interpreter.visit(node.func, wrap_exceptions=wrap_exceptions)
    
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
                raise TypeError(f"** argument must be a mapping, not {type(unpacked_kwargs).__name__}")
            kwargs.update(unpacked_kwargs)
        else:
            kwargs[kw.arg] = await interpreter.visit(kw.value, wrap_exceptions=wrap_exceptions)

    # Handle type() calls explicitly
    if func is type:
        if len(evaluated_args) == 1:
            obj = evaluated_args[0]
            return obj.__class__  # Use __class__ for single-argument type calls
        elif len(evaluated_args) == 3:
            return type(*evaluated_args)
        raise TypeError(f"type() takes 1 or 3 arguments, got {len(evaluated_args)}")

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

    if hasattr(func, '__name__') and func.__name__ == 'append':
        logger.debug(f"Debug: Calling append with args: {evaluated_args}, kwargs: {kwargs}")
    
    if func is str:
        if len(evaluated_args) != 1:
            raise TypeError(f"str() takes exactly one argument ({len(evaluated_args)} given)")
        arg = evaluated_args[0]
        return f"{arg}"

    if func is str and len(evaluated_args) == 1 and isinstance(evaluated_args[0], BaseException):
        exc = evaluated_args[0]
        return exc.args[0] if exc.args else str(exc)

    if func is super:
        if len(evaluated_args) == 0:
            if not (interpreter.current_class and interpreter.current_instance):
                for frame in reversed(interpreter.env_stack):
                    if 'self' in frame and '__current_method__' in frame:
                        interpreter.current_instance = frame['self']
                        interpreter.current_class = frame['self'].__class__
                        break
                if not (interpreter.current_class and interpreter.current_instance):
                    raise TypeError("super() without arguments requires a class instantiation context")
            result = super(interpreter.current_class, interpreter.current_instance)
        elif len(evaluated_args) >= 2:
            cls, obj = evaluated_args[0], evaluated_args[1]
            result = super(cls, obj)
        else:
            raise TypeError("super() requires class and instance arguments")
        return result

    # Special handling for next() on GeneratorWrapper to propagate StopIteration correctly
    if func is next and len(evaluated_args) >= 1:
        gen = evaluated_args[0]
        default = evaluated_args[1] if len(evaluated_args) > 1 else None
        from .generator_wrapper import GeneratorWrapper
        if isinstance(gen, GeneratorWrapper):
            try:
                return gen.__next__()
            except StopIteration as e:
                if hasattr(e, 'value') and e.value is not None:
                    return e.value
                return default
        return next(gen, default)

    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == 'super':
        super_call = await interpreter.visit(node.func.value, wrap_exceptions=wrap_exceptions)
        method = getattr(super_call, node.func.attr)
        return await execute_function(method, evaluated_args, kwargs)

    if func is list and len(evaluated_args) == 1 and hasattr(evaluated_args[0], '__aiter__'):
        return [val async for val in evaluated_args[0]]

    if func is sorted:
        if 'key' in kwargs and callable(kwargs['key']):
            from .async_builtins import async_sorted
            result = await async_sorted(evaluated_args[0] if evaluated_args else [], 
                                     key=kwargs.get('key'), 
                                     reverse=kwargs.get('reverse', False))
            return result
    
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

    if func in (range, list, dict, set, tuple, frozenset, sorted):
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
    interpreter.env_stack[0]['logger'].debug(f"Attempting to await object of type '{type(coro).__name__}' in visit_Await")
    if not asyncio.iscoroutine(coro):
        raise TypeError(f"attempt to await a non-coroutine object of type '{type(coro).__name__}'")
    try:
        awaited_result = await asyncio.wait_for(coro, timeout=60)
        return awaited_result
    except StopAsyncIteration:
        # Propagate StopAsyncIteration directly so user code can catch it
        raise
    except Exception as e:
        # Avoid wrapping exceptions for async generator methods
        if wrap_exceptions and not (hasattr(coro, '__self__') and isinstance(coro.__self__, AsyncGenerator)):
            raise RuntimeError(f"Error awaiting expression: {str(e)}") from e
        else:
            raise


async def visit_Lambda(interpreter, node: ast.Lambda, wrap_exceptions: bool = True) -> Any:
    from .function_utils import LambdaFunction
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    logger.debug(f"Debug: node.args.kwonlyargs type: {type(node.args.kwonlyargs)}, value: {node.args.kwonlyargs}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
    logger.debug(f"kwonly_params defined in visit_Lambda: {kwonly_params}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
    logger.debug(f"kwonly_params after definition: {kwonly_params}, type: {type(kwonly_params)}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:] if num_pos_defaults else [], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
    logger.debug(f"Debug: Before kw_defaults assignment - kwonly_params: {kwonly_params}, type: {type(kwonly_params)}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
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