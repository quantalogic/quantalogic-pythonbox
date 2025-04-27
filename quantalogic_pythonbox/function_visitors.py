import ast
import asyncio
import inspect
from typing import Any, Dict, List


async def contains_yield(node):
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))


async def visit_FunctionDef(interpreter, node: ast.FunctionDef, wrap_exceptions: bool = True) -> None:
    from .function_utils import Function
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
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


async def visit_AsyncFunctionDef(interpreter, node: ast.AsyncFunctionDef, wrap_exceptions: bool = True) -> None:
    from .function_utils import AsyncFunction, AsyncGeneratorFunction
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
    kw_defaults = dict(zip(kwonly_params, kw_defaults_values))
    kw_defaults = {k: v for k, v in kw_defaults.items() if v is not None}

    if await contains_yield(node):
        func = AsyncGeneratorFunction(node, closure, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults)
    else:
        func = AsyncFunction(node, closure, interpreter, pos_kw_params, vararg_name, kwonly_params, kwarg_name, pos_defaults, kw_defaults)

    for decorator in reversed(node.decorator_list):
        dec = await interpreter.visit(decorator, wrap_exceptions=True)
        if asyncio.iscoroutine(dec):
            dec = await dec
        from .execution_utils import execute_function
        func = await execute_function(dec, [func], {})
    interpreter.set_variable(node.name, func)


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
    coro = await interpreter.visit(node.value, is_await_context=True, wrap_exceptions=wrap_exceptions)
    try:
        awaited_result = await asyncio.wait_for(coro, timeout=60)
        return awaited_result
    except Exception as e:
        if wrap_exceptions:
            raise RuntimeError(f"Error awaiting expression: {str(e)}") from e
        else:
            raise


async def visit_Lambda(interpreter, node: ast.Lambda, wrap_exceptions: bool = True) -> Any:
    from .function_utils import LambdaFunction
    closure: List[Dict[str, Any]] = interpreter.env_stack[:]
    pos_kw_params = [arg.arg for arg in node.args.args]
    vararg_name = node.args.vararg.arg if node.args.vararg else None
    kwonly_params = [arg.arg for arg in node.args.kwonlyargs]
    kwarg_name = node.args.kwarg.arg if node.args.kwarg else None
    pos_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) for default in node.args.defaults]
    num_pos_defaults = len(pos_defaults_values)
    pos_defaults = dict(zip(pos_kw_params[-num_pos_defaults:], pos_defaults_values)) if num_pos_defaults else {}
    kw_defaults_values = [await interpreter.visit(default, wrap_exceptions=True) if default else None for default in node.args.kw_defaults]
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