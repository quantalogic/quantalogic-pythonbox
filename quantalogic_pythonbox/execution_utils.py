import ast
import asyncio
import threading
from typing import Any, Dict, List, Optional

from .exceptions import WrappedException, ReturnException


async def execute_function(func: Any, args: List[Any], kwargs: Dict[str, Any]):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    elif callable(func):
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result
    raise TypeError(f"Object {func} is not callable")


async def create_class_instance(cls: type, *args, **kwargs):
    instance = cls.__new__(cls, *args, **kwargs)
    if isinstance(instance, cls) and hasattr(cls, '__init__'):
        init_method = cls.__init__
        await execute_function(init_method, [instance] + list(args), kwargs)
    return instance


async def resolve_exception_type(interpreter, node: Optional[ast.AST]) -> Any:
    if node is None:
        return Exception
    if isinstance(node, ast.Name):
        exc_type = interpreter.get_variable(node.id)
        if exc_type in (Exception, ZeroDivisionError, ValueError, TypeError):
            return exc_type
        return exc_type
    if isinstance(node, ast.Call):
        return await interpreter.visit(node, wrap_exceptions=True)
    return None


def sync_call(func, instance, *args, **kwargs):
    """Synchronously execute an async function.
    
    This method is specifically designed for special methods like __getitem__ that
    may be called in contexts where awaiting is not possible (like slicing operations).
    It creates a new event loop in a separate thread to avoid the 'cannot run event loop 
    while another is running' error.
    
    Args:
        func: The function object to execute
        instance: The instance to bind the function to
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    result_container = []
    error_container = []
    
    def thread_runner():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                coro = func(instance, *args, **kwargs)
                result = loop.run_until_complete(coro)
                result_container.append(result)
            finally:
                loop.close()
        except Exception as e:
            error_container.append(e)
    
    thread = threading.Thread(target=thread_runner)
    thread.start()
    thread.join()
    
    if error_container:
        raise error_container[0]
    if result_container:
        return result_container[0]
    return None


def run_sync_stmt(interpreter, node: ast.AST) -> Any:
    """Execute a single AST statement synchronously.
    
    This is specifically designed for special methods like __getitem__
    that need to execute synchronously but are used in contexts where
    async/await might not be available.
    
    Args:
        interpreter: The ASTInterpreter instance
        node: The AST node to execute
        
    Returns:
        The result of executing the node
    """
    if isinstance(node, ast.Return):
        if node.value:
            value_result = run_sync_expr(interpreter, node.value) if node.value else None
            raise ReturnException(value_result)
        else:
            raise ReturnException(None)
    elif isinstance(node, ast.Expr):
        return run_sync_expr(interpreter, node.value)
    elif isinstance(node, ast.If):
        test_result = run_sync_expr(interpreter, node.test)
        if test_result:
            for stmt in node.body:
                result = run_sync_stmt(interpreter, stmt)
            return result
        else:
            for stmt in node.orelse:
                result = run_sync_stmt(interpreter, stmt)
            return result
    
    raise RuntimeError(f"Cannot synchronously execute node type: {node.__class__.__name__}")


def run_sync_expr(interpreter, node: ast.AST) -> Any:
    """Execute a single AST expression synchronously.
    
    Args:
        interpreter: The ASTInterpreter instance
        node: The AST expression node to execute
        
    Returns:
        The result of evaluating the expression
    """
    if isinstance(node, ast.Name):
        return interpreter.get_variable(node.id)
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = run_sync_expr(interpreter, node.left)
        right = run_sync_expr(interpreter, node.right)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
    
    elif isinstance(node, ast.Call):
        func = run_sync_expr(interpreter, node.func)
        args = [run_sync_expr(interpreter, arg) for arg in node.args]
        kwargs = {kw.arg: run_sync_expr(interpreter, kw.value) for kw in node.keywords}
        
        if callable(func):
            return func(*args, **kwargs)
        return None
    
    elif isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                expr_value = run_sync_expr(interpreter, value.value)
                if value.conversion != -1:
                    if value.conversion == 's':
                        expr_value = str(expr_value)
                    elif value.conversion == 'r':
                        expr_value = repr(expr_value)
                    elif value.conversion == 'a':
                        expr_value = ascii(expr_value)
                
                if value.format_spec is not None:
                    format_spec = run_sync_expr(interpreter, value.format_spec)
                    expr_value = format(expr_value, format_spec)
                
                parts.append(str(expr_value))
        return ''.join(parts)
    
    elif isinstance(node, ast.FormattedValue):
        expr_value = run_sync_expr(interpreter, node.value)
        if node.conversion != -1:
            if node.conversion == 's':
                expr_value = str(expr_value)
            elif node.conversion == 'r':
                expr_value = repr(expr_value)
            elif node.conversion == 'a':
                expr_value = ascii(expr_value)
        return expr_value
    
    elif isinstance(node, ast.Attribute):
        value = run_sync_expr(interpreter, node.value)
        attr = node.attr
        
        if hasattr(value, attr):
            return getattr(value, attr)
        return None
    
    elif isinstance(node, ast.Subscript):
        try:
            target = run_sync_expr(interpreter, node.value)
            
            result = None
            
            if isinstance(node.slice, ast.Slice):
                start = run_sync_expr(interpreter, node.slice.lower) if node.slice.lower else None
                stop = run_sync_expr(interpreter, node.slice.upper) if node.slice.upper else None
                step = run_sync_expr(interpreter, node.slice.step) if node.slice.step else None
                s = slice(start, stop, step)
                
                if isinstance(s, slice):
                    try:
                        logger = interpreter.env_stack[0]["logger"]
                        logger.debug(f"Processing slice object: {s}")
                        if hasattr(target, '__getitem__'):
                            result = target.__getitem__(s)
                            logger.debug(f"Slice operation result: {result}")
                            return result
                        else:
                            raise TypeError(f"Object of type {type(target).__name__} does not support indexing")
                    except Exception as e:
                        logger.debug(f"Slice operation failed: {e}")
                        raise
            else:
                key = run_sync_expr(interpreter, node.slice)
                
                if hasattr(target, '__getitem__'):
                    interpreter.env_stack[0]["logger"].debug(f"Calling __getitem__ on {target} with key {key}")
                    if callable(target.__getitem__):
                        method = target.__getitem__
                        if not hasattr(method, '__self__'):
                            method = method.__get__(target, type(target))
                    result = target.__getitem__(key)
                    interpreter.env_stack[0]["logger"].debug(f"__getitem__ returned: {result}")
                    return result
                else:
                    raise TypeError(f"Object of type {type(target).__name__} does not support indexing")
            
            return result
        except Exception as e:
            lineno = getattr(node, 'lineno', 0)
            col = getattr(node, 'col_offset', 0)
            context_line = "slice operation"
            raise WrappedException(str(e), e, lineno, col, context_line) from e
    
    raise RuntimeError(f"Cannot synchronously evaluate expression type: {node.__class__.__name__}")


class EventLoopManager:
    def __init__(self):
        self.loop = asyncio.get_event_loop()

    async def run_task(self, coro, timeout=None):
        return await asyncio.wait_for(coro, timeout=timeout)