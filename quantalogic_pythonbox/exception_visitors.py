import ast
from typing import Any


async def visit_Try(interpreter, node: ast.Try, wrap_exceptions: bool = True) -> Any:
    from .exceptions import WrappedException, ReturnException
    result = None
    exception_raised = None
    interpreter.env_stack[0]['logger'].debug("Entering visit_Try")
    try:
        for stmt in node.body:
            # Execute body without wrapping to allow exceptions like StopAsyncIteration to propagate
            result = await interpreter.visit(stmt, wrap_exceptions=False)
    except ReturnException as ret:
        interpreter.env_stack[0]['logger'].debug("Propagating ReturnException")
        # Propagate return after executing finalbody
        raise ret
    except Exception as e:
        interpreter.env_stack[0]['logger'].debug(f"Caught exception: {type(e).__name__}")
        # Unwrap the exception
        original_e = e
        while isinstance(original_e, WrappedException):
            original_e = original_e.original_exception
        interpreter.env_stack[0]['logger'].debug(f"Unwrapped exception: {type(original_e).__name__}")
        exception_raised = original_e
        # Track current exception to support bare raise
        interpreter.current_exception = original_e
        for handler in node.handlers:
            exc_type = await interpreter._resolve_exception_type(handler.type) if handler.type else Exception
            interpreter.env_stack[0]['logger'].debug(f"Checking handler for exception type: {exc_type.__name__}")
            if isinstance(original_e, exc_type):
                interpreter.env_stack[0]['logger'].debug("Handler matched")
                if handler.name:
                    interpreter.set_variable(handler.name, original_e)
                try:
                    for stmt in handler.body:
                        result = await interpreter.visit(stmt, wrap_exceptions=True)
                except ReturnException as ret:
                    interpreter.env_stack[0]['logger'].debug("Propagating ReturnException from handler")
                    raise ret
                # Clear current_exception after handling
                interpreter.current_exception = None
                break
        else:
            interpreter.env_stack[0]['logger'].debug("No handler matched")
            if wrap_exceptions and not node.finalbody:
                lineno = getattr(node, "lineno", 1)
                col = getattr(node, "col_offset", 0)
                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                raise WrappedException(f"Uncaught exception: {str(original_e)}", original_e, lineno, col, context_line)
            elif not node.finalbody:
                raise original_e
    else:
        interpreter.env_stack[0]['logger'].debug("Try block completed, executing orelse if present")
        # Execute else clause if no exception was raised
        for stmt in node.orelse:
            result = await interpreter.visit(stmt, wrap_exceptions=False)
    finally:
        if node.finalbody:
            interpreter.env_stack[0]['logger'].debug("Executing finalbody")
            for stmt in node.finalbody:
                # Execute finalbody for side effects, do not override result
                await interpreter.visit(stmt, wrap_exceptions=False)
    if exception_raised and not node.handlers and not node.finalbody:
        interpreter.env_stack[0]['logger'].debug("Reraising unhandled exception")
        raise exception_raised
    interpreter.env_stack[0]['logger'].debug("Exiting visit_Try")
    return result


async def visit_TryStar(interpreter, node: ast.TryStar, wrap_exceptions: bool = True) -> Any:
    from .exceptions import WrappedException
    result = None
    interpreter.env_stack[0]['logger'].debug("Entering visit_TryStar")
    try:
        for stmt in node.body:
            result = await interpreter.visit(stmt, wrap_exceptions=True)
    except Exception as e:
        interpreter.env_stack[0]['logger'].debug(f"Caught exception in TryStar: {type(e).__name__}")
        for handler in node.handlers:
            exc_type = await interpreter._resolve_exception_type(handler.type) if handler.type else Exception
            interpreter.env_stack[0]['logger'].debug(f"Checking TryStar handler for exception type: {exc_type.__name__}")
            if isinstance(e, exc_type):
                interpreter.env_stack[0]['logger'].debug("TryStar handler matched")
                if handler.name:
                    interpreter.set_variable(handler.name, e)
                for stmt in handler.body:
                    result = await interpreter.visit(stmt, wrap_exceptions=True)
                break
        else:
            interpreter.env_stack[0]['logger'].debug("No TryStar handler matched")
            if wrap_exceptions:
                lineno = getattr(node, "lineno", 1)
                col = getattr(node, "col_offset", 0)
                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                raise WrappedException(f"Uncaught exception: {str(e)}", e, lineno, col, context_line)
            else:
                raise
    finally:
        if node.finalbody:
            interpreter.env_stack[0]['logger'].debug("Executing TryStar finalbody")
            for stmt in node.finalbody:
                result = await interpreter.visit(stmt, wrap_exceptions=True)
    interpreter.env_stack[0]['logger'].debug("Exiting visit_TryStar")
    return result


async def visit_Raise(interpreter, node: ast.Raise, wrap_exceptions: bool = True) -> None:
    interpreter.env_stack[0]['logger'].debug("Visiting visit_Raise")
    if node.exc:
        exc = await interpreter.visit(node.exc, wrap_exceptions=True)
        if node.cause:
            cause = await interpreter.visit(node.cause, wrap_exceptions=True)
            exc.__cause__ = cause
        raise exc
    elif node.cause:
        raise ValueError("Cannot raise from a cause without an exception")
    else:
        if interpreter.current_exception:
            raise interpreter.current_exception
        else:
            raise RuntimeError("No active exception to reraise")