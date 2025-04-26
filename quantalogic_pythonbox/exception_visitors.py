import ast
from typing import Any, Optional


async def visit_Try(interpreter, node: ast.Try, wrap_exceptions: bool = True) -> Any:
    from .exceptions import WrappedException, ReturnException
    result = None
    exception_raised = None
    try:
        for stmt in node.body:
            result = await interpreter.visit(stmt, wrap_exceptions=True)
    except ReturnException:
        # Propagate return after executing finalbody
        raise
    except Exception as e:
        exception_raised = e
        for handler in node.handlers:
            exc_type = await interpreter._resolve_exception_type(handler.type) if handler.type else Exception
            if isinstance(e, exc_type) or (exc_type is Exception and not isinstance(e, (ZeroDivisionError, ValueError, TypeError))):
                if handler.name:
                    interpreter.set_variable(handler.name, e)
                for stmt in handler.body:
                    result = await interpreter.visit(stmt, wrap_exceptions=True)
                break
        else:
            if wrap_exceptions and not node.finalbody:
                lineno = getattr(node, "lineno", 1)
                col = getattr(node, "col_offset", 0)
                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                raise WrappedException(f"Uncaught exception: {str(e)}", e, lineno, col, context_line)
            elif not node.finalbody:
                raise
    else:
        # Execute else clause if no exception was raised
        for stmt in node.orelse:
            result = await interpreter.visit(stmt, wrap_exceptions=True)
    finally:
        if node.finalbody:
            for stmt in node.finalbody:
                # Execute finalbody for side effects, do not override result
                await interpreter.visit(stmt, wrap_exceptions=True)
    if exception_raised and not node.handlers and not node.finalbody:
        raise exception_raised
    return result


async def visit_TryStar(interpreter, node: ast.TryStar, wrap_exceptions: bool = True) -> Any:
    from .exceptions import WrappedException
    result = None
    try:
        for stmt in node.body:
            result = await interpreter.visit(stmt, wrap_exceptions=True)
    except Exception as e:
        for handler in node.handlers:
            exc_type = await interpreter._resolve_exception_type(handler.type) if handler.type else Exception
            if isinstance(e, exc_type):
                if handler.name:
                    interpreter.set_variable(handler.name, e)
                for stmt in handler.body:
                    result = await interpreter.visit(stmt, wrap_exceptions=True)
                break
        else:
            if wrap_exceptions:
                lineno = getattr(node, "lineno", 1)
                col = getattr(node, "col_offset", 0)
                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                raise WrappedException(f"Uncaught exception: {str(e)}", e, lineno, col, context_line)
            else:
                raise
    finally:
        if node.finalbody:
            for stmt in node.finalbody:
                result = await interpreter.visit(stmt, wrap_exceptions=True)
    return result


async def visit_Raise(interpreter, node: ast.Raise, wrap_exceptions: bool = True) -> None:
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