import ast
import asyncio
from typing import Any


async def visit_With(interpreter, node: ast.With, wrap_exceptions: bool = True) -> Any:
    from .exceptions import WrappedException
    result = None
    contexts = []
    for item in node.items:
        ctx = await interpreter.visit(item.context_expr, wrap_exceptions=True)
        contexts.append(ctx)
        if asyncio.iscoroutinefunction(getattr(ctx, '__aenter__', None)):
            var = await ctx.__aenter__()
        else:
            var = ctx.__enter__()
            if asyncio.iscoroutine(var):
                var = await var
        if item.optional_vars:
            await interpreter.assign(item.optional_vars, var)
    try:
        for stmt in node.body:
            result = await interpreter.visit(stmt, wrap_exceptions=True)
    except Exception as e:
        for ctx in reversed(contexts):
            if asyncio.iscoroutinefunction(getattr(ctx, '__aexit__', None)):
                await ctx.__aexit__(type(e), e, e.__traceback__)
            else:
                res = ctx.__exit__(type(e), e, e.__traceback__)
                if asyncio.iscoroutine(res):
                    await res
        raise
    else:
        for ctx in reversed(contexts):
            if asyncio.iscoroutinefunction(getattr(ctx, '__aexit__', None)):
                await ctx.__aexit__(None, None, None)
            else:
                res = ctx.__exit__(None, None, None)
                if asyncio.iscoroutine(res):
                    await res
    return result


async def visit_AsyncWith(interpreter, node: ast.AsyncWith, wrap_exceptions: bool = True) -> Any:
    from .exceptions import WrappedException
    result = None
    for item in node.items:
        ctx = await interpreter.visit(item.context_expr, wrap_exceptions=True)
        if hasattr(ctx, '__aenter__'):
            var = await ctx.__aenter__()
        elif hasattr(ctx, '__enter__'):
            var = ctx.__enter__()
        else:
            lineno = getattr(node, "lineno", 1)
            col = getattr(node, "col_offset", 0)
            context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
            raise WrappedException(
                f"Object {ctx} does not support async context management", TypeError(), lineno, col, context_line
            )
        if item.optional_vars:
            await interpreter.assign(item.optional_vars, var)
    try:
        for stmt in node.body:
            result = await interpreter.visit(stmt, wrap_exceptions=True)
    except Exception as e:
        if hasattr(ctx, '__aexit__'):
            await ctx.__aexit__(type(e), e, e.__traceback__)
        elif hasattr(ctx, '__exit__'):
            ctx.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        if hasattr(ctx, '__aexit__'):
            await ctx.__aexit__(None, None, None)
        elif hasattr(ctx, '__exit__'):
            ctx.__exit__(None, None, None)
    return result