import ast
import asyncio
from typing import Any

from .exceptions import ReturnException
from .interpreter_core import ASTInterpreter

async def visit_With(self: ASTInterpreter, node: ast.With, wrap_exceptions: bool = True) -> Any:
    result = None
    contexts = []
    for item in node.items:
        ctx = await self.visit(item.context_expr, wrap_exceptions=wrap_exceptions)
        val = ctx.__enter__()
        # If __enter__ returns a coroutine, await it
        if asyncio.iscoroutine(val):
            val = await val
        contexts.append((ctx, val))
        if item.optional_vars:
            await self.assign(item.optional_vars, val)
    try:
        for stmt in node.body:
            result = await self.visit(stmt, wrap_exceptions=wrap_exceptions)
    except ReturnException as ret:
        for ctx, _ in reversed(contexts):
            exit_result = ctx.__exit__(None, None, None)
            if asyncio.iscoroutine(exit_result):
                await exit_result
        raise ret
    except Exception as e:
        exc_type, exc_value, tb = type(e), e, e.__traceback__
        suppressed = False
        for ctx, _ in reversed(contexts):
            exit_result = ctx.__exit__(exc_type, exc_value, tb)
            if asyncio.iscoroutine(exit_result):
                exit_result = await exit_result
            if exit_result:
                suppressed = True
                break
        if not suppressed:
            raise
    else:
        for ctx, _ in reversed(contexts):
            exit_result = ctx.__exit__(None, None, None)
            if asyncio.iscoroutine(exit_result):
                await exit_result
    return result

async def visit_AsyncWith(self: ASTInterpreter, node: ast.AsyncWith, wrap_exceptions: bool = True) -> Any:
    result = None
    contexts = []
    for item in node.items:
        ctx = await self.visit(item.context_expr, wrap_exceptions=wrap_exceptions)
        val = await ctx.__aenter__()
        contexts.append((ctx, val))
        if item.optional_vars:
            await self.assign(item.optional_vars, val)
    try:
        for stmt in node.body:
            result = await self.visit(stmt, wrap_exceptions=wrap_exceptions)
    except ReturnException as ret:
        for ctx, _ in reversed(contexts):
            await ctx.__aexit__(None, None, None)
        raise ret
    except Exception as e:
        exc_type, exc_value, tb = type(e), e, e.__traceback__
        suppressed = False
        for ctx, _ in reversed(contexts):
            exit_result = await ctx.__aexit__(exc_type, exc_value, tb)
            if exit_result:
                suppressed = True
                break
        if not suppressed:
            raise
    else:
        for ctx, _ in reversed(contexts):
            await ctx.__aexit__(None, None, None)
    return result