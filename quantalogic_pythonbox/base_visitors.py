import ast
from typing import Any


async def visit_Module(interpreter, node: ast.Module, wrap_exceptions: bool = True) -> Any:
    last_value = None
    for stmt in node.body:
        last_value = await interpreter.visit(stmt, wrap_exceptions=True)
    return last_value


async def visit_Expr(interpreter, node: ast.Expr, wrap_exceptions: bool = True) -> Any:
    return await interpreter.visit(node.value, wrap_exceptions=wrap_exceptions)


async def visit_Pass(interpreter, node: ast.Pass, wrap_exceptions: bool = True) -> None:
    return None


async def visit_TypeIgnore(interpreter, node: ast.TypeIgnore, wrap_exceptions: bool = True) -> None:
    pass