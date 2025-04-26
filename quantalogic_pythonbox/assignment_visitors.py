import ast
from typing import Any


async def visit_Assign(interpreter, node: ast.Assign, wrap_exceptions: bool = True) -> None:
    value: Any = await interpreter.visit(node.value, wrap_exceptions=wrap_exceptions)
    for target in node.targets:
        if isinstance(target, ast.Subscript):
            obj = await interpreter.visit(target.value, wrap_exceptions=wrap_exceptions)
            key = await interpreter.visit(target.slice, wrap_exceptions=wrap_exceptions)
            from .slice_utils import CustomSlice
            if isinstance(key, CustomSlice):
                key = slice(key.start, key.stop, key.step)
            obj[key] = value
        else:
            await interpreter.assign(target, value)


async def visit_AugAssign(interpreter, node: ast.AugAssign, wrap_exceptions: bool = True) -> Any:
    if isinstance(node.target, ast.Name):
        current_val: Any = interpreter.get_variable(node.target.id)
    else:
        current_val: Any = await interpreter.visit(node.target, wrap_exceptions=wrap_exceptions)
    right_val: Any = await interpreter.visit(node.value, wrap_exceptions=wrap_exceptions)
    op = node.op
    if isinstance(op, ast.Add):
        result: Any = current_val + right_val
    elif isinstance(op, ast.Sub):
        result = current_val - right_val
    elif isinstance(op, ast.Mult):
        result = current_val * right_val
    elif isinstance(op, ast.Div):
        result = current_val / right_val
    elif isinstance(op, ast.FloorDiv):
        result = current_val // right_val
    elif isinstance(op, ast.Mod):
        result = current_val % right_val
    elif isinstance(op, ast.Pow):
        result = current_val**right_val
    elif isinstance(op, ast.BitAnd):
        result = current_val & right_val
    elif isinstance(op, ast.BitOr):
        result = current_val | right_val
    elif isinstance(op, ast.BitXor):
        result = current_val ^ right_val
    elif isinstance(op, ast.LShift):
        result = current_val << right_val
    elif isinstance(op, ast.RShift):
        result = current_val >> right_val
    else:
        raise Exception("Unsupported augmented operator: " + str(op))
    await interpreter.assign(node.target, result)
    return result


async def visit_AnnAssign(interpreter, node: ast.AnnAssign, wrap_exceptions: bool = True) -> None:
    value = await interpreter.visit(node.value, wrap_exceptions=wrap_exceptions) if node.value else None
    if not interpreter.ignore_typing:
        annotation = await interpreter.visit(node.annotation, wrap_exceptions=True)
        if isinstance(node.target, ast.Name):
            interpreter.type_hints[node.target.id] = annotation
    if value is not None or node.simple:
        await interpreter.assign(node.target, value)


async def visit_NamedExpr(interpreter, node: ast.NamedExpr, wrap_exceptions: bool = True) -> Any:
    value = await interpreter.visit(node.value, wrap_exceptions=wrap_exceptions)
    await interpreter.assign(node.target, value)
    return value