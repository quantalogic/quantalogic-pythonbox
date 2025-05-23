import ast
import inspect
from typing import Any, Dict, List, Tuple

from .interpreter_core import ASTInterpreter
from .function_utils import Function

async def visit_Constant(self: ASTInterpreter, node: ast.Constant, wrap_exceptions: bool = True) -> Any:
    return node.value

async def visit_Name(self: ASTInterpreter, node: ast.Name, wrap_exceptions: bool = True) -> Any:
    if isinstance(node.ctx, ast.Load):
        return self.get_variable(node.id)
    elif isinstance(node.ctx, ast.Store):
        return node.id
    else:
        raise Exception("Unsupported context for Name")

async def visit_List(self: ASTInterpreter, node: ast.List, wrap_exceptions: bool = True) -> List[Any]:
    return [await self.visit(elt, wrap_exceptions=wrap_exceptions) for elt in node.elts]

async def visit_Tuple(self: ASTInterpreter, node: ast.Tuple, wrap_exceptions: bool = True) -> Tuple[Any, ...]:
    elements = [await self.visit(elt, wrap_exceptions=wrap_exceptions) for elt in node.elts]
    return tuple(elements)

async def visit_Dict(self: ASTInterpreter, node: ast.Dict, wrap_exceptions: bool = True) -> Dict[Any, Any]:
    return {
        await self.visit(k, wrap_exceptions=wrap_exceptions): await self.visit(v, wrap_exceptions=wrap_exceptions)
        for k, v in zip(node.keys, node.values)
    }

async def visit_Set(self: ASTInterpreter, node: ast.Set, wrap_exceptions: bool = True) -> set:
    elements = [await self.visit(elt, wrap_exceptions=wrap_exceptions) for elt in node.elts]
    return set(elements)

async def visit_Attribute(self: ASTInterpreter, node: ast.Attribute, wrap_exceptions: bool = True) -> Any:
    value = await self.visit(node.value, wrap_exceptions=wrap_exceptions)
    attr = node.attr
    self.env_stack[0]['logger'].debug(f"Accessing attribute: {attr} on object of type {type(value).__name__}")
    prop = getattr(type(value), attr, None)
    if isinstance(prop, property) and isinstance(prop.fget, Function):
        return await prop.fget(value)
    return getattr(value, attr)

async def visit_Subscript(self: ASTInterpreter, node: ast.Subscript, wrap_exceptions: bool = True) -> Any:
    slice_val = await self.visit(node.slice, wrap_exceptions=wrap_exceptions)
    value = await self.visit(node.value, wrap_exceptions=wrap_exceptions)
    try:
        if hasattr(value, '__getitem__'):
            self.env_stack[0]["logger"].debug(f"Calling __getitem__ on {type(value).__name__} with key {slice_val}")
            method = value.__getitem__
            if inspect.iscoroutinefunction(method):
                result = await method(slice_val)
            else:
                result = method(slice_val)
            self.env_stack[0]["logger"].debug(f"__getitem__ result: {result}, type: {type(result)}")
            return result
        else:
            raise TypeError(f"Object of type {type(value).__name__} does not support indexing")
    except Exception as e:
        self.env_stack[0]["logger"].error(f"Exception in __getitem__ call: {str(e)}")
        if wrap_exceptions:
            from quantalogic_pythonbox.exceptions import WrappedException
            lineno = getattr(node, 'lineno', 0)
            col = getattr(node, 'col_offset', 0)
            context_line = "subscript operation"
            raise WrappedException(str(e), e, lineno, col, context_line) from e
        raise

async def visit_Slice(self: ASTInterpreter, node: ast.Slice, wrap_exceptions: bool = True) -> Any:
    lower: Any = await self.visit(node.lower, wrap_exceptions=wrap_exceptions) if node.lower else None
    upper: Any = await self.visit(node.upper, wrap_exceptions=wrap_exceptions) if node.upper else None
    step: Any = await self.visit(node.step, wrap_exceptions=wrap_exceptions) if node.step else None
    return slice(lower, upper, step)

async def visit_Index(self: ASTInterpreter, node: ast.Index, wrap_exceptions: bool = True) -> Any:
    return await self.visit(node.value, wrap_exceptions=wrap_exceptions)

async def visit_Starred(self: ASTInterpreter, node: ast.Starred, wrap_exceptions: bool = True) -> Any:
    value = await self.visit(node.value, wrap_exceptions=wrap_exceptions)
    if not isinstance(value, (list, tuple, set)):
        raise TypeError(f"Cannot unpack non-iterable object of type {type(value).__name__}")
    return value

async def visit_JoinedStr(self: ASTInterpreter, node: ast.JoinedStr, wrap_exceptions: bool = True) -> str:
    parts = []
    for value in node.values:
        val = await self.visit(value, wrap_exceptions=wrap_exceptions)
        if isinstance(value, ast.FormattedValue):
            parts.append(str(val))
        else:
            parts.append(val)
    return "".join(parts)

async def visit_FormattedValue(self: ASTInterpreter, node: ast.FormattedValue, wrap_exceptions: bool = True) -> Any:
    return await self.visit(node.value, wrap_exceptions=wrap_exceptions)