import ast
import asyncio  # Added import for coroutine checking
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
    prop = getattr(type(value), attr, None)
    if isinstance(prop, property) and isinstance(prop.fget, Function):
        return await prop.fget(value)
    return getattr(value, attr)

async def visit_Subscript(self: ASTInterpreter, node: ast.Subscript, wrap_exceptions: bool = True) -> Any:
    try:

        
        # Normal subscript handling
        value: Any = await self.visit(node.value, wrap_exceptions=wrap_exceptions)
        
        # Check if this is a slice operation
        if isinstance(node.slice, ast.Slice):
            # Process the slice parts
            lower = await self.visit(node.slice.lower, wrap_exceptions=wrap_exceptions) if node.slice.lower else None
            upper = await self.visit(node.slice.upper, wrap_exceptions=wrap_exceptions) if node.slice.upper else None
            step = await self.visit(node.slice.step, wrap_exceptions=wrap_exceptions) if node.slice.step else None
            
            # Use a native Python slice object for built-in types
            slice_val = slice(lower, upper, step)
        else:
            # Regular subscript
            slice_val = await self.visit(node.slice, wrap_exceptions=wrap_exceptions)
        
        # Make sure value has __getitem__ method before accessing with []
        if hasattr(value, '__getitem__'):
            try:
                result = value[slice_val]
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                if wrap_exceptions:
                    from quantalogic_pythonbox.exceptions import WrappedException
                    lineno = getattr(node, 'lineno', 0)
                    col = getattr(node, 'col_offset', 0)
                    if isinstance(slice_val, slice):
                        context_line = f"return {node.value.id}[{slice_val.start}:{slice_val.stop}:{slice_val.step}]"
                    else:
                        context_line = f"return {node.value.id}[{slice_val}]"
                    raise WrappedException(str(e), e, lineno, col, context_line) from e
                raise
        else:
            raise TypeError(f"Object of type {type(value).__name__} does not support indexing")
    except Exception as e:
        if wrap_exceptions:
            from quantalogic_pythonbox.exceptions import WrappedException
            # Extract line and column information from the node
            lineno = getattr(node, 'lineno', 0)
            col = getattr(node, 'col_offset', 0)
            context_line = f"{node.value.__class__.__name__}[slice]"  # Generic slice operation representation
            raise WrappedException(str(e), e, lineno, col, context_line) from e
        raise

async def visit_Slice(self: ASTInterpreter, node: ast.Slice, wrap_exceptions: bool = True) -> Any:
    lower: Any = await self.visit(node.lower, wrap_exceptions=wrap_exceptions) if node.lower else None
    upper: Any = await self.visit(node.upper, wrap_exceptions=wrap_exceptions) if node.upper else None
    step: Any = await self.visit(node.step, wrap_exceptions=wrap_exceptions) if node.step else None
    # Use CustomSlice instead of built-in slice for consistent representation
    from quantalogic_pythonbox.slice_utils import CustomSlice
    return CustomSlice(lower, upper, step)

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