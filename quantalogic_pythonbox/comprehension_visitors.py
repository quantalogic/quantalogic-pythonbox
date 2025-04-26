import ast
from typing import Any, Dict, List


async def visit_ListComp(interpreter, node: ast.ListComp, wrap_exceptions: bool = True) -> List[Any]:
    result = []
    base_frame = interpreter.env_stack[-1].copy()
    interpreter.env_stack.append(base_frame)

    async def rec(gen_idx: int):
        if gen_idx == len(node.generators):
            element = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
            result.append(element)
        else:
            comp = node.generators[gen_idx]
            iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
            if hasattr(iterable, '__aiter__'):
                async for item in iterable:
                    new_frame = interpreter.env_stack[-1].copy()
                    interpreter.env_stack.append(new_frame)
                    await interpreter.assign(comp.target, item)
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    if all(conditions):
                        await rec(gen_idx + 1)
                    interpreter.env_stack.pop()
            else:
                try:
                    for item in iterable:
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        await interpreter.assign(comp.target, item)
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        if all(conditions):
                            await rec(gen_idx + 1)
                        interpreter.env_stack.pop()
                except TypeError as e:
                    lineno = getattr(node, "lineno", 1)
                    col = getattr(node, "col_offset", 0)
                    context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                    from .exceptions import WrappedException
                    raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e

    await rec(0)
    interpreter.env_stack.pop()
    return result


async def visit_DictComp(interpreter, node: ast.DictComp, wrap_exceptions: bool = True) -> Dict[Any, Any]:
    result = {}
    base_frame = interpreter.env_stack[-1].copy()
    interpreter.env_stack.append(base_frame)

    async def rec(gen_idx: int):
        if gen_idx == len(node.generators):
            key = await interpreter.visit(node.key, wrap_exceptions=True)
            val = await interpreter.visit(node.value, wrap_exceptions=True)
            result[key] = val
        else:
            comp = node.generators[gen_idx]
            iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
            if hasattr(iterable, '__aiter__'):
                async for item in iterable:
                    new_frame = interpreter.env_stack[-1].copy()
                    interpreter.env_stack.append(new_frame)
                    await interpreter.assign(comp.target, item)
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    if all(conditions):
                        await rec(gen_idx + 1)
                    interpreter.env_stack.pop()
            else:
                try:
                    for item in iterable:
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        await interpreter.assign(comp.target, item)
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        if all(conditions):
                            await rec(gen_idx + 1)
                        interpreter.env_stack.pop()
                except TypeError as e:
                    lineno = getattr(node, "lineno", 1)
                    col = getattr(node, "col_offset", 0)
                    context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                    from .exceptions import WrappedException
                    raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e

    await rec(0)
    interpreter.env_stack.pop()
    return result


async def visit_SetComp(interpreter, node: ast.SetComp, wrap_exceptions: bool = True) -> set:
    result = set()
    base_frame = interpreter.env_stack[-1].copy()
    interpreter.env_stack.append(base_frame)

    async def rec(gen_idx: int):
        if gen_idx == len(node.generators):
            result.add(await interpreter.visit(node.elt, wrap_exceptions=True))
        else:
            comp = node.generators[gen_idx]
            iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
            if hasattr(iterable, '__aiter__'):
                async for item in iterable:
                    new_frame = interpreter.env_stack[-1].copy()
                    interpreter.env_stack.append(new_frame)
                    await interpreter.assign(comp.target, item)
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    if all(conditions):
                        await rec(gen_idx + 1)
                    interpreter.env_stack.pop()
            else:
                try:
                    for item in iterable:
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        await interpreter.assign(comp.target, item)
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        if all(conditions):
                            await rec(gen_idx + 1)
                        interpreter.env_stack.pop()
                except TypeError as e:
                    lineno = getattr(node, "lineno", 1)
                    col = getattr(node, "col_offset", 0)
                    context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                    from .exceptions import WrappedException
                    raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e

    await rec(0)
    interpreter.env_stack.pop()
    return result


async def visit_GeneratorExp(interpreter, node: ast.GeneratorExp, wrap_exceptions: bool = True) -> Any:
    from .exceptions import has_await
    if not has_await(node):
        result = []
        base_frame = interpreter.env_stack[-1].copy()
        interpreter.env_stack.append(base_frame)

        async def rec(gen_idx: int):
            if gen_idx == len(node.generators):
                element = await interpreter.visit(node.elt, wrap_exceptions=True)
                result.append(element)
            else:
                comp = node.generators[gen_idx]
                iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                if hasattr(iterable, '__aiter__'):
                    async for item in iterable:
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        await interpreter.assign(comp.target, item)
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        if all(conditions):
                            await rec(gen_idx + 1)
                        interpreter.env_stack.pop()
                else:
                    try:
                        for item in iterable:
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            await interpreter.assign(comp.target, item)
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                            if all(conditions):
                                await rec(gen_idx + 1)
                            interpreter.env_stack.pop()
                    except TypeError as e:
                        lineno = getattr(node, "lineno", 1)
                        col = getattr(node, "col_offset", 0)
                        context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                        from .exceptions import WrappedException
                        raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e

        await rec(0)
        interpreter.env_stack.pop()
        return result
    else:
        base_frame = interpreter.env_stack[-1].copy()
        interpreter.env_stack.append(base_frame)

        if len(node.generators) == 1:
            comp = node.generators[0]
            iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
            if hasattr(iterable, '__aiter__'):
                async def gen():
                    try:
                        async for item in iterable:
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            await interpreter.assign(comp.target, item)
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                            if all(conditions):
                                yield await interpreter.visit(node.elt, wrap_exceptions=True)
                            interpreter.env_stack.pop()
                    except Exception as e:
                        lineno = getattr(node, "lineno", 1)
                        col = getattr(node, "col_offset", 0)
                        context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                        from .exceptions import WrappedException
                        raise WrappedException(f"Error in async iteration: {str(e)}", e, lineno, col, context_line) from e
            else:
                async def gen():
                    try:
                        for item in iterable:
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            await interpreter.assign(comp.target, item)
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                            if all(conditions):
                                yield await interpreter.visit(node.elt, wrap_exceptions=True)
                            interpreter.env_stack.pop()
                    except TypeError as e:
                        lineno = getattr(node, "lineno", 1)
                        col = getattr(node, "col_offset", 0)
                        context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                        from .exceptions import WrappedException
                        raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e
        else:
            async def gen():
                async def rec(gen_idx: int):
                    if gen_idx == len(node.generators):
                        yield await interpreter.visit(node.elt, wrap_exceptions=True)
                    else:
                        comp = node.generators[gen_idx]
                        iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                        if hasattr(iterable, '__aiter__'):
                            try:
                                async for item in iterable:
                                    new_frame = interpreter.env_stack[-1].copy()
                                    interpreter.env_stack.append(new_frame)
                                    await interpreter.assign(comp.target, item)
                                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                                    if all(conditions):
                                        async for val in rec(gen_idx + 1):
                                            yield val
                                    interpreter.env_stack.pop()
                            except Exception as e:
                                lineno = getattr(node, "lineno", 1)
                                col = getattr(node, "col_offset", 0)
                                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                                from .exceptions import WrappedException
                                raise WrappedException(f"Error in async iteration: {str(e)}", e, lineno, col, context_line) from e
                        else:
                            try:
                                for item in iterable:
                                    new_frame = interpreter.env_stack[-1].copy()
                                    interpreter.env_stack.append(new_frame)
                                    await interpreter.assign(comp.target, item)
                                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                                    if all(conditions):
                                        async for val in rec(gen_idx + 1):
                                            yield val
                                    interpreter.env_stack.pop()
                            except TypeError as e:
                                lineno = getattr(node, "lineno", 1)
                                col = getattr(node, "col_offset", 0)
                                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                                from .exceptions import WrappedException
                                raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e

                async for val in rec(0):
                    yield val

        interpreter.env_stack.pop()
        return gen()