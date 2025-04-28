import ast
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

async def visit_ListComp(interpreter, node: ast.ListComp, wrap_exceptions: bool = True) -> List[Any]:
    result = []
    base_frame = interpreter.env_stack[-1].copy()
    interpreter.env_stack.append(base_frame)

    if len(node.generators) == 1:
        comp = node.generators[0]
        iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
        logger.debug(f"Retrieved iterable for single generator: {iterable if iterable else 'None'}, type: {type(iterable) if iterable else 'NoneType'}")
        logger.debug(f"Retrieved iterable of type {type(iterable)} for single generator, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
        if hasattr(iterable, '__aiter__'):
            result = []
            async for item in iterable:
                try:
                    # assign comprehension target for async iterable
                    await interpreter.assign(comp.target, item)
                except TypeError:
                    # Skip items that don't match destructuring target
                    continue
                # apply any filter clauses
                if comp.ifs:
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    if not all(conditions):
                        continue
                # evaluate element and append
                value = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                result.append(value)
            # pop comprehension frame and return result early
            interpreter.env_stack.pop()
            return result
        else:
            try:
                logger.debug("Starting for loop for single generator")
                for item in iterable:
                    logger.debug(f"Processing item {item} from iterable, item type: {type(item)}")
                    new_frame = interpreter.env_stack[-1].copy()
                    interpreter.env_stack.append(new_frame)
                    await interpreter.assign(comp.target, item)
                    logger.debug(f"After assignment for single generator, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                    logger.debug(f"Assigned item {item} to target for single generator, item type: {type(item)}")
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs] if comp.ifs else [True]
                    logger.debug(f"Conditions for single generator: {conditions}, all true? {all(conditions)}")
                    if all(conditions):
                        logger.debug(f"Evaluating elt for single generator, scope keys: {list(interpreter.env_stack[-1].keys())}")
                        value = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                        logger.debug(f"Element to append in list comp: {value}, type: {type(value)}")
                        result.append(value)
                        logger.debug(f"Appended value: {value}, type: {type(value)}, result_id: {id(result)}")
                    interpreter.env_stack.pop()
            except TypeError as e:
                logger.debug(f"TypeError in for loop for iterable {iterable}: {e}")
                lineno = getattr(node, "lineno", 1)
                col = getattr(node, "col_offset", 0)
                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                from .exceptions import WrappedException
                raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e
    else:
        async def rec(gen_idx: int):
            num_generators = len(node.generators)
            logger.debug(f"Rec method entered for gen_idx: {gen_idx}, num_generators: {num_generators}, scope keys: {list(interpreter.env_stack[-1].keys())}")
            logger.debug(f"Entering rec with gen_idx: {gen_idx}, current scope keys: {list(interpreter.env_stack[-1].keys())}")
            if gen_idx == len(node.generators):
                logger.debug(f"About to append value from node.elt of type {type(node.elt).__name__} for gen_idx {gen_idx}")
                logger.debug(f"Evaluating elt for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}")
                value = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                logger.debug(f"Element to append in list comp: {value}, type: {type(value)}")
                result.append(value)
                logger.debug(f"Appended value: {value}, type: {type(value)}, result_id: {id(result)}")
            else:
                comp = node.generators[gen_idx]
                iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                logger.debug(f"Retrieved iterable for gen_idx {gen_idx}: {iterable if iterable else 'None'}, type: {type(iterable) if iterable else 'NoneType'}")
                logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
                if hasattr(iterable, '__aiter__'):
                    logger.debug(f"Starting async for loop with iterable of type {type(iterable)} and gen_idx: {gen_idx}")
                    try:
                        async for item in iterable:
                            logger.debug(f"Starting async for loop for gen_idx {gen_idx}, item: {item if item else 'None'}, type: {type(item) if item else 'NoneType'}")
                            logger.debug(f"Starting iteration over item {item} from async iterable in generator {gen_idx}, item type: {type(item)}")
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            try:
                                await interpreter.assign(comp.target, item)
                            except TypeError as e:
                                logger.debug(f"Skipping async comprehension iteration for item {item}: {e}")
                                interpreter.env_stack.pop()
                                continue
                            logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                            logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs] if comp.ifs else [True]
                            logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                            if all(conditions):
                                logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                                try:
                                    await rec(gen_idx + 1)
                                except Exception as e:
                                    logger.error(f"Error in recursive call for gen_idx {gen_idx}: {str(e)}")
                                    raise
                                logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
                            interpreter.env_stack.pop()
                    except Exception as e:
                        logger.error(f"Exception in async for loop for gen_idx {gen_idx}: {str(e)}")
                        raise
                else:
                    try:
                        logger.debug(f"Starting for loop with iterable of type {type(iterable)} and gen_idx: {gen_idx}")
                        for item in iterable:
                            logger.debug(f"Processing item {item} from iterable in generator {gen_idx}, item type: {type(item)}")
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            await interpreter.assign(comp.target, item)
                            logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                            logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs] if comp.ifs else [True]
                            logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                            if all(conditions):
                                logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                                try:
                                    await rec(gen_idx + 1)
                                except Exception as e:
                                    logger.error(f"Error in recursive call for gen_idx {gen_idx}: {str(e)}")
                                    raise
                                logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
                            interpreter.env_stack.pop()
                    except TypeError as e:
                        logger.debug(f"TypeError in for loop for iterable {iterable}: {e}")
                        lineno = getattr(node, "lineno", 1)
                        col = getattr(node, "col_offset", 0)
                        context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                        from .exceptions import WrappedException
                        raise WrappedException(f"Object {iterable} is not iterable", e, lineno, col, context_line) from e

        await rec(0)
    logger.debug(f"List comprehension result after execution: {result}, length: {len(result)}")
    logger.debug(f"Final list comp result: {result}")
    interpreter.env_stack.pop()
    return result


async def visit_DictComp(interpreter, node: ast.DictComp, wrap_exceptions: bool = True) -> Dict[Any, Any]:
    result = {}
    base_frame = interpreter.env_stack[-1].copy()
    interpreter.env_stack.append(base_frame)

    async def rec(gen_idx: int):
        num_generators = len(node.generators)
        logger.debug(f"Rec method entered for gen_idx: {gen_idx}, num_generators: {num_generators}, scope keys: {list(interpreter.env_stack[-1].keys())}")
        logger.debug(f"Entering rec with gen_idx: {gen_idx}, current scope keys: {list(interpreter.env_stack[-1].keys())}")
        if gen_idx == len(node.generators):
            key = await interpreter.visit(node.key, wrap_exceptions=True)
            val = await interpreter.visit(node.value, wrap_exceptions=True)
            result[key] = val
        else:
            comp = node.generators[gen_idx]
            iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
            logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
            if hasattr(iterable, '__aiter__'):
                async for item in iterable:
                    logger.debug(f"Starting async for loop for gen_idx {gen_idx}, item: {item if item else 'None'}, type: {type(item) if item else 'NoneType'}")
                    new_frame = interpreter.env_stack[-1].copy()
                    interpreter.env_stack.append(new_frame)
                    try:
                        await interpreter.assign(comp.target, item)
                    except TypeError as e:
                        logger.debug(f"Skipping async comprehension iteration for item {item}: {e}")
                        interpreter.env_stack.pop()
                        continue
                    logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                    logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                    if all(conditions):
                        logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                        await rec(gen_idx + 1)
                        logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
                    interpreter.env_stack.pop()
            else:
                try:
                    for item in iterable:
                        logger.debug(f"Processing item {item} from iterable in generator {gen_idx}, item type: {type(item)}")
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        await interpreter.assign(comp.target, item)
                        logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                        logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                        if all(conditions):
                            logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                            await rec(gen_idx + 1)
                            logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
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
        num_generators = len(node.generators)
        logger.debug(f"Rec method entered for gen_idx: {gen_idx}, num_generators: {num_generators}, scope keys: {list(interpreter.env_stack[-1].keys())}")
        logger.debug(f"Entering rec with gen_idx: {gen_idx}, current scope keys: {list(interpreter.env_stack[-1].keys())}")
        if gen_idx == len(node.generators):
            result.add(await interpreter.visit(node.elt, wrap_exceptions=True))
        else:
            comp = node.generators[gen_idx]
            iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
            logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
            if hasattr(iterable, '__aiter__'):
                async for item in iterable:
                    logger.debug(f"Starting async for loop for gen_idx {gen_idx}, item: {item if item else 'None'}, type: {type(item) if item else 'NoneType'}")
                    new_frame = interpreter.env_stack[-1].copy()
                    interpreter.env_stack.append(new_frame)
                    try:
                        await interpreter.assign(comp.target, item)
                    except TypeError as e:
                        logger.debug(f"Skipping async comprehension iteration for item {item}: {e}")
                        interpreter.env_stack.pop()
                        continue
                    logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                    logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                    if all(conditions):
                        logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                        await rec(gen_idx + 1)
                        logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
                    interpreter.env_stack.pop()
            else:
                try:
                    for item in iterable:
                        logger.debug(f"Processing item {item} from iterable in generator {gen_idx}, item type: {type(item)}")
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        await interpreter.assign(comp.target, item)
                        logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                        logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                        if all(conditions):
                            logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                            await rec(gen_idx + 1)
                            logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
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
    from .utils import has_await
    if not has_await(node):
        result = []
        base_frame = interpreter.env_stack[-1].copy()
        interpreter.env_stack.append(base_frame)

        async def rec(gen_idx: int):
            num_generators = len(node.generators)
            logger.debug(f"Rec method entered for gen_idx: {gen_idx}, num_generators: {num_generators}, scope keys: {list(interpreter.env_stack[-1].keys())}")
            logger.debug(f"Entering rec with gen_idx: {gen_idx}, current scope keys: {list(interpreter.env_stack[-1].keys())}")
            if gen_idx == len(node.generators):
                logger.debug(f"About to append value from node.elt of type {type(node.elt).__name__} for gen_idx {gen_idx}")
                logger.debug(f"Evaluating elt for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}")
                element = await interpreter.visit(node.elt, wrap_exceptions=True)
                logger.debug(f"Debug: About to append element {element} with current scope keys: {list(interpreter.env_stack[-1].keys())}")
                result.append(element)
            else:
                comp = node.generators[gen_idx]
                iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
                if hasattr(iterable, '__aiter__'):
                    async for item in iterable:
                        logger.debug(f"Starting async for loop for gen_idx {gen_idx}, item: {item if item else 'None'}, type: {type(item) if item else 'NoneType'}")
                        logger.debug(f"Starting iteration over item {item} from async iterable in generator {gen_idx}, item type: {type(item)}")
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        try:
                            await interpreter.assign(comp.target, item)
                        except TypeError as e:
                            logger.debug(f"Skipping async comprehension iteration for item {item}: {e}")
                            interpreter.env_stack.pop()
                            continue
                        logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                        logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                        conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                        logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                        if all(conditions):
                            logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                            try:
                                await rec(gen_idx + 1)
                            except Exception as e:
                                logger.error(f"Error in recursive call for gen_idx {gen_idx}: {str(e)}")
                                raise
                            logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
                        interpreter.env_stack.pop()
                else:
                    try:
                        for item in iterable:
                            logger.debug(f"Processing item {item} from iterable in generator {gen_idx}, item type: {type(item)}")
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            await interpreter.assign(comp.target, item)
                            logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                            logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                            logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                            if all(conditions):
                                logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                                try:
                                    await rec(gen_idx + 1)
                                except Exception as e:
                                    logger.error(f"Error in recursive call for gen_idx {gen_idx}: {str(e)}")
                                    raise
                                logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
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
            logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx 0, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
            if hasattr(iterable, '__aiter__'):
                async def gen():
                    try:
                        async for item in iterable:
                            logger.debug(f"Starting async for loop for gen_idx 0, item: {item if item else 'None'}, type: {type(item) if item else 'NoneType'}")
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            try:
                                await interpreter.assign(comp.target, item)
                            except TypeError as e:
                                logger.debug(f"Skipping async comprehension iteration for item {item}: {e}")
                                interpreter.env_stack.pop()
                                continue
                            logger.debug(f"After assignment for gen_idx 0, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                            logger.debug(f"Assigned item {item} to target for gen_idx 0, item type: {type(item)}")
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                            logger.debug(f"Conditions for gen_idx 0: {conditions}, all true? {all(conditions)}")
                            if all(conditions):
                                logger.debug(f"Before calling rec with gen_idx 1 for item {item}")
                                yield await interpreter.visit(node.elt, wrap_exceptions=True)
                                logger.debug(f"After calling rec with gen_idx 1 for item {item}")
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
                            logger.debug(f"Processing item {item} from iterable in generator 0, item type: {type(item)}")
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            await interpreter.assign(comp.target, item)
                            logger.debug(f"After assignment for gen_idx 0, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                            logger.debug(f"Assigned item {item} to target for gen_idx 0, item type: {type(item)}")
                            conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                            logger.debug(f"Conditions for gen_idx 0: {conditions}, all true? {all(conditions)}")
                            if all(conditions):
                                logger.debug(f"Before calling rec with gen_idx 1 for item {item}")
                                yield await interpreter.visit(node.elt, wrap_exceptions=True)
                                logger.debug(f"After calling rec with gen_idx 1 for item {item}")
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
                    num_generators = len(node.generators)
                    logger.debug(f"Rec method entered for gen_idx: {gen_idx}, num_generators: {num_generators}, scope keys: {list(interpreter.env_stack[-1].keys())}")
                    logger.debug(f"Entering rec with gen_idx: {gen_idx}, current scope keys: {list(interpreter.env_stack[-1].keys())}")
                    if gen_idx == len(node.generators):
                        yield await interpreter.visit(node.elt, wrap_exceptions=True)
                    else:
                        comp = node.generators[gen_idx]
                        iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                        logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
                        if hasattr(iterable, '__aiter__'):
                            try:
                                async for item in iterable:
                                    logger.debug(f"Starting async for loop for gen_idx {gen_idx}, item: {item if item else 'None'}, type: {type(item) if item else 'NoneType'}")
                                    new_frame = interpreter.env_stack[-1].copy()
                                    interpreter.env_stack.append(new_frame)
                                    try:
                                        await interpreter.assign(comp.target, item)
                                    except TypeError as e:
                                        logger.debug(f"Skipping async comprehension iteration for item {item}: {e}")
                                        interpreter.env_stack.pop()
                                        continue
                                    logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                                    logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                                    logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                                    if all(conditions):
                                        logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                                        async for val in rec(gen_idx + 1):
                                            yield val
                                        logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
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
                                    logger.debug(f"Processing item {item} from iterable in generator {gen_idx}, item type: {type(item)}")
                                    new_frame = interpreter.env_stack[-1].copy()
                                    interpreter.env_stack.append(new_frame)
                                    await interpreter.assign(comp.target, item)
                                    logger.debug(f"After assignment for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}, target {comp.target.id if isinstance(comp.target, ast.Name) else 'complex target'} present: {comp.target.id in interpreter.env_stack[-1] if isinstance(comp.target, ast.Name) else 'N/A' }")
                                    logger.debug(f"Assigned item {item} to target for gen_idx {gen_idx}, item type: {type(item)}")
                                    conditions = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                                    logger.debug(f"Conditions for gen_idx {gen_idx}: {conditions}, all true? {all(conditions)}")
                                    if all(conditions):
                                        logger.debug(f"Before calling rec with gen_idx {gen_idx + 1} for item {item}")
                                        async for val in rec(gen_idx + 1):
                                            yield val
                                        logger.debug(f"After calling rec with gen_idx {gen_idx + 1} for item {item}")
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