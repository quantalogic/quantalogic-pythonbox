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
        logger.debug(f"Debug in visit_ListComp: retrieved iterable of type {type(iterable).__name__}, value: {iterable if isinstance(iterable, (int, str, list, dict)) else 'complex object'}")
        # handle async iterables (async generators) and sync iterables
        if hasattr(iterable, '__aiter__'):
            logger.debug('Start async iteration in visit_ListComp')
            async for item in iterable:
                logger.debug(f'Fetched item: {item}')
                new_frame = interpreter.env_stack[-1].copy()
                interpreter.env_stack.append(new_frame)
                try:
                    await interpreter.assign(comp.target, item)
                    if comp.ifs and not all([await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]):
                        interpreter.env_stack.pop()
                        continue
                    val = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                    result.append(val)
                    logger.debug(f'After add, result length: {len(result)}')
                except Exception as e:
                    logger.debug(f'Exception during processing: {type(e).__name__}: {str(e)}')
                    raise
                finally:
                    interpreter.env_stack.pop()
            interpreter.env_stack.pop()
            logger.debug('End of async iteration, final result length: ' + str(len(result)))
            return result
        else:
            result = []
            for item in iterable:
                new_frame = interpreter.env_stack[-1].copy()
                interpreter.env_stack.append(new_frame)
                try:
                    await interpreter.assign(comp.target, item)
                except TypeError:
                    interpreter.env_stack.pop()
                    continue
                if comp.ifs:
                    conds = [await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]
                    if not all(conds):
                        interpreter.env_stack.pop()
                        continue
                val = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                logger.debug(f"Debug in visit_ListComp: adding item {val} to result")
                result.append(val)
                interpreter.env_stack.pop()
            interpreter.env_stack.pop()
            return result
    else:
        async def rec(gen_idx: int):
            num_generators = len(node.generators)
            logger.debug(f"Rec method entered for gen_idx: {gen_idx}, num_generators: {num_generators}, scope keys: {list(interpreter.env_stack[-1].keys())}")
            logger.debug(f"Entering rec with gen_idx: {gen_idx}, current scope keys: {list(interpreter.env_stack[-1].keys())}")
            if gen_idx == len(node.generators):
                logger.debug(f"About to append value from node.elt of type {type(node.elt).__name__} for gen_idx {gen_idx}")
                logger.debug(f"Evaluating elt for gen_idx {gen_idx}, scope keys: {list(interpreter.env_stack[-1].keys())}")
                value = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                logger.debug(f'Adding final value: {value} to result, length before: {len(result)}')
                result.append(value)
                logger.debug(f'After add, result length: {len(result)}')
            else:
                comp = node.generators[gen_idx]
                iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
                if hasattr(iterable, '__aiter__'):
                    logger.debug('Start async iteration for gen_idx ' + str(gen_idx))
                    async for item in iterable:
                        logger.debug(f'Fetched item for gen_idx {gen_idx}: {item}')
                        new_frame = interpreter.env_stack[-1].copy()
                        interpreter.env_stack.append(new_frame)
                        try:
                            await interpreter.assign(comp.target, item)
                            logger.debug(f'Assigned item to target for gen_idx {gen_idx}: {comp.target.id if hasattr(comp.target, "id") else "target"} = {item}, scope keys: {list(interpreter.env_stack[-1].keys())}')
                            if comp.ifs and not all([await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]):
                                interpreter.env_stack.pop()
                                continue
                            await rec(gen_idx + 1)
                        except Exception as e:
                            logger.debug(f'Exception in rec loop gen_idx {gen_idx}: {type(e).__name__}: {str(e)}')
                            raise
                        finally:
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
                logger.debug(f"Debug in visit_GeneratorExp: adding item {element} to result")
                result.append(element)
            else:
                comp = node.generators[gen_idx]
                iterable = await interpreter.visit(comp.iter, wrap_exceptions=wrap_exceptions)
                logger.debug(f"Retrieved iterable of type {type(iterable)} for gen_idx {gen_idx}, has __aiter__: {hasattr(iterable, '__aiter__')}, has __anext__: {hasattr(iterable, '__anext__')}")
                if hasattr(iterable, '__aiter__'):
                    async for item in iterable:
                        interpreter.logger.debug(f"Debug in visit_GeneratorExp: received item {item} in async for loop")
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
                logger.debug('Start async iteration')
                async def gen():
                    try:
                        async for item in iterable:
                            logger.debug(f"Received item {item} in async for loop")
                            new_frame = interpreter.env_stack[-1].copy()
                            interpreter.env_stack.append(new_frame)
                            try:
                                await interpreter.assign(comp.target, item)
                                logger.debug(f'Debug: Verified assigned value for {comp.target.id}: {interpreter.env_stack[-1].get(comp.target.id, "not found")}')
                                logger.debug(f'Debug: comp.ifs length: {len(comp.ifs)}')
                                logger.debug(f'Assigned item to {comp.target.id if hasattr(comp.target, "id") else "target"}: {item}, scope keys: {list(interpreter.env_stack[-1].keys())}')
                                if comp.ifs and not all([await interpreter.visit(if_clause, wrap_exceptions=True) for if_clause in comp.ifs]):
                                    interpreter.env_stack.pop()
                                    continue
                                logger.debug('Debug: No if conditions or conditions passed, proceeding to compute elt')
                                val = await interpreter.visit(node.elt, wrap_exceptions=wrap_exceptions)
                                logger.debug(f'Debug: Value computed for elt: {val}')
                                yield val
                            except Exception as e:
                                lineno = getattr(node, "lineno", 1)
                                col = getattr(node, "col_offset", 0)
                                context_line = interpreter.source_lines[lineno - 1] if interpreter.source_lines and lineno <= len(interpreter.source_lines) else ""
                                from .exceptions import WrappedException
                                raise WrappedException(f"Error in async iteration: {str(e)}", e, lineno, col, context_line) from e
                            finally:
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