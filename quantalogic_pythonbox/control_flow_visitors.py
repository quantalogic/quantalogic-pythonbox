import ast
import logging
from typing import Any

from .exceptions import BreakException, ContinueException, ReturnException
from .interpreter_core import ASTInterpreter

logging.basicConfig(level=logging.DEBUG)  # Set debug level

# Configure logging
logger = logging.getLogger(__name__)

async def async_enumerate(async_iterable, start=0):
    """Async-compatible enumerate for async iterables."""
    index = start
    async for item in async_iterable:
        logger.debug(f"Yielding tuple: ({index}, {item}) ")
        yield index, item
        index += 1

async def visit_If(self: ASTInterpreter, node: ast.If, wrap_exceptions: bool = True) -> Any:
    logger.debug("Visiting If")
    if await self.visit(node.test, wrap_exceptions=wrap_exceptions):
        branch = node.body
    else:
        branch = node.orelse
    result = None
    if branch:
        for stmt in branch[:-1]:
            await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        result = await self.visit(branch[-1], wrap_exceptions=wrap_exceptions)
    logger.debug(f"If result: {result}")
    return result

async def visit_While(self: ASTInterpreter, node: ast.While, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting While")
    while await self.visit(node.test, wrap_exceptions=wrap_exceptions):
        try:
            for stmt in node.body:
                await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        except StopAsyncIteration:
            # Propagate to user try/except when wrapping is disabled
            if not wrap_exceptions:
                raise
            logger.debug("StopAsyncIteration in While, breaking loop")
            break
        except BreakException:
            logger.debug("Break in While")
            break
        except ContinueException:
            logger.debug("Continue in While")
            continue
    for stmt in node.orelse:
        await self.visit(stmt, wrap_exceptions=wrap_exceptions)
    logger.debug("While completed")

async def visit_For(self: ASTInterpreter, node: ast.For, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting For")
    iter_obj: Any = await self.visit(node.iter, wrap_exceptions=wrap_exceptions)
    broke = False
    if hasattr(iter_obj, '__aiter__'):
        async for item in iter_obj:
            await self.assign(node.target, item)
            try:
                for stmt in node.body:
                    await self.visit(stmt, wrap_exceptions=wrap_exceptions)
            except BreakException:
                logger.debug("Break in async For")
                broke = True
                break
            except ContinueException:
                logger.debug("Continue in async For")
                continue
    else:
        for item in iter_obj:
            await self.assign(node.target, item)
            try:
                for stmt in node.body:
                    await self.visit(stmt, wrap_exceptions=wrap_exceptions)
            except BreakException:
                logger.debug("Break in sync For")
                broke = True
                break
            except ContinueException:
                logger.debug("Continue in sync For")
                continue
    if not broke:
        for stmt in node.orelse:
            await self.visit(stmt, wrap_exceptions=wrap_exceptions)
    logger.debug("For completed")

async def visit_AsyncFor(self: ASTInterpreter, node: ast.AsyncFor, wrap_exceptions: bool = True) -> None:
    iter_obj = await self.visit(node.iter, wrap_exceptions=wrap_exceptions)
    logger.debug(f"Visit_AsyncFor called for node at line {node.lineno if hasattr(node, 'lineno') else 'unknown'}, iterable after visit: {type(iter_obj).__name__}, value: {iter_obj if isinstance(iter_obj, (int, str, list, dict)) else type(iter_obj)}")
    logger.debug(f"Generator context active: {self.generator_context.get('active', False)}")
    iterable = iter_obj
    logger.debug(f"AsyncFor iterable type: {type(iterable).__name__}")
    if not hasattr(iterable, '__aiter__'):
        logger.error(f"Object {iterable} does not have __aiter__, raising TypeError")
        raise TypeError(f"Object {iterable} is not an async iterable")
    iterator = iterable.__aiter__()
    logger.debug(f"Iterator created: type {type(iterator).__name__}")
    broke = False
    while True:
        try:
            logger.debug(f"Calling __anext__ on iterator of type {type(iterator).__name__}")
            value = await iterator.__anext__()
            logger.debug(f"Received value from __anext__: {value}, active: {self.generator_context.get('active', False)}")
            await self.assign(node.target, value)
            logger.debug(f"Assigned item in AsyncFor: value {value}, type {type(value).__name__}, target: {node.target.id if isinstance(node.target, ast.Name) else 'unknown'}")
            logger.debug(f"AsyncFor iteration count: {self.iteration_count if hasattr(self, 'iteration_count') else 0}")  # Add a counter if needed, but for now, log per iteration
        except StopAsyncIteration:
            logger.debug("__anext__ raised StopAsyncIteration, breaking loop")
            break
        except Exception as e:
            logger.error(f"Exception in __anext__: {str(e)}, type: {type(e).__name__}")
            raise
        try:
            logger.debug(f"Debug in visit_AsyncFor: iterated value {value}")
            for stmt in node.body:
                await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        except BreakException:
            logger.debug("Break in AsyncFor")
            broke = True
            break
        except ContinueException:
            logger.debug("Continue in AsyncFor")
            continue
    if not broke:
        for stmt in node.orelse:
            await self.visit(stmt, wrap_exceptions=wrap_exceptions)
    logger.debug("AsyncFor completed")

async def visit_Break(self: ASTInterpreter, node: ast.Break, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting Break")
    raise BreakException()

async def visit_Continue(self: ASTInterpreter, node: ast.Continue, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting Continue")
    raise ContinueException()

async def visit_Return(self: ASTInterpreter, node: ast.Return, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting Return with value node of type %s", type(node.value).__name__ if node.value else 'None')
    value: Any = await self.visit(node.value, wrap_exceptions=wrap_exceptions) if node.value is not None else None
    logger.debug(f"Executing return with value: {value}, type: {type(value).__name__ if value is not None else 'NoneType'}")
    raise ReturnException(value)

async def visit_IfExp(self: ASTInterpreter, node: ast.IfExp, wrap_exceptions: bool = True) -> Any:
    logger.debug("Visiting IfExp")
    result = await self.visit(node.body, wrap_exceptions=wrap_exceptions) if await self.visit(node.test, wrap_exceptions=wrap_exceptions) else await self.visit(node.orelse, wrap_exceptions=wrap_exceptions)
    logger.debug(f"IfExp result: {result}")
    return result