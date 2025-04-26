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
    logger.debug(f"Visit_AsyncFor called for node at line {node.lineno}, iterable type after visit: {type(await self.visit(node.iter, wrap_exceptions=wrap_exceptions)) if hasattr(node, 'iter') else 'N/A'}")
    logger.debug(f"Entering visit_AsyncFor with iterable type: {type(node.iter).__name__}")
    try:
        iter_obj = await self.visit(node.iter, wrap_exceptions=wrap_exceptions)
        logger.debug(f"Iterable object created: type {type(iter_obj)}, value {iter_obj}")
        logger.debug("Starting async for loop iteration")
        iterable = iter_obj
        logger.debug(f"AsyncFor iterable type: {type(iterable)}")
        broke = False
        if hasattr(iterable, '__aiter__'):
            iterator = iterable.__aiter__()
            while True:
                try:
                    logger.debug(f"Calling __anext__ on iterator of type {type(iterator).__name__}")
                    value = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                logger.debug(f"AsyncFor iteration with value: {value}, type: {type(value)}")
                await self.assign(node.target, value)
                logger.debug(f"Assigned item in AsyncFor: value {value}, type {type(value)}")
                try:
                    for stmt in node.body:
                        await self.visit(stmt, wrap_exceptions=wrap_exceptions)
                except BreakException:
                    logger.debug("Break in AsyncFor")
                    broke = True
                    break
                except ContinueException:
                    logger.debug("Continue in AsyncFor")
                    continue
        else:
            raise TypeError(f"Object {iterable} is not an async iterable")
        if not broke:
            for stmt in node.orelse:
                await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        logger.debug("AsyncFor completed")
    except Exception as e:
        logger.error(f"Exception in visit_AsyncFor: {str(e)}")
        raise

async def visit_Break(self: ASTInterpreter, node: ast.Break, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting Break")
    raise BreakException()

async def visit_Continue(self: ASTInterpreter, node: ast.Continue, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting Continue")
    raise ContinueException()

async def visit_Return(self: ASTInterpreter, node: ast.Return, wrap_exceptions: bool = True) -> None:
    logger.debug("Visiting Return")
    value: Any = await self.visit(node.value, wrap_exceptions=wrap_exceptions) if node.value is not None else None
    logger.debug(f"Returning value: {value}")
    raise ReturnException(value)

async def visit_IfExp(self: ASTInterpreter, node: ast.IfExp, wrap_exceptions: bool = True) -> Any:
    logger.debug("Visiting IfExp")
    result = await self.visit(node.body, wrap_exceptions=wrap_exceptions) if await self.visit(node.test, wrap_exceptions=wrap_exceptions) else await self.visit(node.orelse, wrap_exceptions=wrap_exceptions)
    logger.debug(f"IfExp result: {result}")
    return result