import ast
import logging
from typing import Any

from .exceptions import BreakException, ContinueException, ReturnException
from .interpreter_core import ASTInterpreter

logging.basicConfig(level=logging.DEBUG)  # Set debug level

# Configure logging
logger = logging.getLogger(__name__)

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
    logger.debug("Visiting AsyncFor")
    from .exceptions import YieldException
    
    # Execute normally - yields will bubble up through YieldException
    iterable = await self.visit(node.iter, wrap_exceptions=wrap_exceptions)
    broke = False
    
    # Check if we're inside a generator
    is_in_generator = (hasattr(self, 'generator_context') and 
                      self.generator_context.get('active', False))
    
    if hasattr(iterable, '__aiter__'):
        async for value in iterable:
            logger.debug("AsyncFor iteration with value: %s", value)
            await self.assign(node.target, value)
            
            # Execute body statements
            for stmt in node.body:
                try:
                    await self.visit(stmt, wrap_exceptions=wrap_exceptions)
                except BreakException:
                    logger.debug("Break in AsyncFor")
                    broke = True
                    break
                except ContinueException:
                    logger.debug("Continue in AsyncFor")
                    break  # Break from stmt loop to continue with next iteration
                except YieldException as ye:
                    # If we're inside a generator, we need to handle yields specially
                    if is_in_generator:
                        # Mark that this statement (the async for loop) should not advance
                        # to the next statement after the yield
                        self.generator_context['loop_suspended'] = True
                        logger.debug("AsyncFor suspended due to yield in generator")
                    # Re-raise the YieldException to let the generator handle it
                    raise ye
            
            if broke:
                break
    else:
        raise TypeError(f"Object {iterable} is not an async iterable")
    
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
    logger.debug("Visiting Return")
    value: Any = await self.visit(node.value, wrap_exceptions=wrap_exceptions) if node.value is not None else None
    raise ReturnException(value)

async def visit_IfExp(self: ASTInterpreter, node: ast.IfExp, wrap_exceptions: bool = True) -> Any:
    logger.debug("Visiting IfExp")
    result = await self.visit(node.body, wrap_exceptions=wrap_exceptions) if await self.visit(node.test, wrap_exceptions=wrap_exceptions) else await self.visit(node.orelse, wrap_exceptions=wrap_exceptions)
    logger.debug(f"IfExp result: {result}")
    return result