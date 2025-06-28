import ast
from typing import Any

from .exceptions import BreakException, ContinueException, ReturnException
from .interpreter_core import ASTInterpreter

async def visit_If(self: ASTInterpreter, node: ast.If, wrap_exceptions: bool = True) -> Any:
    if await self.visit(node.test, wrap_exceptions=wrap_exceptions):
        branch = node.body
    else:
        branch = node.orelse
    result = None
    if branch:
        for stmt in branch[:-1]:
            await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        result = await self.visit(branch[-1], wrap_exceptions=wrap_exceptions)
    return result

async def visit_While(self: ASTInterpreter, node: ast.While, wrap_exceptions: bool = True) -> None:
    while await self.visit(node.test, wrap_exceptions=wrap_exceptions):
        try:
            for stmt in node.body:
                await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        except BreakException:
            break
        except ContinueException:
            continue
    for stmt in node.orelse:
        await self.visit(stmt, wrap_exceptions=wrap_exceptions)

async def visit_For(self: ASTInterpreter, node: ast.For, wrap_exceptions: bool = True) -> None:
    iter_obj: Any = await self.visit(node.iter, wrap_exceptions=wrap_exceptions)
    broke = False
    if hasattr(iter_obj, '__aiter__'):
        async for item in iter_obj:
            await self.assign(node.target, item)
            try:
                for stmt in node.body:
                    await self.visit(stmt, wrap_exceptions=wrap_exceptions)
            except BreakException:
                broke = True
                break
            except ContinueException:
                continue
    else:
        for item in iter_obj:
            await self.assign(node.target, item)
            try:
                for stmt in node.body:
                    await self.visit(stmt, wrap_exceptions=wrap_exceptions)
            except BreakException:
                broke = True
                break
            except ContinueException:
                continue
    if not broke:
        for stmt in node.orelse:
            await self.visit(stmt, wrap_exceptions=wrap_exceptions)

async def visit_AsyncFor(self: ASTInterpreter, node: ast.AsyncFor, wrap_exceptions: bool = True) -> None:
    from .exceptions import YieldException
    
    # Execute normally - yields will bubble up through YieldException
    iterable = await self.visit(node.iter, wrap_exceptions=wrap_exceptions)
    broke = False
    
    # Check if we're inside a generator
    is_in_generator = (hasattr(self, 'generator_context') and 
                      self.generator_context.get('active', False))
    
    if hasattr(iterable, '__aiter__'):
        # Check if we're resuming from a previous suspension
        loop_state_key = f'asyncfor_state_{id(node)}'
        loop_state = self.generator_context.get(loop_state_key) if is_in_generator else None
        
        # Get or create the async iterator
        aiterator_key = f'asyncfor_aiter_{id(node)}'
        if is_in_generator and aiterator_key in self.generator_context:
            aiterator = self.generator_context[aiterator_key]
        else:
            aiterator = iterable.__aiter__()
            if is_in_generator:
                self.generator_context[aiterator_key] = aiterator
        
        if loop_state:
            # Resume from where we left off - continue with remaining statements
            current_value = loop_state['current_value']
            stmt_index = loop_state['stmt_index']
            
            # Continue executing the remaining statements in the current iteration
            for i in range(stmt_index, len(node.body)):
                stmt = node.body[i]
                try:
                    await self.visit(stmt, wrap_exceptions=wrap_exceptions)
                except BreakException:
                    broke = True
                    break
                except ContinueException:
                    break
                except YieldException as ye:
                    # Store the state for resumption
                    self.generator_context[loop_state_key] = {
                        'current_value': current_value,
                        'stmt_index': i + 1  # Next statement to execute
                    }
                    self.generator_context['loop_suspended'] = True
                    raise ye
            
            # Clear the loop state since we completed this iteration's remaining statements
            self.generator_context.pop(loop_state_key, None)
            
            if broke:
                # Clean up the iterator if we break
                if is_in_generator:
                    self.generator_context.pop(aiterator_key, None)
                return
                
        # Continue with the async iteration
        try:
            while True:
                try:
                    value = await aiterator.__anext__()
                except StopAsyncIteration:
                    break
                    
                await self.assign(node.target, value)
                
                # Execute body statements
                for stmt_index, stmt in enumerate(node.body):
                    try:
                        await self.visit(stmt, wrap_exceptions=wrap_exceptions)
                    except BreakException:
                        broke = True
                        break
                    except ContinueException:
                        break  # Break from stmt loop to continue with next iteration
                    except YieldException as ye:
                        # If we're inside a generator, store the loop state
                        if is_in_generator:
                            loop_state_data = {
                                'current_value': value,
                                'stmt_index': stmt_index + 1  # Next statement to execute
                            }
                            self.generator_context[loop_state_key] = loop_state_data
                            self.generator_context['loop_suspended'] = True
                        # Re-raise the YieldException to let the generator handle it
                        raise ye
                
                if broke:
                    break
        finally:
            # Clean up the iterator state when done, but only if the loop is not suspended
            if is_in_generator:
                # Only clean up if we're not suspended (i.e., the loop is actually finished)
                loop_suspended = self.generator_context.get('loop_suspended', False)
                if not loop_suspended:
                    self.generator_context.pop(aiterator_key, None)
                    self.generator_context.pop(loop_state_key, None)
                
    else:
        raise TypeError(f"Object {iterable} is not an async iterable")
    
    if not broke:
        for stmt in node.orelse:
            await self.visit(stmt, wrap_exceptions=wrap_exceptions)

async def visit_Break(self: ASTInterpreter, node: ast.Break, wrap_exceptions: bool = True) -> None:
    raise BreakException()

async def visit_Continue(self: ASTInterpreter, node: ast.Continue, wrap_exceptions: bool = True) -> None:
    raise ContinueException()

async def visit_Return(self: ASTInterpreter, node: ast.Return, wrap_exceptions: bool = True) -> None:
    value: Any = await self.visit(node.value, wrap_exceptions=wrap_exceptions) if node.value is not None else None
    raise ReturnException(value)

async def visit_IfExp(self: ASTInterpreter, node: ast.IfExp, wrap_exceptions: bool = True) -> Any:
    result = await self.visit(node.body, wrap_exceptions=wrap_exceptions) if await self.visit(node.test, wrap_exceptions=wrap_exceptions) else await self.visit(node.orelse, wrap_exceptions=wrap_exceptions)
    return result