import ast
import asyncio
import inspect
from typing import Any, Dict, List, Optional
from collections import deque

from .interpreter_core import ASTInterpreter
from .exceptions import ReturnException


class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            return next(self.gen)
        except StopIteration:
            self.closed = True
            raise

    def send(self, value):
        if self.closed:
            raise StopIteration
        try:
            return self.gen.send(value)
        except StopIteration:
            self.closed = True
            raise

    def throw(self, exc):
        if self.closed:
            raise StopIteration
        try:
            return self.gen.throw(exc)
        except StopIteration:
            self.closed = True
            raise

    def close(self):
        self.closed = True
        self.gen.close()


class Function:
    def __init__(self, node: ast.FunctionDef, closure: List[Dict[str, Any]], interpreter: ASTInterpreter,
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        self.node: ast.FunctionDef = node
        self.closure: List[Dict[str, Any]] = closure[:]
        self.interpreter: ASTInterpreter = interpreter
        # Fix: Support positional-only parameters
        self.posonly_params = [arg.arg for arg in node.args.posonlyargs] if hasattr(node.args, 'posonlyargs') else []
        self.pos_kw_params = pos_kw_params
        self.vararg_name = vararg_name
        self.kwonly_params = kwonly_params
        self.kwarg_name = kwarg_name
        self.pos_defaults = pos_defaults
        self.kw_defaults = kw_defaults
        self.defining_class = None
        self.is_generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}
        local_frame[self.node.name] = self

        # Fix: Handle positional-only parameters
        num_posonly = len(self.posonly_params)
        num_pos_kw = len(self.pos_kw_params)
        total_pos = num_posonly + num_pos_kw
        pos_args_used = 0

        for i, arg in enumerate(args):
            if i < num_posonly:
                local_frame[self.posonly_params[i]] = arg
                pos_args_used += 1
            elif i < total_pos:
                local_frame[self.pos_kw_params[i - num_posonly]] = arg
                pos_args_used += 1
            elif self.vararg_name:
                if self.vararg_name not in local_frame:
                    local_frame[self.vararg_name] = []
                local_frame[self.vararg_name].append(arg)
            else:
                raise TypeError(f"Function '{self.node.name}' takes {total_pos} positional arguments but {len(args)} were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.posonly_params:
                raise TypeError(f"Function '{self.node.name}' got an unexpected keyword argument '{kwarg_name}' (positional-only)")
            elif kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError(f"Function '{self.node.name}' got multiple values for argument '{kwarg_name}'")
                local_frame[kwarg_name] = kwarg_value
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
            else:
                raise TypeError(f"Function '{self.node.name}' got an unexpected keyword argument '{kwarg_name}'")

        for param in self.pos_kw_params:
            if param not in local_frame and param in self.pos_defaults:
                local_frame[param] = self.pos_defaults[param]
        for param in self.kwonly_params:
            if param not in local_frame and param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]

        if self.kwarg_name and self.kwarg_name in local_frame:
            local_frame[self.kwarg_name] = dict(local_frame[self.kwarg_name])

        missing_args = [param for param in self.posonly_params if param not in local_frame]
        missing_args += [param for param in self.pos_kw_params if param not in local_frame and param not in self.pos_defaults]
        missing_args += [param for param in self.kwonly_params if param not in local_frame and param not in self.kw_defaults]
        if missing_args:
            raise TypeError(f"Function '{self.node.name}' missing required arguments: {', '.join(missing_args)}")

        if self.pos_kw_params and self.pos_kw_params[0] == 'self' and args:
            local_frame['self'] = args[0]
            local_frame['__current_method__'] = self

        new_env_stack.append(local_frame)
        new_interp: ASTInterpreter = self.interpreter.spawn_from_env(new_env_stack)
        
        if self.defining_class and args:
            new_interp.current_class = self.defining_class
            new_interp.current_instance = args[0]

        if self.is_generator:
            # Updated: Return a proper synchronous generator for sync functions
            def generator():
                for body_stmt in self.node.body:
                    if isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Yield):
                        # Use asyncio.run to handle any awaits in the yield value
                        value = asyncio.run(new_interp.visit(body_stmt.value, wrap_exceptions=True))
                        yield value
                    elif isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.YieldFrom):
                        sub_iterable = asyncio.run(new_interp.visit(body_stmt.value, wrap_exceptions=True))
                        if hasattr(sub_iterable, '__aiter__'):
                            raise TypeError("cannot 'yield from' an asynchronous iterable in a synchronous generator")
                        else:
                            for v in sub_iterable:
                                yield v
                    elif isinstance(body_stmt, ast.Return):
                        value = asyncio.run(new_interp.visit(body_stmt.value, wrap_exceptions=True)) if body_stmt.value else None
                        raise StopIteration(value)
                    else:
                        asyncio.run(new_interp.visit(body_stmt, wrap_exceptions=True))
            return GeneratorWrapper(generator())
        else:
            last_value = None
            try:
                for stmt in self.node.body:
                    last_value = await new_interp.visit(stmt, wrap_exceptions=True)
                return last_value
            except ReturnException as ret:
                return ret.value
            return last_value

    def _send_sync(self, gen, value):
        try:
            return next(gen) if value is None else gen.send(value)
        except StopIteration:
            raise

    def _throw_sync(self, gen, exc):
        try:
            return gen.throw(exc)
        except StopIteration:
            raise

    def __get__(self, instance: Any, owner: Any):
        if instance is None:
            return self
        async def method(*args: Any, **kwargs: Any) -> Any:
            return await self(instance, *args, **kwargs)
        method.__self__ = instance
        return method


class AsyncFunction:
    def __init__(self, node: ast.AsyncFunctionDef, closure: List[Dict[str, Any]], interpreter: ASTInterpreter,
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        self.node: ast.AsyncFunctionDef = node
        self.closure: List[Dict[str, Any]] = closure[:]
        self.interpreter: ASTInterpreter = interpreter
        # Fix: Support positional-only parameters
        self.posonly_params = [arg.arg for arg in node.args.posonlyargs] if hasattr(node.args, 'posonlyargs') else []
        self.pos_kw_params = pos_kw_params
        self.vararg_name = vararg_name
        self.kwonly_params = kwonly_params
        self.kwarg_name = kwarg_name
        self.pos_defaults = pos_defaults
        self.kw_defaults = kw_defaults

    async def __call__(self, *args: Any, _return_locals: bool = False, **kwargs: Any) -> Any:
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}
        local_frame[self.node.name] = self

        num_posonly = len(self.posonly_params)
        num_pos_kw = len(self.pos_kw_params)
        total_pos = num_posonly + num_pos_kw
        pos_args_used = 0

        for i, arg in enumerate(args):
            if i < num_posonly:
                local_frame[self.posonly_params[i]] = arg
                pos_args_used += 1
            elif i < total_pos:
                local_frame[self.pos_kw_params[i - num_posonly]] = arg
                pos_args_used += 1
            elif self.vararg_name:
                if self.vararg_name not in local_frame:
                    local_frame[self.vararg_name] = []
                local_frame[self.vararg_name].append(arg)
            else:
                raise TypeError(f"Async function '{self.node.name}' takes {total_pos} positional arguments but {len(args)} were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.posonly_params:
                raise TypeError(f"Async function '{self.node.name}' got an unexpected keyword argument '{kwarg_name}' (positional-only)")
            elif kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError(f"Async function '{self.node.name}' got multiple values for argument '{kwarg_name}'")
                local_frame[kwarg_name] = kwarg_value
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
            else:
                raise TypeError(f"Async function '{self.node.name}' got an unexpected keyword argument '{kwarg_name}'")

        for param in self.pos_kw_params:
            if param not in local_frame and param in self.pos_defaults:
                local_frame[param] = self.pos_defaults[param]
        for param in self.kwonly_params:
            if param not in local_frame and param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]

        missing_args = [param for param in self.posonly_params if param not in local_frame]
        missing_args += [param for param in self.pos_kw_params if param not in local_frame and param not in self.pos_defaults]
        missing_args += [param for param in self.kwonly_params if param not in local_frame and param not in self.kw_defaults]
        if missing_args:
            raise TypeError(f"Async function '{self.node.name}' missing required arguments: {', '.join(missing_args)}")

        new_env_stack.append(local_frame)
        new_interp: ASTInterpreter = self.interpreter.spawn_from_env(new_env_stack)
        last_value = None
        try:
            for stmt in self.node.body:
                last_value = await new_interp.visit(stmt, wrap_exceptions=True)
            if _return_locals:
                local_vars = {k: v for k, v in local_frame.items() if not k.startswith('__')}
                return last_value, local_vars
            return last_value
        except ReturnException as ret:
            if _return_locals:
                local_vars = {k: v for k, v in local_frame.items() if not k.startswith('__')}
                return ret.value, local_vars
            return ret.value
        finally:
            if not _return_locals:
                new_env_stack.pop()


class AsyncGeneratorFunction:
    def __init__(self, node: ast.AsyncFunctionDef, closure: List[Dict[str, Any]], interpreter: ASTInterpreter,
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        self.node: ast.AsyncFunctionDef = node
        self.closure: List[Dict[str, Any]] = closure[:]
        self.interpreter: ASTInterpreter = interpreter
        self.posonly_params = [arg.arg for arg in node.args.posonlyargs] if hasattr(node.args, 'posonlyargs') else []
        self.pos_kw_params = pos_kw_params
        self.vararg_name = vararg_name
        self.kwonly_params = kwonly_params
        self.kwarg_name = kwarg_name
        self.pos_defaults = pos_defaults
        self.kw_defaults = kw_defaults

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}
        local_frame[self.node.name] = self

        num_posonly = len(self.posonly_params)
        num_pos_kw = len(self.pos_kw_params)
        total_pos = num_posonly + num_pos_kw
        pos_args_used = 0

        for i, arg in enumerate(args):
            if i < num_posonly:
                local_frame[self.posonly_params[i]] = arg
                pos_args_used += 1
            elif i < total_pos:
                local_frame[self.pos_kw_params[i - num_posonly]] = arg
                pos_args_used += 1
            elif self.vararg_name:
                if self.vararg_name not in local_frame:
                    local_frame[self.vararg_name] = []
                local_frame[self.vararg_name].append(arg)
            else:
                raise TypeError(f"Async generator '{self.node.name}' takes {total_pos} positional arguments but {len(args)} were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.posonly_params:
                raise TypeError(f"Async generator '{self.node.name}' got an unexpected keyword argument '{kwarg_name}' (positional-only)")
            elif kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError(f"Async generator '{self.node.name}' got multiple values for argument '{kwarg_name}'")
                local_frame[kwarg_name] = kwarg_value
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
            else:
                raise TypeError(f"Async generator '{self.node.name}' got an unexpected keyword argument '{kwarg_name}'")

        for param in self.pos_kw_params:
            if param not in local_frame and param in self.pos_defaults:
                local_frame[param] = self.pos_defaults[param]
        for param in self.kwonly_params:
            if param not in local_frame and param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]

        missing_args = [param for param in self.posonly_params if param not in local_frame]
        missing_args += [param for param in self.pos_kw_params if param not in local_frame and param not in self.pos_defaults]
        missing_args += [param for param in self.kwonly_params if param not in local_frame and param not in self.kw_defaults]
        if missing_args:
            raise TypeError(f"Async generator '{self.node.name}' missing required arguments: {', '.join(missing_args)}")

        new_env_stack.append(local_frame)
        new_interp: ASTInterpreter = self.interpreter.spawn_from_env(new_env_stack)

        # Store the function node locally
        function_node = self.node

        # Helper class to represent a frame of execution with state
        class ExecutionFrame:
            def __init__(self, node, is_for_loop=False):
                self.node = node
                self.is_for_loop = is_for_loop
                self.index = 0
                self.variables = {}
                
                # For loop specific
                if is_for_loop:
                    self.target = node.target
                    self.iterator = None
                    self.is_async_iterator = False
                    
                # Determine statements to execute
                if hasattr(node, 'body'):
                    self.statements = node.body
                elif isinstance(node, list):
                    self.statements = node
                else:
                    self.statements = [node]

        class ImprovedAsyncGenerator:
            def __init__(self):
                self.running = False
                self.exhausted = False
                self.execution_stack = deque()
                # Use the function node passed from outer scope
                self.node = function_node
                self.execution_stack.append(ExecutionFrame(self.node.body))
                new_interp.generator_context = {
                    'active': True,
                    'yielded': False,
                    'yield_value': None,
                    'yield_from': False,
                    'yield_from_iterable': None,
                    'sent_value': None
                }
                self.interpreter = new_interp
                self.raise_pending = None

            def __aiter__(self):
                return self

            async def initialize_for_loop(self, frame):
                try:
                    # Get the iterable
                    iterable = await self.interpreter.visit(frame.node.iter, wrap_exceptions=True)
                    # Create an iterator
                    if hasattr(iterable, '__aiter__'):
                        frame.iterator = await iterable.__aiter__()
                        frame.is_async_iterator = True
                    else:
                        frame.iterator = iter(iterable)
                        frame.is_async_iterator = False
                    return True
                except Exception as e:
                    frame.iterator = None
                    return False

            async def advance_for_loop(self, frame):
                try:
                    # Get the next item
                    if frame.is_async_iterator:
                        try:
                            item = await frame.iterator.__anext__()
                        except StopAsyncIteration:
                            return False
                    else:
                        try:
                            item = next(frame.iterator)
                        except StopIteration:
                            return False
                    
                    # Assign to target
                    await self.interpreter.assign(frame.target, item)
                    
                    # Push loop body to execution stack
                    self.execution_stack.append(ExecutionFrame(frame.node.body))
                    return True
                except Exception as e:
                    # Skip to loop end on error
                    return False

            async def __anext__(self):
                if self.exhausted:
                    raise StopAsyncIteration
                if self.running:
                    raise RuntimeError("AsyncGenerator already running")
                
                try:
                    self.running = True
                    
                    # Handle any pending exceptions
                    if self.raise_pending:
                        exc = self.raise_pending
                        self.raise_pending = None
                        raise exc
                    
                    # Run until we yield a value or exhaust the generator
                    while self.execution_stack:
                        # Get current execution frame
                        frame = self.execution_stack[-1]
                        
                        # If we've processed all statements in this frame
                        if frame.index >= len(frame.statements):
                            self.execution_stack.pop()
                            
                            # If this was a loop body, go back to the loop head
                            if self.execution_stack and self.execution_stack[-1].is_for_loop:
                                loop_frame = self.execution_stack[-1]
                                has_more = await self.advance_for_loop(loop_frame)
                                if not has_more:
                                    self.execution_stack.pop()  # Loop is done
                            continue
                        
                        # Get current statement
                        stmt = frame.statements[frame.index]
                        
                        # Special handling for different statement types
                        if isinstance(stmt, (ast.For, ast.AsyncFor)):
                            # Initialize this as a for loop
                            loop_frame = ExecutionFrame(stmt, is_for_loop=True)
                            success = await self.initialize_for_loop(loop_frame)
                            
                            if success:
                                # Push the loop frame
                                self.execution_stack.append(loop_frame)
                                # Start the first iteration
                                has_items = await self.advance_for_loop(loop_frame)
                                if not has_items:
                                    self.execution_stack.pop()  # Empty loop
                            
                            # Move to next statement
                            frame.index += 1
                            continue
                        
                        # For Expr nodes that might contain yields
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Yield, ast.YieldFrom)):
                            value = await self.interpreter.visit(stmt.value, wrap_exceptions=True)
                            frame.index += 1  # Move to next statement for next time
                            
                            # Check if a yield was registered during execution
                            if self.interpreter.generator_context.get('yielded'):
                                self.interpreter.generator_context['yielded'] = False
                                value = self.interpreter.generator_context.get('yield_value')
                                return value
                            
                            # For raw yields that might not go through the generator context
                            return value
                        
                        # Execute the current statement
                        try:
                            result = await self.interpreter.visit(stmt, wrap_exceptions=True)
                        except ReturnException as ret:
                            # Return ends the generator with the return value
                            self.exhausted = True
                            if ret.value is not None:
                                return ret.value
                            raise StopAsyncIteration
                        except Exception as e:
                            # Re-raise any other exceptions
                            raise
                        
                        # Check if a yield occurred during execution
                        if self.interpreter.generator_context.get('yielded'):
                            self.interpreter.generator_context['yielded'] = False
                            value = self.interpreter.generator_context.get('yield_value')
                            
                            # Move to the next statement
                            frame.index += 1
                            return value
                        
                        # Move to the next statement if no yield occurred
                        frame.index += 1
                    
                    # If we've run out of statements, the generator is exhausted
                    self.exhausted = True
                    raise StopAsyncIteration
                    
                except StopAsyncIteration:
                    self.exhausted = True
                    raise
                except Exception as e:
                    if not isinstance(e, StopAsyncIteration):
                        self.exhausted = True
                    raise
                finally:
                    self.running = False

            async def asend(self, value):
                if self.exhausted:
                    raise StopAsyncIteration
                if self.running:
                    raise RuntimeError("AsyncGenerator already running")
                
                # Set the sent value
                self.interpreter.generator_context['sent_value'] = value
                
                # Continue execution
                return await self.__anext__()

            async def athrow(self, exc_type, exc_val=None, exc_tb=None):
                if self.exhausted:
                    raise StopAsyncIteration
                if self.running:
                    raise RuntimeError("AsyncGenerator already running")
                
                # Create the exception
                if exc_val is None:
                    exc_val = exc_type()
                elif isinstance(exc_val, type):
                    exc_val = exc_val()
                
                # Store the exception to be raised during next execution
                self.raise_pending = exc_val
                
                try:
                    # Try to resume execution - the exception will be thrown
                    return await self.__anext__()
                except StopAsyncIteration:
                    # If the generator is exhausted, propagate StopAsyncIteration
                    raise
                except Exception as e:
                    # If the exception wasn't handled, the generator is exhausted
                    self.exhausted = True
                    raise

            async def aclose(self):
                if self.exhausted:
                    return
                
                try:
                    # Try to throw GeneratorExit
                    await self.athrow(GeneratorExit)
                except (StopAsyncIteration, GeneratorExit):
                    # This is expected
                    pass
                
                # Ensure we're marked as exhausted
                self.exhausted = True

        # Return the async generator instance
        return ImprovedAsyncGenerator()


class LambdaFunction:
    def __init__(self, node: ast.Lambda, closure: List[Dict[str, Any]], interpreter: ASTInterpreter,
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        self.node: ast.Lambda = node
        self.closure: List[Dict[str, Any]] = closure[:]
        self.interpreter: ASTInterpreter = interpreter
        self.pos_kw_params = pos_kw_params
        self.vararg_name = vararg_name
        self.kwonly_params = kwonly_params
        self.kwarg_name = kwarg_name
        self.pos_defaults = pos_defaults
        self.kw_defaults = kw_defaults

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}

        num_pos = len(self.pos_kw_params)
        for i, arg in enumerate(args):
            if i < num_pos:
                local_frame[self.pos_kw_params[i]] = arg
            elif self.vararg_name:
                if self.vararg_name not in local_frame:
                    local_frame[self.vararg_name] = []
                local_frame[self.vararg_name].append(arg)
            else:
                raise TypeError(f"Lambda takes {num_pos} positional arguments but {len(args)} were given")
        if self.vararg_name and self.vararg_name not in local_frame:
            local_frame[self.vararg_name] = tuple()

        for kwarg_name, kwarg_value in kwargs.items():
            if kwarg_name in self.pos_kw_params or kwarg_name in self.kwonly_params:
                if kwarg_name in local_frame:
                    raise TypeError(f"Lambda got multiple values for argument '{kwarg_name}'")
                local_frame[kwarg_name] = kwarg_value
            elif self.kwarg_name:
                if self.kwarg_name not in local_frame:
                    local_frame[self.kwarg_name] = {}
                local_frame[self.kwarg_name][kwarg_name] = kwarg_value
            else:
                raise TypeError(f"Lambda got an unexpected keyword argument '{kwarg_name}'")

        for param in self.pos_kw_params:
            if param not in local_frame and param in self.pos_defaults:
                local_frame[param] = self.pos_defaults[param]
        for param in self.kwonly_params:
            if param not in local_frame and param in self.kw_defaults:
                local_frame[param] = self.kw_defaults[param]

        missing_args = [param for param in self.pos_kw_params if param not in local_frame and param not in self.pos_defaults]
        missing_args += [param for param in self.kwonly_params if param not in local_frame and param not in self.kw_defaults]
        if missing_args:
            raise TypeError(f"Lambda missing required arguments: {', '.join(missing_args)}")

        new_env_stack.append(local_frame)
        new_interp: ASTInterpreter = self.interpreter.spawn_from_env(new_env_stack)
        return await new_interp.visit(self.node.body, wrap_exceptions=True)