# quantalogic_pythonbox/async_generator.py
"""
Async generator function handling for the PythonBox interpreter.
"""

import ast
from typing import Any, Dict, List, Optional

from .interpreter_core import ASTInterpreter


class StatefulAsyncGenerator:
    """A stateful async generator that properly implements asend and athrow"""
    
    def __init__(self, interpreter, statements):
        self.interpreter = interpreter
        self.statements = statements
        self.coroutine = None
        self.started = False
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        """Standard async iteration protocol"""
        return await self.asend(None)
        
    async def asend(self, value):
        """Send a value to the generator"""
        if not self.started:
            self.started = True
            self.coroutine = self._run_generator()
            return await self.coroutine.asend(None)
        else:
            return await self.coroutine.asend(value)
            
    async def athrow(self, exc_type, exc_value=None, _traceback=None):
        """Throw an exception into the generator"""
        if not self.started:
            self.started = True
            self.coroutine = self._run_generator()
            await self.coroutine.asend(None)  # Start the generator
            
        # Create the exception instance
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            if exc_value is None:
                exception = exc_type()
            else:
                exception = exc_type(exc_value)
        else:
            exception = exc_type
            
        # Store the exception in context for the yield to throw
        self.interpreter.generator_context['thrown_exception'] = exception
        self.interpreter.generator_context['exception_thrown'] = True
        
        # Set the resuming flag for the specific yield that was last executed
        last_yield_key = self.interpreter.generator_context.get('last_yield_key')
        if last_yield_key:
            resuming_key = f'resuming_{last_yield_key}'
            self.interpreter.generator_context[resuming_key] = True
            self.interpreter.generator_context['resuming_from_asend'] = True
        
        # Continue execution by sending None - the exception will be thrown when the yield resumes
        try:
            result = await self.coroutine.asend(None)
            return result
        except StopAsyncIteration:
            raise
            
    async def aclose(self):
        """Close the generator"""
        if self.coroutine:
            await self.coroutine.aclose()
        
    async def _run_generator(self):
        """The actual async generator implementation"""
        from .exceptions import YieldException
        import uuid
        
        # Generate a unique ID for this generator instance
        generator_id = str(uuid.uuid4())
        
        # Set up the interpreter context with a stack for nested generators
        self.interpreter.generator_context['active'] = True
        self.interpreter.generator_context['yielding'] = True
        
        # Push this generator onto the stack
        if 'generator_stack' not in self.interpreter.generator_context:
            self.interpreter.generator_context['generator_stack'] = []
        self.interpreter.generator_context['generator_stack'].append(generator_id)
        
        try:
            # Execute the entire function body naturally - let yields bubble up
            stmt_index = 0
            while stmt_index < len(self.statements):
                stmt = self.statements[stmt_index]
                
                # Track current statement index for athrow
                self.interpreter.generator_context['current_stmt_index'] = stmt_index
                
                # Check if we need to re-execute a statement (for athrow)
                re_execute = self.interpreter.generator_context.get('re_execute_statement', False)
                if re_execute:
                    self.interpreter.generator_context['re_execute_statement'] = False
                
                try:
                    await self.interpreter.visit(stmt, wrap_exceptions=True)
                    # If no exception, move to next statement
                    stmt_index += 1
                except YieldException as ye:
                    # When we catch a YieldException, yield the value to the caller
                    # and wait for a value to be sent back
                    sent_value = yield ye.value
                    
                    # Store the sent value for any subsequent yield expressions
                    self.interpreter.generator_context['last_sent_value'] = sent_value
                    
                    # Only set resuming flags if there's actually a sent value (asend operation)
                    # For normal iteration, sent_value will be None and we don't want to set resuming flags
                    if sent_value is not None:
                        # Set the resuming flag for the specific yield that was just executed
                        last_yield_key = self.interpreter.generator_context.get('last_yield_key')
                        if last_yield_key:
                            resuming_key = f'resuming_{last_yield_key}'
                            self.interpreter.generator_context[resuming_key] = True
                            # Set a flag to indicate we're resuming from asend
                            self.interpreter.generator_context['resuming_from_asend'] = True
                    
                    # Check if we're inside a suspended loop
                    loop_suspended = self.interpreter.generator_context.get('loop_suspended', False)
                    if loop_suspended:
                        # Don't advance to the next statement - the loop should continue
                        self.interpreter.generator_context['loop_suspended'] = False
                        # Continue with the same statement (the loop)
                    else:
                        # Check if this is a yield expression within a statement (e.g., assignment)
                        # In that case, we need to re-execute the statement to complete it
                        current_stmt = self.statements[stmt_index]
                        
                        # Check if we need to re-execute because of athrow
                        exception_thrown = self.interpreter.generator_context.get('exception_thrown', False)
                        if exception_thrown:
                            # Don't advance - re-execute the same statement so the exception can be thrown
                            # Clear the flag to prevent infinite loops
                            # Note: don't clear exception_thrown here, let the yield visitor handle it
                            continue
                        
                        # If the statement is an assignment or other complex statement containing yield,
                        # we should re-execute it so the yield expression can return the sent value
                        elif isinstance(current_stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Return)):
                            # Don't advance - re-execute the same statement to complete the assignment
                            pass
                        elif isinstance(current_stmt, ast.Expr) and isinstance(current_stmt.value, ast.Yield):
                            # For simple yield statements (ast.Expr containing ast.Yield), advance to the next statement
                            stmt_index += 1
                        elif isinstance(current_stmt, ast.Expr):
                            # For other expression statements that might contain yield, re-execute
                            pass
                        else:
                            # For other statements, advance to the next statement
                            stmt_index += 1
                except Exception:
                    raise
                    
        finally:
            self.interpreter.generator_context['active'] = False
            self.interpreter.generator_context['yielding'] = False
            # Pop this generator from the stack
            if 'generator_stack' in self.interpreter.generator_context:
                try:
                    self.interpreter.generator_context['generator_stack'].remove(generator_id)
                except ValueError:
                    pass  # Generator ID not in stack, that's okay
            # Clean up the generator-specific resumption flag
            self.interpreter.generator_context.pop(f'resuming_yield_{generator_id}', None)


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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Set up the environment for the generator
        new_env_stack: List[Dict[str, Any]] = self.closure[:]
        local_frame: Dict[str, Any] = {}
        local_frame[self.node.name] = self

        # Handle parameter binding (same as before)
        num_posonly = len(self.posonly_params)
        num_pos_kw = len(self.pos_kw_params)
        total_pos = num_posonly + num_pos_kw

        for i, arg in enumerate(args):
            if i < num_posonly:
                local_frame[self.posonly_params[i]] = arg
            elif i < total_pos:
                local_frame[self.pos_kw_params[i - num_posonly]] = arg
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

        # Return a stateful async generator that properly implements asend/athrow
        return StatefulAsyncGenerator(new_interp, self.node.body)

    async def _execute_with_yields(self, interpreter, statements):
        """Execute statements and yield any values produced by yield expressions."""
        from .exceptions import YieldException
        import inspect
        
        for stmt in statements:
            try:
                result = await interpreter.visit(stmt, wrap_exceptions=True)
                
                # Check if the result is an async generator (from visit_AsyncFor in generator context)
                if inspect.isasyncgen(result):
                    # Iterate through the async generator and yield its values
                    async for value in result:
                        yield value
                
            except YieldException as ye:
                yield ye.value