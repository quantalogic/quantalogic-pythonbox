# quantalogic_pythonbox/function_base.py
"""
Base function handling for synchronous functions in the PythonBox interpreter.
"""

import ast
import logging
from typing import Any, Dict, List, Optional

from .interpreter_core import ASTInterpreter
from .exceptions import ReturnException
from .generator_wrapper import GeneratorWrapper

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Function:
    def __init__(self, node: ast.FunctionDef, closure: List[Dict[str, Any]], interpreter: ASTInterpreter,
                 pos_kw_params: List[str], vararg_name: Optional[str], kwonly_params: List[str],
                 kwarg_name: Optional[str], pos_defaults: Dict[str, Any], kw_defaults: Dict[str, Any]) -> None:
        self.node: ast.FunctionDef = node
        self.closure: List[Dict[str, Any]] = closure[:]
        self.interpreter: ASTInterpreter = interpreter
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
        new_interp.env_stack[0]['logger'].debug(f"Calling function {self.node.name}")
        
        if self.defining_class and args:
            new_interp.current_class = self.defining_class
            new_interp.current_instance = args[0]

        if self.is_generator:
            # Return a GeneratorWrapper for synchronous generators
            async def execute_generator():
                new_interp.generator_context = {
                    'active': True,
                    'yielded': False,
                    'yield_value': None,
                    'sent_value': None,
                    'yield_from': False,
                    'yield_from_iterator': None
                }
                
                values = []
                return_value = None
                
                try:
                    for stmt in self.node.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Yield):
                            value = await new_interp.visit(stmt.value.value, wrap_exceptions=False) if stmt.value.value else None
                            values.append(value)
                            continue
                            
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.YieldFrom):
                            iterable = await new_interp.visit(stmt.value.value, wrap_exceptions=False)
                            for val in iterable:
                                values.append(val)
                            continue
                            
                        if isinstance(stmt, ast.Return):
                            return_value = await new_interp.visit(stmt.value, wrap_exceptions=False) if stmt.value else None
                            raise StopIteration(return_value)
                            
                        await new_interp.visit(stmt, wrap_exceptions=False)
                        
                        if new_interp.generator_context.get('yielded'):
                            new_interp.generator_context['yielded'] = False
                            value = new_interp.generator_context.get('yield_value')
                            values.append(value)
                            
                        if new_interp.generator_context.get('yield_from'):
                            new_interp.generator_context['yield_from'] = False
                            iterator = new_interp.generator_context.get('yield_from_iterator')
                            if iterator:
                                for val in iterator:
                                    values.append(val)
                except StopIteration as e:
                    if e.__cause__ is None:  # Only convert manual StopIteration raises
                        raise RuntimeError("generator raised StopIteration") from e
                    raise  # Propagate StopIteration from return
                except ReturnException as ret:
                    return_value = ret.value
                
                # Create a generator that includes the return value
                def gen_with_return():
                    for val in values:
                        yield val
                    # This ensures the return value is captured in StopIteration
                    return return_value
                    
                return GeneratorWrapper(gen_with_return())
                
            gen = await execute_generator()
            return gen  # Return GeneratorWrapper instead of list
        else:
            last_value = None
            try:
                for stmt in self.node.body:
                    last_value = await new_interp.visit(stmt, wrap_exceptions=False)
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
        
        # For methods that need special handling (like __getitem__ which is used in slicing)
        if self.node.name in ['__getitem__', '__setitem__', '__delitem__', '__contains__']:
            # For __getitem__ special method, handle it directly
            if self.node.name == '__getitem__':
                async def special_method(key):
                    logger.debug(f"Entering special_method for key: {key}, type: {type(key)}")
                    try:
                        # Get the function body
                        body = self.node.body
                        
                        # Create a new environment with the instance and the key parameter
                        new_env_stack = self.closure[:]
                        local_frame = {}
                        
                        # Add the instance as 'self'
                        local_frame[self.pos_kw_params[0]] = instance
                        
                        # Add the key parameter
                        if len(self.pos_kw_params) > 1:
                            local_frame[self.pos_kw_params[1]] = key
                        
                        new_env_stack.append(local_frame)
                        
                        # Create a new interpreter with our environment
                        new_interp = self.interpreter.spawn_from_env(new_env_stack)
                        
                        # Execute the function body synchronously and capture the return value
                        result = None
                        for stmt in body:
                            try:
                                exec_result = await new_interp.run_sync_stmt(stmt)
                                logger.debug(f"Executed statement in __getitem__: {type(stmt).__name__} at line {getattr(stmt, 'lineno', 'unknown')}, result: {exec_result}")
                            except ReturnException as re:
                                return re.value
                            except Exception as e:
                                logger.error(f"Exception in statement execution for {type(stmt).__name__}: {str(e)}")
                                raise
                            if exec_result is not None:
                                result = exec_result
                        logger.debug(f"__getitem__ returning: {result}, type: {type(result)}")
                        return result
                    except Exception as e:
                        logger.error(f"Error in {self.node.name}: {str(e)}")
                        from quantalogic_pythonbox.exceptions import WrappedException
                        lineno = getattr(self.node, 'lineno', 0)
                        col = getattr(self.node, 'col_offset', 0)
                        context_line = f"{self.node.name} call"
                        raise WrappedException(str(e), e, lineno, col, context_line) from e
            else:  
                # Create a method that will be called synchronously using direct AST execution
                async def special_method(*args: Any, **kwargs: Any) -> Any:
                    logger.debug(f"Entering special_method for {self.node.name}")
                    # Get the function body
                    body = self.node.body
                    
                    # Create a new environment with the instance and arguments
                    new_env_stack = self.closure[:]
                    local_frame = {}
                    
                    # Add the function itself to the local scope
                    local_frame[self.node.name] = self
                    
                    # Add the instance as the first parameter (self)
                    if len(self.pos_kw_params) > 0:
                        local_frame[self.pos_kw_params[0]] = instance
                    
                    # Process the arguments
                    for i, arg in enumerate(args):
                        param_idx = i + 1  # +1 to account for 'self'
                        if param_idx < len(self.pos_kw_params):
                            local_frame[self.pos_kw_params[param_idx]] = arg
                    
                    # Add any keyword arguments
                    for name, value in kwargs.items():
                        local_frame[name] = value
                    
                    new_env_stack.append(local_frame)
                    
                    # Create a new interpreter with our environment
                    new_interp = self.interpreter.spawn_from_env(new_env_stack)
                    
                    # Execute each statement in the function body synchronously
                    result = None
                    for stmt in body:
                        try:
                            exec_result = await new_interp.run_sync_stmt(stmt)
                            logger.debug(f"Executed statement in special_method: {exec_result}")
                            if isinstance(exec_result, ReturnException):
                                return exec_result.value
                            elif exec_result is not None:
                                result = exec_result
                                break
                        except Exception as e:
                            logger.error(f"Exception in statement execution: {str(e)}")
                            raise
                    logger.debug(f"special_method returning: {result}")
                    return result
            special_method.__self__ = instance
            return special_method
        else:
            # Regular async binding for normal methods
            async def method(*args: Any, **kwargs: Any) -> Any:
                return await self(instance, *args, **kwargs)
            method.__self__ = instance
            return method