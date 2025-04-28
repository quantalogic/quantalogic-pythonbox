import ast
import logging
import threading
from typing import Any, Dict, List, Optional, Callable
import asyncio

import json
import math
import random
import re
import datetime
import time
import collections
import itertools
import functools
import operator
import typing
import fractions
import statistics
import array
import bisect
import heapq
import copy
import enum
import uuid

import psutil

from .exceptions import BreakException, ContinueException, ReturnException, WrappedException
from .scope import Scope


class ASTInterpreter:
    def __init__(
        self,
        allowed_modules: List[str],
        env_stack: Optional[List[Dict[str, Any]]] = None,
        source: Optional[str] = None,
        restrict_os: bool = True,
        namespace: Optional[Dict[str, Any]] = None,
        max_recursion_depth: int = 1000,
        max_operations: int = 10000000,
        max_memory_mb: int = 1024,
        sync_mode: bool = False,
        safe_builtins: Optional[Dict[str, Any]] = None,
        ignore_typing: bool = False
    ) -> None:
        self.allowed_modules: List[str] = allowed_modules
        self.modules: Dict[str, Any] = {mod: __import__(mod) for mod in allowed_modules}
        self.restrict_os: bool = restrict_os
        self.sync_mode: bool = sync_mode
        self.max_operations: int = max_operations
        self.operations_count: int = 0
        self.max_memory_mb: int = max_memory_mb
        self.process = psutil.Process()
        self.type_hints: Dict[str, Any] = {}
        self.special_methods: Dict[str, Callable] = {}
        self.ignore_typing: bool = ignore_typing
        
        self.generator_context = {
            'yield_queue': asyncio.Queue(),
            'sent_queue': asyncio.Queue(),
            'active': False,
            'yielded': False,
            'yield_value': None, 
            'yield_from': False,
            'yield_from_iterable': None,
            'sent_value': None
        }

        def raise_(exc):
            raise exc

        default_safe_builtins: Dict[str, Any] = {
            'None': None,
            'False': False,
            'True': True,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'bool': bool,
            'object': object,
            'range': range,
            'iter': iter,
            'next': next,
            'sorted': sorted,
            'frozenset': frozenset,
            'super': super,
            'len': len,
            'property': property,
            'staticmethod': staticmethod,
            'classmethod': classmethod,
            '__name__': '__main__',
            'round': round,
            'min': min,
            'max': max,
            'abs': abs,
            'sum': sum,
            'zip': zip,
            'map': map,
            'pow': pow,
            'divmod': divmod,
            'all': all,
            'any': any,
            'filter': filter,
            'enumerate': enumerate,
            'chr': chr,
            'ord': ord,
            'hex': hex,
            'bin': bin,
            'oct': oct,
            'isinstance': isinstance,
            'type': type,
            'dir': dir,
            'getattr': getattr,
            'setattr': setattr,
            'delattr': delattr,
            'callable': callable,
            'hasattr': hasattr,
            'hash': hash,
            'id': id,
            'repr': repr,
            'ascii': ascii,
            'format': format,
            'bytes': bytes,
            'bytearray': bytearray,
            'slice': slice,
            'complex': complex,
            'reversed': reversed,
            'input': lambda x='': "mocked_input",
            'eval': lambda *args: raise_(ValueError("eval is not allowed")),
            'exec': lambda *args: raise_(ValueError("exec is not allowed")),
            'BaseException': BaseException,
            'Exception': Exception,
            'AttributeError': AttributeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'NameError': NameError,
            'ZeroDivisionError': ZeroDivisionError,
            'OverflowError': OverflowError,
            'RuntimeError': RuntimeError,
            'NotImplementedError': NotImplementedError,
            'StopIteration': StopIteration,
            'StopAsyncIteration': StopAsyncIteration,
            'GeneratorExit': GeneratorExit,
            'AssertionError': AssertionError,
            'json': json,
            'math': math,
            'random': random,
            're': re,
            'datetime': datetime,
            'time': time,
            'collections': collections,
            'itertools': itertools,
            'functools': functools,
            'operator': operator,
            'typing': typing,
            'fractions': fractions,
            'statistics': statistics,
            'array': array,
            'bisect': bisect,
            'heapq': heapq,
            'copy': copy,
            'enum': enum.Enum,
            'uuid': uuid,
        }

        self.safe_builtins = safe_builtins if safe_builtins is not None else default_safe_builtins

        if env_stack is None:
            self.env_stack: List[Dict[str, Any]] = [{}]
            self.env_stack[0].update(self.modules)

            allowed_builtins = {
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "str": str,
                "repr": repr,
                "id": id,
                "Exception": Exception,
                "ZeroDivisionError": ZeroDivisionError,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "print": print,
                "getattr": self.safe_getattr,
                "vars": lambda obj=None: vars(obj) if obj else dict(self.env_stack[-1]),
            }

            self.safe_builtins.update(allowed_builtins)
            self.safe_builtins["__import__"] = self.safe_import

            self.env_stack[0]["__builtins__"] = self.safe_builtins
            self.env_stack[0].update(self.safe_builtins)
            self.env_stack[0]["logger"] = logging.getLogger(__name__)

            if namespace is not None:
                self.env_stack[0].update(namespace)
        else:
            self.env_stack = env_stack
            if "__builtins__" not in self.env_stack[0]:
                allowed_builtins = {
                    "enumerate": enumerate,
                    "zip": zip,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "str": str,
                    "repr": repr,
                    "id": id,
                    "Exception": Exception,
                    "ZeroDivisionError": ZeroDivisionError,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "print": print,
                    "getattr": self.safe_getattr,
                    "vars": lambda obj=None: vars(obj) if obj else dict(self.env_stack[-1]),
                }

                self.safe_builtins.update(allowed_builtins)
                self.safe_builtins["__import__"] = self.safe_import

                self.env_stack[0]["__builtins__"] = self.safe_builtins
                self.env_stack[0].update(self.safe_builtins)
                self.env_stack[0]["logger"] = logging.getLogger(__name__)

            if namespace is not None:
                self.env_stack[0].update(namespace)

        if self.restrict_os:
            os_related_modules = {"os", "sys", "subprocess", "shutil", "platform"}
            for mod in os_related_modules:
                if mod in self.modules:
                    del self.modules[mod]
            for mod in list(self.allowed_modules):
                if mod in os_related_modules:
                    self.allowed_modules.remove(mod)

        if 'time' in self.modules:
            self.modules['time'].sleep = lambda x: None

        self.source_lines: Optional[List[str]] = source.splitlines() if source else None
        self.var_cache: Dict[str, Any] = {}
        self.recursion_depth: int = 0
        self.max_recursion_depth: int = max_recursion_depth
        self.loop = None
        self.current_class = None
        self.current_instance = None
        self.current_exception = None
        self.last_exception = None
        self.lock = threading.Lock()

        if "decimal" in self.modules:
            dec = self.modules["decimal"]
            self.env_stack[0]["Decimal"] = dec.Decimal
            self.env_stack[0]["getcontext"] = dec.getcontext
            self.env_stack[0]["setcontext"] = dec.setcontext
            self.env_stack[0]["localcontext"] = dec.localcontext
            self.env_stack[0]["Context"] = dec.Context

    def initialize_visitors(self):
        """Attach visitor methods to the interpreter instance."""
        from . import visit_handlers
        for handler_name in visit_handlers.__all__:
            handler = getattr(visit_handlers, handler_name)
            setattr(self, handler_name, handler.__get__(self, ASTInterpreter))

    def safe_import(self, name: str, globals=None, locals=None, fromlist=(), level=0) -> Any:
        self.env_stack[0]['logger'].debug("Attempting import of module '%s', allowed modules: %s" % (name, self.allowed_modules))
        os_related_modules = {"os", "sys", "subprocess", "shutil", "platform"}
        if self.restrict_os and name in os_related_modules:
            self.env_stack[0]['logger'].debug("Module '%s' blocked due to OS restriction" % name)
            raise ImportError("Module '%s' is blocked due to OS restriction." % name)
        if name not in self.allowed_modules:
            self.env_stack[0]['logger'].debug("Module '%s' not in allowed modules, raising ImportError" % name)
            raise ImportError("Module '%s' is not allowed. Only %s are permitted." % (name, self.allowed_modules))
        return self.modules[name]

    def safe_getattr(self, obj: Any, name: str, default: Any = None) -> Any:
        if name.startswith('__') and name.endswith('__') and name not in ['__init__', '__call__']:
            raise AttributeError("Access to dunder attribute '%s' is restricted." % name)
        return getattr(obj, name, default)

    def spawn_from_env(self, env_stack: List[Dict[str, Any]]) -> "ASTInterpreter":
        new_interp = ASTInterpreter(
            self.allowed_modules,
            env_stack,
            source="\n".join(self.source_lines) if self.source_lines else None,
            restrict_os=self.restrict_os,
            max_recursion_depth=self.max_recursion_depth,
            max_operations=self.max_operations,
            max_memory_mb=self.max_memory_mb,
            sync_mode=self.sync_mode,
            safe_builtins=self.safe_builtins,
            ignore_typing=self.ignore_typing
        )
        new_interp.loop = self.loop
        new_interp.var_cache = self.var_cache.copy()
        new_interp.initialize_visitors()
        return new_interp

    def get_variable(self, name: str) -> Any:
        with self.lock:
            if name in self.var_cache:
                return self.var_cache[name]
            for frame in reversed(self.env_stack):
                if name in frame:
                    self.var_cache[name] = frame[name]
                    return frame[name]
            raise NameError("Name '%s' is not defined." % name)

    def set_variable(self, name: str, value: Any) -> None:
        with self.lock:
            if "__global_names__" in self.env_stack[-1] and name in self.env_stack[-1]["__global_names__"]:
                self.env_stack[0][name] = value
                if name in self.var_cache:
                    del self.var_cache[name]
            elif "__nonlocal_names__" in self.env_stack[-1] and name in self.env_stack[-1]["__nonlocal_names__"]:
                for frame in reversed(self.env_stack[:-1]):
                    if name in frame:
                        frame[name] = value
                        if name in self.var_cache:
                            del self.var_cache[name]
                        return
                raise NameError("Nonlocal name '%s' not found in outer scope" % name)
            else:
                self.env_stack[-1][name] = value
                if name in self.var_cache:
                    del self.var_cache[name]

    async def assign(self, target: ast.AST, value: Any, wrap_exceptions: bool = True) -> None:
        self.env_stack[0]['logger'].debug(f"Assigning value {value} of type {type(value)} to target of type {type(target)}")
        if isinstance(target, ast.Name):
            self.set_variable(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            star_index = None
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Starred):
                    if star_index is not None:
                        raise Exception("Multiple starred expressions not supported")
                    star_index = i
            if star_index is None:
                if len(target.elts) != len(value):
                    raise ValueError("not enough values to unpack (expected %d, got %d)" % (len(target.elts), len(value)))
                for t, v in zip(target.elts, value):
                    await self.assign(t, v)
            else:
                total = len(value)
                before = target.elts[:star_index]
                after = target.elts[star_index + 1:]
                if len(before) + len(after) > total:
                    raise ValueError("not enough values to unpack (expected at least %d, got %d)" % (len(before) + len(after), total))
                for i, elt2 in enumerate(before):
                    await self.assign(elt2, value[i])
                starred_count = total - len(before) - len(after)
                await self.assign(target.elts[star_index].value, value[len(before):len(before) + starred_count])
                for j, elt2 in enumerate(after):
                    await self.assign(elt2, value[len(before) + starred_count + j])
        elif isinstance(target, ast.Attribute):
            obj = await self.visit(target.value, wrap_exceptions=True)
            prop = getattr(type(obj), target.attr, None)
            if isinstance(prop, property) and prop.fset:
                from .execution_utils import execute_function
                await execute_function(prop.fset, [obj, value], {})
            else:
                setattr(obj, target.attr, value)
        elif isinstance(target, ast.Subscript):
            obj = await self.visit(target.value, wrap_exceptions=True)
            key = await self.visit(target.slice, wrap_exceptions=True)
            obj[key] = value
        else:
            raise Exception("Unsupported assignment target type: " + str(type(target)))

    async def visit(self, node: ast.AST, is_await_context: bool = False, wrap_exceptions: bool = True) -> Any:
        self.operations_count += 1
        if self.operations_count > self.max_operations:
            raise RuntimeError("Exceeded maximum operations (%d)" % self.max_operations)
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        if memory_usage > self.max_memory_mb:
            raise MemoryError("Memory usage exceeded limit (%d MB)" % self.max_memory_mb)
        
        self.recursion_depth += 1
        if self.recursion_depth > self.max_recursion_depth:
            raise RecursionError("Maximum recursion depth exceeded (%d)" % self.max_recursion_depth)
        
        method_name: str = "visit_" + node.__class__.__name__
        method = getattr(self, method_name, self.generic_visit)
        self.env_stack[0]["logger"].debug("Visiting %s at line %s" % (method_name, getattr(node, 'lineno', 'unknown')))
        
        try:
            if self.sync_mode:
                from .execution_utils import sync_call
                func = method.__func__ if hasattr(method, '__func__') else method
                if method_name == "visit_Call":
                    result = sync_call(func, self, node, is_await_context, wrap_exceptions)
                else:
                    result = sync_call(func, self, node, wrap_exceptions=wrap_exceptions)
            elif method_name == "visit_Call":
                result = await method(node, is_await_context, wrap_exceptions)
            else:
                result = await method(node, wrap_exceptions=wrap_exceptions)
            self.recursion_depth -= 1
            return result
        except (ReturnException, BreakException, ContinueException):
            self.recursion_depth -= 1
            raise
        except Exception as e:
            self.recursion_depth -= 1
            if not wrap_exceptions:
                raise
            lineno = getattr(node, "lineno", None) or 1
            col = getattr(node, "col_offset", None) or 0
            context_line = self.source_lines[lineno - 1] if self.source_lines and 1 <= lineno <= len(self.source_lines) else ""
            raise WrappedException(
                "Error line %d, col %d:\n%s\nDescription: %s" % (lineno, col, context_line, str(e)), e, lineno, col, context_line
            ) from e

    async def generic_visit(self, node: ast.AST, wrap_exceptions: bool = True) -> Any:
        lineno = getattr(node, "lineno", None) or 1
        context_line = self.source_lines[lineno - 1] if self.source_lines and 1 <= lineno <= len(self.source_lines) else ""
        raise Exception(
            "Unsupported AST node type: %s at line %d.\nContext: %s" % (node.__class__.__name__, lineno, context_line)
        )

    async def execute_async(self, node: ast.Module, entry_point: str = None) -> Any:
        self.env_stack[0]['logger'].debug("Starting execute_async visit")
        self.env_stack[0]['logger'].debug(f"Parsed AST nodes: {[type(n).__name__ for n in ast.walk(node)]}")
        if entry_point:
            await self.visit(node)  # Ensure all top-level definitions are processed
            func = self.get_variable(entry_point)
            if func:
                result = await func()
            else:
                raise ValueError(f"Entry point '{entry_point}' not defined.")
        else:
            result = await self.visit(node)
        local_vars = {k: v for k, v in self.env_stack[-1].items() if not k.startswith('__')}
        self.env_stack[0]['logger'].debug(f"Returning from execute_async: result type {type(result).__name__}, value {result}")
        return (result, local_vars)

    def find_function_node(self, node: ast.Module, func_name: str) -> Optional[ast.AsyncFunctionDef]:
        for stmt in node.body:
            if isinstance(stmt, ast.AsyncFunctionDef) and stmt.name == func_name:
                return stmt
        return None

    async def visit_AugAssign(self, node: ast.AugAssign, wrap_exceptions: bool = True) -> Any:
        value = await self.visit(node.value, wrap_exceptions=wrap_exceptions)
        target = node.target
        if isinstance(target, ast.Name):
            current = self.get_variable(target.id)
            new_value = self._apply_binary_op(current, node.op, value)
            self.set_variable(target.id, new_value)
        elif isinstance(target, ast.Attribute):
            obj = await self.visit(target.value, wrap_exceptions=wrap_exceptions)
            current = getattr(obj, target.attr)
            new_value = self._apply_binary_op(current, node.op, value)
            setattr(obj, target.attr, new_value)
        elif isinstance(target, ast.Subscript):
            obj = await self.visit(target.value, wrap_exceptions=wrap_exceptions)
            key = await self.visit(target.slice, wrap_exceptions=wrap_exceptions)
            current = obj[key]
            new_value = self._apply_binary_op(current, node.op, value)
            obj[key] = new_value
        else:
            raise Exception("Unsupported target for AugAssign: %s" % type(target))
        return None

    def _apply_binary_op(self, left, op, right):
        import operator as _operator
        ops = {
            ast.Add: _operator.add,
            ast.Sub: _operator.sub,
            ast.Mult: _operator.mul,
            ast.Div: _operator.truediv,
            ast.FloorDiv: _operator.floordiv,
            ast.Mod: _operator.mod,
            ast.Pow: _operator.pow,
            ast.LShift: _operator.lshift,
            ast.RShift: _operator.rshift,
            ast.BitOr: _operator.or_,
            ast.BitXor: _operator.xor,
            ast.BitAnd: _operator.and_,
        }
        for ast_op, func in ops.items():
            if isinstance(op, ast_op):
                return func(left, right)
        try:
            return left + right
        except Exception:
            raise Exception("Unsupported operator in AugAssign: %s" % op)

    async def visit_Name(self, node: ast.Name, wrap_exceptions: bool = True) -> Any:
        logger = self.env_stack[0]['logger']
        if node.id == 'kwonly_params':
            logger.debug(f"Query for 'kwonly_params' in context: env_stack depth {len(self.env_stack)}, node line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
        value = self.env_stack[-1].get(node.id, None)
        logger.debug(f"Visiting name: {node.id}, retrieved value: {value}, type: {type(value) if value is not None else 'NoneType'}, wrap_exceptions: {wrap_exceptions}")
        if value is None and wrap_exceptions:
            raise NameError(f"name '{node.id}' is not defined")
        return value

    async def visit_Import(self, node: ast.Import, wrap_exceptions: bool = True) -> Any:
        self.env_stack[0]['logger'].debug("Handling import of modules: %s" % str([alias.name for alias in node.names]))
        for alias in node.names:
            try:
                module = self.safe_import(alias.name)
                if alias.asname:
                    self.set_variable(alias.asname, module)
                else:
                    self.set_variable(alias.name, module)
            except ImportError as e:
                self.env_stack[0]['logger'].debug("Import failed for %s: %s" % (alias.name, str(e)))
                if wrap_exceptions:
                    raise WrappedException("Import error: %s" % str(e), e) from e
                else:
                    raise
        return None

    async def visit_Try(self, node: ast.Try, wrap_exceptions: bool = True) -> Any:
        self.env_stack[0]['logger'].debug("Entering visit_Try for node at line " + str(getattr(node, 'lineno', 'unknown')))
        try:
            self.env_stack[0]['logger'].debug("Executing try block")
            for stmt in node.body:
                stmt_result = await self.visit(stmt, wrap_exceptions=False)
                self.env_stack[0]['logger'].debug("Executed statement in try block: " + str(stmt.__class__.__name__) + ", result: " + str(stmt_result))
        except Exception as e:
            self.env_stack[0]['logger'].debug("Exception caught in try block: type " + str(type(e).__name__) + ", message: " + str(e))
            matched = False
            for handler in node.handlers:
                if handler.type:
                    try:
                        exc_type = await self._resolve_exception_type(handler.type)
                    except Exception as type_err:
                        self.env_stack[0]['logger'].debug(f"Error resolving exception type: {type_err}")
                        continue  # Skip this handler if type resolution fails
                    self.env_stack[0]['logger'].debug("Checking handler for exception type: " + str(exc_type) + ", caught exception type: " + str(type(e).__name__))
                    self.env_stack[0]['logger'].debug(f"Caught exception type: {type(e).__name__}, handler exception type: {exc_type}")
                    self.env_stack[0]['logger'].debug(f"Checking if exception {e} is instance of {exc_type} or its cause if wrapped")
                    self.env_stack[0]['logger'].debug(f"Resolved exc_type: {exc_type}, type of exc_type: {type(exc_type)}")
                    self.env_stack[0]['logger'].debug(f"Caught exception: {e}, type of caught exception: {type(e)}")
                    if isinstance(e, exc_type) or (isinstance(e, WrappedException) and isinstance(e.__cause__, exc_type)):
                        self.env_stack[0]['logger'].debug("Exception matched, entering except block for handler at line " + str(getattr(handler, 'lineno', 'unknown')))
                        if handler.name:
                            self.set_variable(handler.name, e)
                        for stmt in handler.body:
                            stmt_result = await self.visit(stmt, wrap_exceptions=False)
                            self.env_stack[0]['logger'].debug("Executed statement in except block: " + str(stmt.__class__.__name__) + ", result: " + str(stmt_result))
                            if isinstance(stmt, ast.Return):
                                value = await self.visit(stmt.value, wrap_exceptions=wrap_exceptions) if stmt.value else None
                                raise ReturnException(value)
                        matched = True
                        break
                else:  # Bare except
                    self.env_stack[0]['logger'].debug("Entering bare except block")
                    if handler.name:
                        self.set_variable(handler.name, e)
                    for stmt in handler.body:
                        stmt_result = await self.visit(stmt, wrap_exceptions=False)
                        self.env_stack[0]['logger'].debug("Executed statement in bare except block: " + str(stmt.__class__.__name__) + ", result: " + str(stmt_result))
                        if isinstance(stmt, ast.Return):
                            value = await self.visit(stmt.value, wrap_exceptions=wrap_exceptions) if stmt.value else None
                            raise ReturnException(value)
                    matched = True
                    break
            if not matched:
                self.env_stack[0]['logger'].debug("No handler matched, re-raising exception: " + str(type(e).__name__))
                raise e
        else:
            self.env_stack[0]['logger'].debug("Try block completed without exceptions, executing orelse if present")
            if node.orelse:
                for stmt in node.orelse:
                    stmt_result = await self.visit(stmt, wrap_exceptions=wrap_exceptions)
                    self.env_stack[0]['logger'].debug("Executed statement in orelse block: " + str(stmt.__class__.__name__) + ", result: " + str(stmt_result))
                    if isinstance(stmt, ast.Return):
                        value = await self.visit(stmt.value, wrap_exceptions=wrap_exceptions) if stmt.value else None
                        raise ReturnException(value)
        finally:
            self.env_stack[0]['logger'].debug("Executing finally block if present")
            if node.finalbody:
                for stmt in node.finalbody:
                    stmt_result = await self.visit(stmt, wrap_exceptions=wrap_exceptions)
                    self.env_stack[0]['logger'].debug("Executed statement in finally block: " + str(stmt.__class__.__name__) + ", result: " + str(stmt_result))
                    if isinstance(stmt, ast.Return):
                        value = await self.visit(stmt.value, wrap_exceptions=wrap_exceptions) if stmt.value else None
                        raise ReturnException(value)
        self.env_stack[0]['logger'].debug("Exiting visit_Try")
        return None

    def new_scope(self):
        return Scope(self.env_stack)

    async def _resolve_exception_type(self, node: Optional[ast.AST]) -> Any:
        if node is None:
            return Exception
        # If exception type is a simple name, try resolving from env or builtins
        if isinstance(node, ast.Name):
            name = node.id
            try:
                return self.get_variable(name)
            except Exception:
                import builtins
                if hasattr(builtins, name):
                    return getattr(builtins, name)
                # Unable to resolve, fall back
        return await self.visit(node, wrap_exceptions=True)

    async def run_sync_stmt(self, stmt: ast.stmt) -> Any:
        return await self.visit(stmt, wrap_exceptions=True)

    async def run_sync_expr(self, expr: ast.expr) -> Any:
        return await self.visit(expr, wrap_exceptions=True)

    async def visit_Tuple(self, node: ast.Tuple, wrap_exceptions: bool = True) -> tuple:
        self.env_stack[0]['logger'].debug(f"Evaluating Tuple with {len(node.elts)} elements")
        elements = [await self.visit(elt, wrap_exceptions=wrap_exceptions) for elt in node.elts]
        tuple_value = tuple(elements)
        self.env_stack[0]['logger'].debug(f"Tuple evaluated to {tuple_value}")
        return tuple_value

    async def visit_Await(self, node: ast.Await, wrap_exceptions: bool = True) -> Any:
        self.env_stack[0]['logger'].debug(f"Visiting Await at line {node.lineno if hasattr(node, 'lineno') else 'unknown'}")
        value = await self.visit(node.value, wrap_exceptions=True)
        try:
            result = await value
        except Exception as e:
            # Propagate StopAsyncIteration without wrapping to allow async generator return value handling
            if isinstance(e, StopAsyncIteration):
                raise
            if wrap_exceptions:
                raise RuntimeError(f"Error awaiting expression: {str(e)}") from e
            else:
                raise
        self.env_stack[0]['logger'].debug(f"Await evaluated to {result}")
        return result