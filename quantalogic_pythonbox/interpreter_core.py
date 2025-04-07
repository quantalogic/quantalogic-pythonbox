import ast
import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

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
            'active': False,
            'yielded': False,
            'yield_value': None, 
            'yield_from': False,
            'yield_from_iterable': None
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

        from . import visit_handlers
        for handler_name in visit_handlers.__all__:
            handler = getattr(visit_handlers, handler_name)
            setattr(self, handler_name, handler.__get__(self, ASTInterpreter))

    def safe_import(self, name: str, globals=None, locals=None, fromlist=(), level=0) -> Any:
        os_related_modules = {"os", "sys", "subprocess", "shutil", "platform"}
        if self.restrict_os and name in os_related_modules:
            raise ImportError(f"Import Error: Module '{name}' is blocked due to OS restriction.")
        if name not in self.allowed_modules:
            raise ImportError(f"Import Error: Module '{name}' is not allowed. Only {self.allowed_modules} are permitted.")
        return self.modules[name]

    def safe_getattr(self, obj: Any, name: str, default: Any = None) -> Any:
        if name.startswith('__') and name.endswith('__') and name not in ['__init__', '__call__']:
            raise AttributeError(f"Access to dunder attribute '{name}' is restricted.")
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
        return new_interp

    def get_variable(self, name: str) -> Any:
        with self.lock:
            if name in self.var_cache:
                return self.var_cache[name]
            for frame in reversed(self.env_stack):
                if name in frame:
                    self.var_cache[name] = frame[name]
                    return frame[name]
            raise NameError(f"Name '{name}' is not defined.")

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
                raise NameError(f"Nonlocal name '{name}' not found in outer scope")
            else:
                self.env_stack[-1][name] = value
                if name in self.var_cache:
                    del self.var_cache[name]

    async def assign(self, target: ast.AST, value: Any) -> None:
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
                    # Fix: Raise ValueError for unpacking mismatch
                    raise ValueError(f"not enough values to unpack (expected {len(target.elts)}, got {len(value)})")
                for t, v in zip(target.elts, value):
                    await self.assign(t, v)
            else:
                total = len(value)
                before = target.elts[:star_index]
                after = target.elts[star_index + 1:]
                if len(before) + len(after) > total:
                    # Fix: Raise ValueError for unpacking mismatch
                    raise ValueError(f"not enough values to unpack (expected at least {len(before) + len(after)}, got {total})")
                for i, elt2 in enumerate(before):
                    await self.assign(elt2, value[i])
                starred_count = total - len(before) - len(after)
                await self.assign(target.elts[star_index].value, value[len(before):len(before) + starred_count])
                for j, elt2 in enumerate(after):
                    await self.assign(elt2, value[len(before) + starred_count + j])
        elif isinstance(target, ast.Attribute):
            obj = await self.visit(target.value, wrap_exceptions=True)
            prop = getattr(type(obj), target.attr, None)
            # Fix: Use property setter if available
            if isinstance(prop, property) and prop.fset:
                await self._execute_function(prop.fset, [obj, value], {})
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
            raise RuntimeError(f"Exceeded maximum operations ({self.max_operations})")
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        if memory_usage > self.max_memory_mb:
            raise MemoryError(f"Memory usage exceeded limit ({self.max_memory_mb} MB)")
        
        self.recursion_depth += 1
        if self.recursion_depth > self.max_recursion_depth:
            raise RecursionError(f"Maximum recursion depth exceeded ({self.max_recursion_depth})")
        
        method_name: str = "visit_" + node.__class__.__name__
        method = getattr(self, method_name, self.generic_visit)
        self.env_stack[0]["logger"].debug(f"Visiting {method_name} at line {getattr(node, 'lineno', 'unknown')}")
        
        try:
            if self.sync_mode and not hasattr(node, 'await'):
                result = method(node, wrap_exceptions=wrap_exceptions)
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
                f"Error line {lineno}, col {col}:\n{context_line}\nDescription: {str(e)}", e, lineno, col, context_line
            ) from e

    async def generic_visit(self, node: ast.AST, wrap_exceptions: bool = True) -> Any:
        lineno = getattr(node, "lineno", None) or 1
        context_line = self.source_lines[lineno - 1] if self.source_lines and 1 <= lineno <= len(self.source_lines) else ""
        raise Exception(
            f"Unsupported AST node type: {node.__class__.__name__} at line {lineno}.\nContext: {context_line}"
        )

    async def visit_Module(self, node: ast.Module, wrap_exceptions: bool = True) -> Any:
        result = None
        for stmt in node.body:
            result = await self.visit(stmt, wrap_exceptions=wrap_exceptions)
        return result

    async def execute_async(self, node: ast.Module) -> Any:
        return await self.visit(node)

    def new_scope(self):
        return Scope(self.env_stack)

    async def _resolve_exception_type(self, node: Optional[ast.AST]) -> Any:
        if node is None:
            return Exception
        if isinstance(node, ast.Name):
            exc_type = self.get_variable(node.id)
            if exc_type in (Exception, ZeroDivisionError, ValueError, TypeError):
                return exc_type
            return exc_type
        if isinstance(node, ast.Call):
            return await self.visit(node, wrap_exceptions=True)
        return None

    async def _create_class_instance(self, cls: type, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        if isinstance(instance, cls) and hasattr(cls, '__init__'):
            init_method = cls.__init__
            await self._execute_function(init_method, [instance] + list(args), kwargs)
        return instance

    async def _execute_function(self, func: Any, args: List[Any], kwargs: Dict[str, Any]):
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        elif callable(func):
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        raise TypeError(f"Object {func} is not callable")


class AsyncFunction:
    async def __call__(self, *args: Any, _return_locals: bool = False, **kwargs: Any) -> Any:
        pass


class AsyncExecutionResult:
    def __init__(self, result: Any, error: Optional[str], execution_time: float, local_variables: Optional[Dict[str, Any]]):
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.local_variables = local_variables


class EventLoopManager:
    def __init__(self):
        self.loop = asyncio.get_event_loop()

    async def run_task(self, coro, timeout=None):
        return await asyncio.wait_for(coro, timeout=timeout)


event_loop_manager = EventLoopManager()
timeout = 60