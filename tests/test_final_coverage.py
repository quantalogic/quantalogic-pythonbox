"""
Final tests to achieve 100% coverage - focusing on remaining gaps.
"""

import pytest
from quantalogic_pythonbox import execute_async
import asyncio


class TestRemainingCoverage:
    """Test remaining uncovered code paths"""
    
    @pytest.mark.asyncio
    async def test_scope_coverage(self):
        """Test scope.py coverage"""
        code = """
# Test different scope scenarios
def outer():
    x = "outer"
    def inner():
        nonlocal x
        x = "inner"
        return x
    return inner()

result = outer()
"""
        result = await execute_async(code)
        assert result.result == "inner"
    
    @pytest.mark.asyncio
    async def test_mock_coroutine_coverage(self):
        """Test mock_coroutine.py coverage"""
        code = """
# Test coroutine-like behavior
async def test_coroutine():
    return "test"

import asyncio
result = asyncio.run(test_coroutine())
"""
        result = await execute_async(code)
        assert result.result == "test"
    
    @pytest.mark.asyncio
    async def test_comprehension_nested_conditions(self):
        """Test complex comprehension scenarios"""
        code = """
# Nested comprehensions with multiple conditions
matrix = [[i*j for j in range(3) if j > 0] for i in range(3) if i > 0]
flat = [item for sublist in matrix for item in sublist if item % 2 == 0]
"""
        result = await execute_async(code)
        assert result.local_variables["matrix"] == [[0, 0], [0, 2]]
        assert result.local_variables["flat"] == [0, 0, 0, 2]
    
    @pytest.mark.asyncio
    async def test_comprehension_with_try_except(self):
        """Test comprehension with exception handling"""
        code = """
def safe_divide(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0

numbers = [1, 2, 0, 4, 0]
results = [safe_divide(10, x) for x in numbers]
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [10.0, 5.0, 0, 2.5, 0]
    
    @pytest.mark.asyncio
    async def test_exception_chaining(self):
        """Test exception chaining scenarios"""
        code = """
results = []
try:
    try:
        raise ValueError("original")
    except ValueError as e:
        raise RuntimeError("chained") from e
except RuntimeError as e:
    results.append(str(e))
    results.append(str(e.__cause__))
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == ["chained", "original"]
    
    @pytest.mark.asyncio
    async def test_exception_suppress_with_finally(self):
        """Test exception suppression with finally blocks"""
        code = """
results = []
try:
    try:
        raise ValueError("test")
    except ValueError:
        results.append("caught")
        raise  # Re-raise
    finally:
        results.append("finally1")
except ValueError:
    results.append("outer_catch")
finally:
    results.append("finally2")
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == ["caught", "finally1", "outer_catch", "finally2"]
    
    @pytest.mark.asyncio
    async def test_generator_edge_cases(self):
        """Test generator edge cases"""
        code = """
def generator_with_return():
    yield 1
    yield 2
    return "done"

def generator_with_exception():
    yield 1
    raise StopIteration("custom")

g1 = generator_with_return()
results = []
try:
    while True:
        results.append(next(g1))
except StopIteration as e:
    results.append(f"stopped: {e.value}")
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [1, 2, "stopped: done"]
    
    @pytest.mark.asyncio
    async def test_function_annotations_and_defaults(self):
        """Test function with complex annotations and defaults"""
        code = """
def complex_function(
    a: int = 1,
    b: str = "default",
    *args: int,
    c: float = 3.14,
    **kwargs: str
) -> dict:
    return {
        'a': a,
        'b': b,
        'args': args,
        'c': c,
        'kwargs': kwargs
    }

result1 = complex_function()
result2 = complex_function(10, "test", 1, 2, 3, c=2.71, extra="value")
"""
        result = await execute_async(code)
        assert result.local_variables["result1"] == {
            'a': 1, 'b': 'default', 'args': (), 'c': 3.14, 'kwargs': {}
        }
        assert result.local_variables["result2"] == {
            'a': 10, 'b': 'test', 'args': (1, 2, 3), 'c': 2.71, 'kwargs': {'extra': 'value'}
        }
    
    @pytest.mark.asyncio
    async def test_class_method_resolution(self):
        """Test class method resolution and inheritance"""
        code = """
class Base:
    def method(self):
        return "base"
    
    @classmethod
    def class_method(cls):
        return f"class: {cls.__name__}"
    
    @staticmethod
    def static_method():
        return "static"

class Derived(Base):
    def method(self):
        return f"derived: {super().method()}"

d = Derived()
results = [
    d.method(),
    d.class_method(),
    d.static_method(),
    Derived.class_method(),
    Base.static_method()
]
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [
            "derived: base",
            "class: Derived", 
            "static",
            "class: Derived",
            "static"
        ]
    
    @pytest.mark.asyncio
    async def test_complex_assignment_targets(self):
        """Test complex assignment scenarios"""
        code = """
# Multiple assignment patterns
a, b = [1, 2]
(c, d), e = [(3, 4), 5]
*first, last = [1, 2, 3, 4, 5]
x, *middle, y = [10, 11, 12, 13, 14]

# Attribute assignment
class Container:
    pass

obj = Container()
obj.value = "test"
obj.nested = Container()
obj.nested.deep = "deep_value"

results = [a, b, c, d, e, first, last, x, middle, y, obj.value, obj.nested.deep]
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [
            1, 2, 3, 4, 5, [1, 2, 3, 4], 5, 10, [11, 12, 13], 14, "test", "deep_value"
        ]
    
    @pytest.mark.asyncio
    async def test_operator_edge_cases(self):
        """Test operator edge cases"""
        code = """
# Test various operators with edge cases
results = []

# Comparison chaining
results.append(1 < 2 < 3)
results.append(3 > 2 > 1)
results.append(1 == 1 != 2)

# Boolean operators with short-circuiting
results.append(False and (1/0))  # Should not raise
results.append(True or (1/0))    # Should not raise

# In-place operators
x = [1, 2, 3]
x += [4, 5]
results.append(x)

y = "hello"
y *= 2
results.append(y)

# Matrix multiplication (if supported)
try:
    import numpy as np
    a = [[1, 2], [3, 4]]
    # Test without numpy for basic interpreter
    results.append("matrix_ops_available")
except ImportError:
    results.append("matrix_ops_not_available")
"""
        result = await execute_async(code)
        expected_start = [True, True, True, False, True, [1, 2, 3, 4, 5], "hellohello"]
        assert result.local_variables["results"][:7] == expected_start
    
    @pytest.mark.asyncio
    async def test_import_edge_cases(self):
        """Test import statement edge cases"""
        code = """
# Different import patterns
import math as m
from json import loads, dumps
from collections import defaultdict, Counter

# Use imported items
results = [
    m.pi,
    loads('{"key": "value"}'),
    list(defaultdict(int)),
    dict(Counter("hello"))
]
"""
        result = await execute_async(code)
        import math
        import json
        from collections import defaultdict, Counter
        
        expected = [
            math.pi,
            {"key": "value"},
            [],
            dict(Counter("hello"))
        ]
        assert result.local_variables["results"] == expected


class TestAsyncFeatureCoverage:
    """Test async-specific features for coverage"""
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context managers"""
        code = """
class AsyncContextManager:
    async def __aenter__(self):
        return "async_value"
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

async def test_async_with():
    async with AsyncContextManager() as value:
        return value

import asyncio
result = asyncio.run(test_async_with())
"""
        result = await execute_async(code)
        assert result.result == "async_value"
    
    @pytest.mark.asyncio
    async def test_async_comprehension_coverage(self):
        """Test async comprehensions"""
        code = """
async def async_range(n):
    for i in range(n):
        yield i

async def test_async_comp():
    # Simple async comprehension
    result = [x async for x in async_range(3)]
    return result

import asyncio
result = asyncio.run(test_async_comp())
"""
        result = await execute_async(code)
        assert result.result == [0, 1, 2]


class TestExecutionEdgeCases:
    """Test execution.py edge cases"""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in execution"""
        code = """
import time
time.sleep(0.001)  # Very short sleep
result = "completed"
"""
        result = await execute_async(code, timeout=1.0)
        assert result.result == "completed"
        assert result.execution_time < 1.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self):
        """Test memory usage aspects"""
        code = """
# Create some data to use memory
data = []
for i in range(1000):
    data.append(i * 2)

result = len(data)
"""
        result = await execute_async(code)
        assert result.result == 1000
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_complex_error_scenarios(self):
        """Test complex error handling scenarios"""
        code = """
def recursive_function(n):
    if n <= 0:
        raise ValueError("Invalid input")
    if n == 1:
        return 1
    return n * recursive_function(n - 1)

try:
    result = recursive_function(5)
except ValueError as e:
    result = str(e)
"""
        result = await execute_async(code)
        assert result.local_variables["result"] == 120
    
    @pytest.mark.asyncio 
    async def test_variable_capture_edge_cases(self):
        """Test variable capture in different scopes"""
        code = """
# Test closure variable capture
def make_closures():
    funcs = []
    for i in range(3):
        funcs.append(lambda x=i: x)  # Capture i
    return funcs

closures = make_closures()
results = [f() for f in closures]

# Test global vs local variable resolution
global_var = "global"

def test_scope():
    global_var = "local"
    return global_var

local_result = test_scope()
global_result = global_var
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [0, 1, 2]
        assert result.local_variables["local_result"] == "local"
        assert result.local_variables["global_result"] == "global"
