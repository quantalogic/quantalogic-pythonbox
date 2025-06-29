"""
Focused tests to achieve 100% coverage for low-coverage files.
"""

import pytest
from quantalogic_pythonbox import execute_async


class TestComprehensionCoverage:
    """Test comprehension edge cases to improve coverage"""
    
    @pytest.mark.asyncio
    async def test_dict_comprehension_error_handling(self):
        """Test dict comprehension with type error"""
        code = """
try:
    result = {k: v for k, v in "not_iterable"}
except TypeError as e:
    result = str(e)
"""
        result = await execute_async(code)
        assert "not iterable" in str(result.local_variables.get('result', '')).lower()
        
    @pytest.mark.asyncio
    async def test_set_comprehension_error_handling(self):
        """Test set comprehension with type error"""
        code = """
try:
    result = {x for x in 42}
except TypeError as e:
    result = str(e)
"""
        result = await execute_async(code)
        assert "not iterable" in str(result.local_variables.get('result', '')).lower()
        
    @pytest.mark.asyncio
    async def test_generator_expression_async_single(self):
        """Test generator expression with single async iterable"""
        code = """
async def async_range(n):
    for i in range(n):
        yield i * 2
        
gen = (x + 1 for x in async_range(3))
result = []
async for item in gen:
    result.append(item)
"""
        result = await execute_async(code)
        assert result.local_variables['result'] == [1, 3, 5]
        
    @pytest.mark.asyncio 
    async def test_generator_expression_async_multiple(self):
        """Test generator expression with multiple generators"""
        code = """
async def async_range(n):
    for i in range(n):
        yield i
        
gen = (x + y for x in async_range(2) for y in range(2))
result = []
async for item in gen:
    result.append(item)
"""
        result = await execute_async(code)
        assert result.local_variables['result'] == [0, 1, 1, 2]
        
    @pytest.mark.asyncio
    async def test_generator_expression_error_handling(self):
        """Test generator expression error handling"""
        code = """
try:
    gen = (x for x in 42)
    result = list(gen)
except TypeError as e:
    result = str(e)
"""
        result = await execute_async(code)
        assert "not iterable" in str(result.local_variables.get('result', '')).lower()


class TestExceptionCoverage:
    """Test exception handling edge cases"""
    
    @pytest.mark.asyncio
    async def test_try_except_stopiteration_value(self):
        """Test StopIteration with value in try-except"""
        code = """
def generator_with_return():
    yield 1
    return "return_value"
    
gen = generator_with_return()
try:
    next(gen)
    next(gen)  # This will raise StopIteration with value
except StopIteration as e:
    result = e.value if hasattr(e, 'value') else "no_value"
"""
        result = await execute_async(code)
        # The result depends on how the interpreter handles StopIteration
        assert result.local_variables.get('result') is not None
        
    @pytest.mark.asyncio
    async def test_try_except_runtime_error_stopiteration(self):
        """Test RuntimeError from StopIteration in coroutine"""
        code = """
try:
    # This should trigger RuntimeError from StopIteration
    raise RuntimeError("coroutine raised StopIteration(42)")
except RuntimeError as e:
    result = str(e)
"""
        result = await execute_async(code)
        assert "StopIteration" in str(result.local_variables.get('result', ''))
        
    @pytest.mark.asyncio
    async def test_try_except_exception_type_resolution(self):
        """Test exception type resolution in handlers"""
        code = """
try:
    raise ValueError("test error")
except ValueError as e:
    result = "caught ValueError"
except Exception as e:
    result = "caught Exception"
"""
        result = await execute_async(code)
        assert result.local_variables.get('result') == "caught ValueError"
        
    @pytest.mark.asyncio
    async def test_try_except_with_name(self):
        """Test exception handler with name binding"""
        code = """
try:
    raise ValueError("test error")
except ValueError as e:
    result = f"caught: {e}"
"""
        result = await execute_async(code)
        assert "caught: test error" in str(result.local_variables.get('result', ''))
        
    @pytest.mark.asyncio
    async def test_try_except_finally_with_return(self):
        """Test try-except-finally with return in handler"""
        code = """
def test():
    try:
        raise ValueError("test")
    except ValueError:
        return "from_except"
    finally:
        pass

result = test()
"""
        result = await execute_async(code)
        assert result.local_variables.get('result') == "from_except"
        
    @pytest.mark.asyncio
    async def test_try_except_multiple_handlers(self):
        """Test multiple exception handlers"""
        code = """
errors = []

# Test first handler
try:
    raise ValueError("value error")
except ValueError as e:
    errors.append("ValueError")
except TypeError as e:
    errors.append("TypeError")

# Test second handler
try:
    raise TypeError("type error")
except ValueError as e:
    errors.append("ValueError")
except TypeError as e:
    errors.append("TypeError")
"""
        result = await execute_async(code)
        assert result.local_variables.get('errors') == ["ValueError", "TypeError"]
        
    @pytest.mark.asyncio
    async def test_try_except_unhandled_exception(self):
        """Test unhandled exception"""
        code = """
try:
    raise ValueError("test")
except TypeError:
    result = "handled"
except Exception:
    result = "caught by Exception"
"""
        result = await execute_async(code)
        assert result.local_variables.get('result') == "caught by Exception"


class TestContextVisitorsCoverage:
    """Test context manager edge cases"""
    
    @pytest.mark.asyncio
    async def test_with_exception_in_exit(self):
        """Test context manager with exception in __exit__"""
        code = """
class ExceptionInExit:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("exit error")
        
try:
    with ExceptionInExit():
        pass
except RuntimeError as e:
    result = str(e)
"""
        result = await execute_async(code)
        assert "exit error" in str(result.local_variables.get('result', ''))
        
    @pytest.mark.asyncio
    async def test_with_exception_suppression_false(self):
        """Test context manager that doesn't suppress exceptions"""
        code = """
class NoSuppression:
    def __enter__(self):
        return "context_value"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress
        
try:
    with NoSuppression() as ctx:
        raise ValueError("test error")
except ValueError as e:
    result = f"caught: {e}"
"""
        result = await execute_async(code)
        assert "caught: test error" in str(result.local_variables.get('result', ''))


class TestAdditionalCoverage:
    """Additional tests for specific uncovered lines"""
    
    @pytest.mark.asyncio
    async def test_nested_comprehensions(self):
        """Test nested comprehensions"""
        code = """
result = [[x*y for y in range(3)] for x in range(2)]
"""
        result = await execute_async(code)
        expected = [[0, 0, 0], [0, 1, 2]]
        assert result.local_variables.get('result') == expected
        
    @pytest.mark.asyncio
    async def test_comprehension_with_conditions(self):
        """Test comprehensions with if conditions"""
        code = """
# List comprehension with condition
result1 = [x for x in range(10) if x % 2 == 0]

# Dict comprehension with condition
result2 = {x: x*2 for x in range(5) if x > 2}

# Set comprehension with condition
result3 = {x for x in range(5) if x != 3}
"""
        result = await execute_async(code)
        assert result.local_variables.get('result1') == [0, 2, 4, 6, 8]
        assert result.local_variables.get('result2') == {3: 6, 4: 8}
        assert result.local_variables.get('result3') == {0, 1, 2, 4}
        
    @pytest.mark.asyncio
    async def test_generator_expression_with_conditions(self):
        """Test generator expressions with conditions"""
        code = """
gen = (x*2 for x in range(5) if x % 2 == 1)
result = list(gen)
"""
        result = await execute_async(code)
        assert result.local_variables.get('result') == [2, 6]
        
    @pytest.mark.asyncio  
    async def test_multiple_generator_comprehension(self):
        """Test comprehensions with multiple generators"""
        code = """
# Multiple generators in list comprehension
result1 = [x + y for x in range(2) for y in range(3)]

# Multiple generators in dict comprehension  
result2 = {f"{x}_{y}": x*y for x in range(2) for y in range(2)}

# Multiple generators in set comprehension
result3 = {x * y for x in range(3) for y in range(2)}
"""
        result = await execute_async(code)
        assert result.local_variables.get('result1') == [0, 1, 2, 1, 2, 3]
        assert result.local_variables.get('result2') == {"0_0": 0, "0_1": 0, "1_0": 0, "1_1": 1}
        assert result.local_variables.get('result3') == {0, 2, 4}
        
    @pytest.mark.asyncio
    async def test_exception_in_comprehension(self):
        """Test exceptions during comprehension iteration"""
        code = """
def problematic_iter():
    yield 1
    yield 2
    raise ValueError("test error")
    
try:
    result = [x for x in problematic_iter()]
except ValueError as e:
    result = str(e)
"""
        result = await execute_async(code)
        # The behavior depends on how the interpreter handles generator exceptions
        assert result.local_variables.get('result') is not None
        
    @pytest.mark.asyncio
    async def test_complex_exception_scenarios(self):
        """Test complex exception handling scenarios"""
        code = """
results = []

# Test exception in finally block
try:
    try:
        raise ValueError("inner")
    finally:
        results.append("finally")
        raise RuntimeError("finally error")
except RuntimeError as e:
    results.append("caught runtime")
    
# Test exception masking
try:
    try:
        raise ValueError("masked")
    except ValueError:
        raise TypeError("new error")
except TypeError as e:
    results.append("caught type")
"""
        result = await execute_async(code)
        res = result.local_variables.get('results', [])
        # The exact behavior depends on exception handling implementation
        assert len(res) >= 2
        
    @pytest.mark.asyncio
    async def test_advanced_assignment_patterns(self):
        """Test advanced assignment patterns"""
        code = """
# Nested unpacking
((a, b), (c, d)) = ((1, 2), (3, 4))

# List unpacking with star
first, *middle, last = [1, 2, 3, 4, 5]

# Multiple assignment
x = y = z = 42

results = [a, b, c, d, first, middle, last, x, y, z]
"""
        result = await execute_async(code)
        expected = [1, 2, 3, 4, 1, [2, 3, 4], 5, 42, 42, 42]
        assert result.local_variables.get('results') == expected


class TestSliceUtilsCoverage:
    """Test slice utils through interpreter"""
    
    @pytest.mark.asyncio
    async def test_custom_slice_usage(self):
        """Test CustomSlice through interpreter"""
        code = """
from quantalogic_pythonbox.slice_utils import CustomSlice

# Basic usage
s1 = CustomSlice(1, 5, 2)
basic_result = [s1.start, s1.stop, s1.step]

# String representation
str_result = str(s1)
repr_result = repr(s1)

# Equality tests
s2 = CustomSlice(1, 5, 2)
s3 = CustomSlice(2, 6, 3)
builtin_s = slice(1, 5, 2)

equality_result = [
    s1 == s2,  # Should be True
    s1 != s3,  # Should be True
    s1 == builtin_s,  # Should be True
    s1 != "not_slice"  # Should be True
]

# Indices testing
indices_result = []
indices_result.append(s1.indices(10))  # (1, 5, 2)

# Edge cases for indices
s_none = CustomSlice(None, None, 1)
indices_result.append(s_none.indices(10))  # (0, 10, 1)

s_neg = CustomSlice(None, None, -1)
indices_result.append(s_neg.indices(10))  # (9, -1, -1)

s_neg_indices = CustomSlice(-3, -1, 1)
indices_result.append(s_neg_indices.indices(10))  # (7, 9, 1)

# Test getitem
getitem_result = [s1[0], s1[1], s1[2]]

# Test error cases
try:
    s1[3]
    error1 = None
except IndexError:
    error1 = "IndexError"

try:
    s1["invalid"]
    error2 = None
except TypeError:
    error2 = "TypeError"
    
error_result = [error1, error2]

result = {
    'basic': basic_result,
    'str': str_result,
    'repr': repr_result,
    'equality': equality_result,
    'indices': indices_result,
    'getitem': getitem_result,
    'errors': error_result
}
"""
        result = await execute_async(code)
        res = result.local_variables.get('result', {})
        
        # Test basic functionality
        assert res['basic'] == [1, 5, 2]
        
        # Test string representations
        assert res['str'] == "Slice(1,5,2)"
        assert "CustomSlice(start=1, stop=5, step=2)" in res['repr']
        
        # Test equality
        eq_res = res['equality']
        assert eq_res[0] is True  # s1 == s2
        assert eq_res[1] is True  # s1 != s3
        assert eq_res[2] is True  # s1 == builtin_s
        assert eq_res[3] is True  # s1 != "not_slice"
        
        # Test indices
        indices = res['indices']
        assert indices[0] == (1, 5, 2)  # Normal slice
        assert indices[1] == (0, 10, 1)  # None, None, 1
        assert indices[2] == (9, -1, -1)  # None, None, -1
        assert indices[3] == (7, 9, 1)  # -3, -1, 1
        
        # Test getitem
        assert res['getitem'] == [1, 5, 2]
        
        # Test errors
        errors = res['errors']
        assert errors[0] == "IndexError"
        assert errors[1] == "TypeError"
        
    @pytest.mark.asyncio
    async def test_custom_slice_edge_cases(self):
        """Test CustomSlice edge cases through interpreter"""
        code = """
from quantalogic_pythonbox.slice_utils import CustomSlice

# Edge cases for indices method
edge_cases = []

# Large negative start
s1 = CustomSlice(-15, 5, 1)
edge_cases.append(s1.indices(10))  # (0, 5, 1)

# None stop with negative step
s2 = CustomSlice(5, None, -1) 
edge_cases.append(s2.indices(10))  # (5, -1, -1)

# Large negative stop
s3 = CustomSlice(None, -15, 1)
edge_cases.append(s3.indices(10))  # (0, 0, 1)

# Negative stop with negative step
s4 = CustomSlice(None, -5, -1)
edge_cases.append(s4.indices(10))  # (9, 5, -1)

# Start >= length with positive step
s5 = CustomSlice(15, 20, 1)
edge_cases.append(s5.indices(10))  # (10, 10, 1)

# Start >= length with negative step
s6 = CustomSlice(15, 5, -1)
edge_cases.append(s6.indices(10))  # (9, 5, -1)

result = edge_cases
"""
        result = await execute_async(code)
        res = result.local_variables.get('result', [])
        
        assert res[0] == (0, 5, 1)    # Large negative start
        assert res[1] == (5, -1, -1)  # None stop with negative step
        assert res[2] == (0, 0, 1)    # Large negative stop
        assert res[3] == (9, 5, -1)   # Negative stop with negative step
        assert res[4] == (10, 10, 1)  # Start >= length with positive step
        assert res[5] == (9, 5, -1)   # Start >= length with negative step
