import pytest
from quantalogic_pythonbox import execute_async
from quantalogic_pythonbox.generator_wrapper import GeneratorWrapper


@pytest.mark.asyncio
async def test_generator_wrapper_comprehensive():
    """Test all methods of GeneratorWrapper"""
    source = '''
def test_generator():
    def gen():
        try:
            yield 1
            yield 2
            x = yield 3
            yield x * 2
            return "final"
        except GeneratorExit:
            return "closed"
        except ValueError as e:
            yield f"error: {e}"
    
    g = gen()
    results = []
    
    # Test __next__
    results.append(next(g))  # 1
    results.append(next(g))  # 2
    results.append(next(g))  # 3
    
    # Test send
    results.append(g.send(5))  # 10
    
    # Test throw
    try:
        results.append(g.throw(ValueError, "test error"))
    except StopIteration:
        pass
    
    return results
'''
    result = await execute_async(source, entry_point='test_generator')
    assert result.result == [1, 2, 3, 10]


def test_generator_wrapper_methods():
    """Test GeneratorWrapper methods directly"""
    def simple_gen():
        yield 1
        yield 2
        return "done"
    
    wrapper = GeneratorWrapper(simple_gen())
    
    # Test __iter__
    assert iter(wrapper) is wrapper
    
    # Test __next__
    assert next(wrapper) == 1
    assert next(wrapper) == 2
    
    # Test StopIteration with return value
    try:
        next(wrapper)
        assert False, "Should have raised StopIteration"
    except StopIteration as e:
        assert e.value == "done"
    
    # Test that generator is closed
    assert wrapper.closed is True
    assert wrapper.return_value == "done"
    
    # Test operations on closed generator
    try:
        next(wrapper)
        assert False, "Should have raised StopIteration"
    except StopIteration as e:
        assert e.value == "done"


def test_generator_wrapper_send():
    """Test GeneratorWrapper send method"""
    def echo_gen():
        x = yield 1
        y = yield x * 2
        return y + 1
    
    wrapper = GeneratorWrapper(echo_gen())
    
    # First next
    assert next(wrapper) == 1
    
    # Send value
    assert wrapper.send(5) == 10
    
    # Send final value and get return
    try:
        wrapper.send(3)
        assert False, "Should have raised StopIteration"
    except StopIteration as e:
        assert e.value == 4


def test_generator_wrapper_throw():
    """Test GeneratorWrapper throw method"""
    def error_gen():
        try:
            yield 1
            yield 2
        except ValueError as e:
            yield f"caught: {e}"
        yield 3
    
    wrapper = GeneratorWrapper(error_gen())
    
    # Normal operation
    assert next(wrapper) == 1
    
    # Throw exception
    assert wrapper.throw(ValueError, "test error") == "caught: test error"
    
    # Continue after exception
    assert next(wrapper) == 3


def test_generator_wrapper_close():
    """Test GeneratorWrapper close behavior"""
    def closeable_gen():
        try:
            yield 1
            yield 2
        except GeneratorExit:
            return "closed properly"
        finally:
            pass
    
    wrapper = GeneratorWrapper(closeable_gen())
    
    # Get first value
    assert next(wrapper) == 1
    
    # Close the generator
    wrapper.close()
    
    # Should be closed now
    assert wrapper.closed is True
    
    # Further operations should raise StopIteration
    try:
        next(wrapper)
        assert False, "Should have raised StopIteration"
    except StopIteration:
        pass


@pytest.mark.asyncio
async def test_slice_edge_cases():
    """Test edge cases in slice handling"""
    source = '''
def test_slices():
    data = list(range(10))
    
    return [
        data[slice(None)],        # Full slice
        data[slice(2, None)],     # From index 2
        data[slice(None, 5)],     # Up to index 5
        data[slice(None, None, 2)], # Every second element
        data[slice(1, 8, 3)],     # Complex slice
        data[slice(-1, -5, -1)],  # Reverse slice
    ]
'''
    result = await execute_async(source, entry_point='test_slices')
    expected = [
        list(range(10)),           # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        list(range(2, 10)),        # [2, 3, 4, 5, 6, 7, 8, 9]
        list(range(5)),            # [0, 1, 2, 3, 4]
        list(range(0, 10, 2)),     # [0, 2, 4, 6, 8]
        list(range(1, 8, 3)),      # [1, 4, 7]
        list(range(9, 5, -1)),     # [9, 8, 7, 6]
    ]
    assert result.result == expected


@pytest.mark.asyncio
async def test_complex_assignments():
    """Test complex assignment patterns"""
    source = '''
def test_assignments():
    # Nested unpacking
    a, (b, c), d = [1, [2, 3], 4]
    
    # Multiple assignment
    x = y = z = 5
    
    # Augmented assignment with attributes
    class Obj:
        def __init__(self):
            self.value = 10
    
    obj = Obj()
    obj.value += 5
    
    # Augmented assignment with subscripts
    lst = [1, 2, 3]
    lst[1] *= 3
    
    return (a, b, c, d, x, y, z, obj.value, lst)
'''
    result = await execute_async(source, entry_point='test_assignments')
    assert result.result == (1, 2, 3, 4, 5, 5, 5, 15, [1, 6, 3])


@pytest.mark.asyncio
async def test_attribute_operations():
    """Test attribute operations including getattr, setattr, delattr"""
    source = '''
def test_attributes():
    class TestObj:
        def __init__(self):
            self.existing = "value"
    
    obj = TestObj()
    
    # getattr with default
    val1 = getattr(obj, 'existing', 'default')
    val2 = getattr(obj, 'missing', 'default')
    
    # setattr
    setattr(obj, 'new_attr', 'new_value')
    
    # hasattr
    has_existing = hasattr(obj, 'existing')
    has_new = hasattr(obj, 'new_attr')
    has_missing = hasattr(obj, 'missing')
    
    # delattr
    delattr(obj, 'existing')
    has_after_del = hasattr(obj, 'existing')
    
    return (val1, val2, obj.new_attr, has_existing, has_new, has_missing, has_after_del)
'''
    result = await execute_async(source, entry_point='test_attributes')
    assert result.result == ('value', 'default', 'new_value', True, True, False, False)


@pytest.mark.asyncio
async def test_advanced_exception_handling():
    """Test advanced exception handling patterns"""
    source = '''
def test_exceptions():
    results = []
    
    # Exception with multiple except blocks
    try:
        raise ValueError("test")
    except (TypeError, AttributeError):
        results.append("wrong")
    except ValueError as e:
        results.append(f"caught: {e}")
    except:
        results.append("fallback")
    
    # Re-raising exceptions
    try:
        try:
            raise RuntimeError("inner")
        except RuntimeError:
            raise ValueError("outer")
    except ValueError as e:
        results.append(f"reraised: {e}")
    
    # Exception with finally block that executes
    try:
        try:
            raise KeyError("key")
        finally:
            results.append("finally")
    except KeyError:
        results.append("after finally")
    
    return results
'''
    result = await execute_async(source, entry_point='test_exceptions')
    assert result.result == ["caught: test", "reraised: outer", "finally", "after finally"]


@pytest.mark.asyncio
async def test_lambda_edge_cases():
    """Test edge cases with lambda functions"""
    source = '''
def test_lambdas():
    # Lambda with default arguments
    f1 = lambda x, y=10: x + y
    
    # Lambda with *args and **kwargs
    f2 = lambda *args, **kwargs: (args, kwargs)
    
    # Nested lambdas
    f3 = lambda x: lambda y: x + y
    
    # Lambda in comprehension
    funcs = [lambda x, i=i: x + i for i in range(3)]
    
    return [
        f1(5),
        f1(5, 20),
        f2(1, 2, a=3, b=4),
        f3(10)(5),
        [f(1) for f in funcs]
    ]
'''
    result = await execute_async(source, entry_point='test_lambdas')
    assert result.result == [
        15,
        25,
        ((1, 2), {'a': 3, 'b': 4}),
        15,
        [1, 2, 3]
    ]


@pytest.mark.asyncio
async def test_iterator_protocol():
    """Test custom iterator implementation"""
    source = '''
class Counter:
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count >= self.max_count:
            raise StopIteration
        self.count += 1
        return self.count

def test_iterator():
    counter = Counter(3)
    return list(counter)
'''
    result = await execute_async(source, entry_point='test_iterator')
    assert result.result == [1, 2, 3]


@pytest.mark.asyncio
async def test_descriptor_protocol():
    """Test descriptor protocol (__get__, __set__, __delete__)"""
    source = '''
class Descriptor:
    def __init__(self, name):
        self.name = name
        self.value = None
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.value
    
    def __set__(self, obj, value):
        self.value = f"set_{value}"
    
    def __delete__(self, obj):
        self.value = None

class TestClass:
    attr = Descriptor("attr")

def test_descriptor():
    obj = TestClass()
    obj.attr = "test"
    val = obj.attr
    del obj.attr
    val_after_del = obj.attr
    return (val, val_after_del)
'''
    result = await execute_async(source, entry_point='test_descriptor')
    assert result.result == ("set_test", None)


@pytest.mark.asyncio
async def test_context_manager_edge_cases():
    """Test context manager edge cases"""
    source = '''
class ContextManager:
    def __init__(self, suppress_exception=False):
        self.suppress_exception = suppress_exception
        self.entered = False
        self.exited = False
        self.exception_info = None
    
    def __enter__(self):
        self.entered = True
        return "context_value"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        self.exception_info = (exc_type, exc_val, exc_tb)
        return self.suppress_exception

def test_context_managers():
    results = []
    
    # Normal execution
    with ContextManager() as value:
        results.append(f"normal: {value}")
    
    # Exception suppressed
    with ContextManager(suppress_exception=True):
        raise ValueError("suppressed")
        results.append("should not reach")  # This won't execute
    results.append("after suppressed")
    
    # Exception not suppressed
    try:
        with ContextManager(suppress_exception=False):
            raise RuntimeError("not suppressed")
    except RuntimeError:
        results.append("caught not suppressed")
    
    return results
'''
    result = await execute_async(source, entry_point='test_context_managers')
    assert result.result == ["normal: context_value", "after suppressed", "caught not suppressed"]


@pytest.mark.asyncio
async def test_async_iterator_protocol():
    """Test async iterator protocol"""
    source = '''
class AsyncCounter:
    def __init__(self, max_count):
        self.max_count = max_count
        self.count = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.count >= self.max_count:
            raise StopAsyncIteration
        self.count += 1
        return self.count

async def test_async_iterator():
    counter = AsyncCounter(3)
    result = []
    async for item in counter:
        result.append(item)
    return result
'''
    result = await execute_async(source, entry_point='test_async_iterator')
    assert result.result == [1, 2, 3]


@pytest.mark.asyncio
async def test_comprehension_edge_cases():
    """Test edge cases in comprehensions"""
    source = '''
def test_comprehensions():
    # Multiple if conditions
    result1 = [x for x in range(10) if x % 2 == 0 if x % 3 == 0]
    
    # Nested comprehensions with different variable names
    result2 = [[x + y for x in range(3)] for y in range(2)]
    
    # Dict comprehension with expression keys
    result3 = {f"key_{i}": i**2 for i in range(3)}
    
    # Set comprehension with duplicates
    result4 = {x % 3 for x in range(10)}
    
    # Generator comprehension consumed multiple times
    gen = (x * 2 for x in range(3))
    result5 = list(gen)
    result6 = list(gen)  # Should be empty
    
    return (result1, result2, result3, result4, result5, result6)
'''
    result = await execute_async(source, entry_point='test_comprehensions')
    assert result.result == (
        [0, 6],                                    # Numbers divisible by both 2 and 3
        [[0, 1, 2], [1, 2, 3]],                   # Nested comprehensions
        {"key_0": 0, "key_1": 1, "key_2": 4},     # Dict comprehension
        {0, 1, 2},                                # Set comprehension (unique remainders)
        [0, 2, 4],                                # Generator first consumption
        []                                        # Generator second consumption (empty)
    )


@pytest.mark.asyncio 
async def test_function_edge_cases():
    """Test edge cases in function calls and definitions"""
    source = '''
def test_functions():
    # Function with all parameter types
    def complex_func(pos_only, /, normal, default=10, *args, kw_only, kw_default=20, **kwargs):
        return (pos_only, normal, default, args, kw_only, kw_default, kwargs)
    
    # Call with all parameter types
    result = complex_func(1, 2, 3, 4, 5, kw_only=6, kw_default=7, extra=8)
    
    # Function with annotations
    def annotated_func(x: int, y: str = "default") -> tuple:
        return (x, y)
    
    # Call annotated function
    result2 = annotated_func(42)
    
    return (result, result2)
'''
    result = await execute_async(source, entry_point='test_functions')
    assert result.result == (
        (1, 2, 3, (4, 5), 6, 7, {'extra': 8}),
        (42, "default")
    )


@pytest.mark.asyncio
async def test_import_edge_cases():
    """Test edge cases in import statements"""
    source = '''
# Import with multiple aliases
from math import sin as sine, cos as cosine, pi as PI

# Import everything from math (should work with allowed modules)
import math

def test_imports():
    return [
        sine(PI/2),        # Should be 1.0
        cosine(0),         # Should be 1.0
        math.sqrt(4),      # Should be 2.0
    ]
'''
    result = await execute_async(source, entry_point='test_imports', allowed_modules=['math'])
    assert abs(result.result[0] - 1.0) < 0.0001
    assert abs(result.result[1] - 1.0) < 0.0001
    assert result.result[2] == 2.0
