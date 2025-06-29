import pytest
from quantalogic_pythonbox import execute_async
from quantalogic_pythonbox.slice_utils import CustomSlice


# Context managers tests
@pytest.mark.asyncio
async def test_with_statement_exception_handling():
    source = '''
class ExceptionCtx:
    def __init__(self):
        self.entered = False
        self.exited = False
    
    def __enter__(self):
        self.entered = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False  # Don't suppress exception

def test_exception():
    ctx = ExceptionCtx()
    try:
        with ctx:
            raise ValueError("test error")
    except ValueError:
        pass
    return ctx.entered and ctx.exited
'''
    result = await execute_async(source, entry_point='test_exception')
    assert result.result is True


@pytest.mark.asyncio
async def test_with_statement_exception_suppression():
    source = '''
class SuppressingCtx:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return True  # Suppress exception

def test_suppression():
    with SuppressingCtx():
        raise ValueError("test error")
    return "no error"
'''
    result = await execute_async(source, entry_point='test_suppression')
    assert result.result == "no error"


@pytest.mark.asyncio
async def test_with_multiple_contexts():
    source = '''
class Counter:
    def __init__(self, name):
        self.name = name
        self.count = 0
    
    def __enter__(self):
        self.count += 1
        return self.count
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def test_multiple():
    c1 = Counter("first")
    c2 = Counter("second")
    with c1 as v1, c2 as v2:
        return v1 + v2
'''
    result = await execute_async(source, entry_point='test_multiple')
    assert result.result == 2


@pytest.mark.asyncio
async def test_async_with_statement():
    source = '''
class AsyncCtx:
    def __init__(self):
        self.value = "async"
    
    async def __aenter__(self):
        return self.value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

async def test_async_with():
    async with AsyncCtx() as value:
        return value
'''
    result = await execute_async(source, entry_point='test_async_with')
    assert result.result == "async"


@pytest.mark.asyncio
async def test_async_with_exception():
    source = '''
class AsyncExceptionCtx:
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False  # Don't suppress

async def test_async_exception():
    try:
        async with AsyncExceptionCtx():
            raise ValueError("async error")
    except ValueError:
        return "caught"
    return "not caught"
'''
    result = await execute_async(source, entry_point='test_async_exception')
    assert result.result == "caught"


# Comprehension tests
@pytest.mark.asyncio
async def test_nested_list_comprehension():
    source = '''
def test_nested():
    return [[x + y for x in [1, 2]] for y in [10, 20]]
'''
    result = await execute_async(source, entry_point='test_nested')
    assert result.result == [[11, 12], [21, 22]]


@pytest.mark.asyncio
async def test_dict_comprehension_with_conditions():
    source = '''
def test_dict_comp():
    return {x: x**2 for x in range(5) if x % 2 == 0}
'''
    result = await execute_async(source, entry_point='test_dict_comp')
    assert result.result == {0: 0, 2: 4, 4: 16}


@pytest.mark.asyncio
async def test_set_comprehension():
    source = '''
def test_set_comp():
    return {x for x in [1, 2, 2, 3, 3, 3]}
'''
    result = await execute_async(source, entry_point='test_set_comp')
    assert result.result == {1, 2, 3}


@pytest.mark.asyncio
async def test_generator_expression():
    source = '''
def test_gen_exp():
    gen = (x * 2 for x in range(3))
    return list(gen)
'''
    result = await execute_async(source, entry_point='test_gen_exp')
    assert result.result == [0, 2, 4]


@pytest.mark.asyncio
async def test_comprehension_with_async_iterable():
    source = '''
class AsyncIterable:
    def __init__(self, items):
        self.items = items
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

async def test_async_comp():
    async_iter = AsyncIterable([1, 2, 3])
    result = []
    async for item in async_iter:
        result.append(item * 2)
    return result
'''
    result = await execute_async(source, entry_point='test_async_comp')
    assert result.result == [2, 4, 6]


# Generator tests
@pytest.mark.asyncio
async def test_generator_send():
    source = '''
def generator_func():
    x = yield 1
    y = yield x * 2
    return y + 1

def test_send():
    gen = generator_func()
    first = next(gen)  # Get 1
    second = gen.send(5)  # Send 5, get 10
    try:
        gen.send(3)  # Send 3, generator returns 4
    except StopIteration as e:
        return e.value
    return None
'''
    result = await execute_async(source, entry_point='test_send')
    assert result.result == 4


@pytest.mark.asyncio
async def test_generator_throw():
    source = '''
def generator_func():
    try:
        yield 1
    except ValueError as e:
        yield str(e)
    yield 2

def test_throw():
    gen = generator_func()
    first = next(gen)  # Get 1
    second = gen.throw(ValueError, "test error")  # Throw exception, get "test error"
    third = next(gen)  # Get 2
    return [first, second, third]
'''
    result = await execute_async(source, entry_point='test_throw')
    assert result.result == [1, "test error", 2]


@pytest.mark.asyncio
async def test_generator_close():
    source = '''
def generator_func():
    try:
        yield 1
        yield 2
    except GeneratorExit:
        return "closed"

def test_close():
    gen = generator_func()
    first = next(gen)
    gen.close()
    return first
'''
    result = await execute_async(source, entry_point='test_close')
    assert result.result == 1


# Exception handling tests
@pytest.mark.asyncio
async def test_try_except_else_finally():
    source = '''
def test_all_blocks():
    result = []
    try:
        result.append("try")
        # No exception
    except ValueError:
        result.append("except")
    else:
        result.append("else")
    finally:
        result.append("finally")
    return result
'''
    result = await execute_async(source, entry_point='test_all_blocks')
    assert result.result == ["try", "else", "finally"]


@pytest.mark.asyncio
async def test_try_except_else_finally_with_exception():
    source = '''
def test_with_exception():
    result = []
    try:
        result.append("try")
        raise ValueError("test")
    except ValueError:
        result.append("except")
    else:
        result.append("else")
    finally:
        result.append("finally")
    return result
'''
    result = await execute_async(source, entry_point='test_with_exception')
    assert result.result == ["try", "except", "finally"]


@pytest.mark.asyncio
async def test_nested_try_except():
    source = '''
def test_nested():
    try:
        try:
            raise ValueError("inner")
        except ValueError as e:
            raise RuntimeError("outer") from e
    except RuntimeError as e:
        return str(e.__cause__)
'''
    result = await execute_async(source, entry_point='test_nested')
    assert result.result == "inner"


@pytest.mark.asyncio
async def test_exception_with_custom_args():
    source = '''
def test_custom_exception():
    try:
        raise ValueError(1, 2, 3)
    except ValueError as e:
        return e.args
'''
    result = await execute_async(source, entry_point='test_custom_exception')
    assert result.result == (1, 2, 3)


# Assignment tests
@pytest.mark.asyncio
async def test_augmented_assignment_all_ops():
    source = '''
def test_aug_assign():
    x = 10
    x += 5
    x -= 2
    x *= 3
    x //= 4
    x %= 7
    x **= 2
    return x
'''
    result = await execute_async(source, entry_point='test_aug_assign')
    expected = 10
    expected += 5  # 15
    expected -= 2  # 13
    expected *= 3  # 39
    expected //= 4  # 9
    expected %= 7  # 2
    expected **= 2  # 4
    assert result.result == expected


@pytest.mark.asyncio
async def test_tuple_unpacking_assignment():
    source = '''
def test_unpacking():
    a, b, c = [1, 2, 3]
    x, *y, z = [10, 20, 30, 40, 50]
    return (a, b, c, x, y, z)
'''
    result = await execute_async(source, entry_point='test_unpacking')
    assert result.result == (1, 2, 3, 10, [20, 30, 40], 50)


@pytest.mark.asyncio
async def test_starred_assignment():
    source = '''
def test_starred():
    first, *middle, last = [1, 2, 3, 4, 5]
    return (first, middle, last)
'''
    result = await execute_async(source, entry_point='test_starred')
    assert result.result == (1, [2, 3, 4], 5)


# Function tests
@pytest.mark.asyncio
async def test_function_with_defaults_and_kwargs():
    source = '''
def func(a, b=10, *args, c=20, **kwargs):
    return (a, b, args, c, kwargs)

def test_call():
    return func(1, 2, 3, 4, c=30, d=40, e=50)
'''
    result = await execute_async(source, entry_point='test_call')
    assert result.result == (1, 2, (3, 4), 30, {'d': 40, 'e': 50})


@pytest.mark.asyncio
async def test_function_annotations():
    source = '''
def func(x: int, y: str = "default") -> str:
    return f"{x}: {y}"

def test_annotations():
    return func(42)
'''
    result = await execute_async(source, entry_point='test_annotations')
    assert result.result == "42: default"


@pytest.mark.asyncio
async def test_lambda_with_defaults():
    source = '''
def test_lambda():
    f = lambda x, y=10: x + y
    return f(5)
'''
    result = await execute_async(source, entry_point='test_lambda')
    assert result.result == 15


# Async function tests
@pytest.mark.asyncio
async def test_async_function_with_await():
    source = '''
import asyncio

async def helper(x):
    await asyncio.sleep(0.01)
    return x * 2

async def test_async():
    result = await helper(5)
    return result
'''
    result = await execute_async(source, entry_point='test_async', allowed_modules=['asyncio'])
    assert result.result == 10


@pytest.mark.asyncio
async def test_async_generator():
    source = '''
async def async_gen():
    yield 1
    yield 2
    yield 3

async def test_async_gen():
    result = []
    async for item in async_gen():
        result.append(item)
    return result
'''
    result = await execute_async(source, entry_point='test_async_gen')
    assert result.result == [1, 2, 3]


# Import tests
@pytest.mark.asyncio
async def test_import_with_alias():
    source = '''
import json as j

def test_import():
    return j.dumps({"key": "value"})
'''
    result = await execute_async(source, entry_point='test_import', allowed_modules=['json'])
    assert result.result == '{"key": "value"}'


@pytest.mark.asyncio
async def test_from_import_with_alias():
    source = '''
from math import sqrt as square_root

def test_from_import():
    return square_root(16)
'''
    result = await execute_async(source, entry_point='test_from_import', allowed_modules=['math'])
    assert result.result == 4.0


# Slice tests with CustomSlice
def test_custom_slice_edge_cases():
    # Test with None values
    s1 = CustomSlice(None, None, None)
    assert s1.start is None
    assert s1.stop is None
    assert s1.step is None
    
    # Test with negative values
    s2 = CustomSlice(-1, -5, -2)
    assert s2.start == -1
    assert s2.stop == -5
    assert s2.step == -2
    
    # Test indexing edge cases
    s3 = CustomSlice(0, 10, 1)
    with pytest.raises(IndexError):
        _ = s3[3]  # Only 0, 1, 2 are valid
    
    with pytest.raises(TypeError):
        _ = s3[slice(0, 1)]  # slice objects not supported


@pytest.mark.asyncio
async def test_slice_operations():
    source = '''
def test_slice():
    data = list(range(10))
    return [
        data[1:5],      # Basic slice
        data[::2],      # Step slice
        data[::-1],     # Reverse slice
        data[1:8:2],    # Start, stop, step
        data[-3:],      # Negative start
        data[:-2],      # Negative stop
    ]
'''
    result = await execute_async(source, entry_point='test_slice')
    expected = [
        [1, 2, 3, 4],
        [0, 2, 4, 6, 8],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [1, 3, 5, 7],
        [7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7]
    ]
    assert result.result == expected


# Class tests
@pytest.mark.asyncio
async def test_class_with_properties():
    source = '''
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

def test_properties():
    c = Circle(5)
    area1 = c.area
    c.radius = 10
    area2 = c.area
    return (area1, area2)
'''
    result = await execute_async(source, entry_point='test_properties')
    assert abs(result.result[0] - 78.53975) < 0.001
    assert abs(result.result[1] - 314.159) < 0.001


@pytest.mark.asyncio
async def test_class_methods_and_static_methods():
    source = '''
class MathUtils:
    base = 10
    
    @classmethod
    def from_string(cls, s):
        return cls(int(s))
    
    @staticmethod
    def add(a, b):
        return a + b
    
    def __init__(self, value):
        self.value = value

def test_methods():
    obj = MathUtils.from_string("42")
    static_result = MathUtils.add(1, 2)
    return (obj.value, static_result)
'''
    result = await execute_async(source, entry_point='test_methods')
    assert result.result == (42, 3)


# Operator tests
@pytest.mark.asyncio
async def test_all_comparison_operators():
    source = '''
def test_comparisons():
    return [
        5 == 5,   # Equal
        5 != 3,   # Not equal
        5 > 3,    # Greater than
        3 < 5,    # Less than
        5 >= 5,   # Greater or equal
        3 <= 5,   # Less or equal
        5 is 5,   # Identity (might be True for small ints)
        5 is not 3,  # Not identity
        3 in [1, 2, 3],  # In
        4 not in [1, 2, 3],  # Not in
    ]
'''
    result = await execute_async(source, entry_point='test_comparisons')
    expected = [True, True, True, True, True, True, True, True, True, True]
    assert result.result == expected


@pytest.mark.asyncio
async def test_bitwise_operators():
    source = '''
def test_bitwise():
    a, b = 12, 10  # 1100, 1010 in binary
    return [
        a & b,    # AND: 8 (1000)
        a | b,    # OR: 14 (1110)
        a ^ b,    # XOR: 6 (0110)
        ~a,       # NOT: -13
        a << 1,   # Left shift: 24
        a >> 1,   # Right shift: 6
    ]
'''
    result = await execute_async(source, entry_point='test_bitwise')
    assert result.result == [8, 14, 6, -13, 24, 6]


# Control flow tests
@pytest.mark.asyncio
async def test_complex_control_flow():
    source = '''
def test_control():
    result = []
    
    # Nested loops with continue and break
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if i + j > 2:
                break
            result.append((i, j))
    
    # While loop with else
    x = 0
    while x < 3:
        x += 1
    else:
        result.append("while_else")
    
    # For loop with else (no break)
    for i in range(2):
        pass
    else:
        result.append("for_else")
    
    return result
'''
    result = await execute_async(source, entry_point='test_control')
    assert result.result == [(0, 1), (1, 0), "while_else", "for_else"]


# Test error handling edge cases
@pytest.mark.asyncio
async def test_name_error():
    source = '''
def test_name():
    return undefined_variable
'''
    result = await execute_async(source, entry_point='test_name')
    assert result.error is not None
    assert "NameError" in result.error


@pytest.mark.asyncio
async def test_type_error():
    source = '''
def test_type():
    return "string" + 42
'''
    result = await execute_async(source, entry_point='test_type')
    assert result.error is not None
    assert "TypeError" in result.error


@pytest.mark.asyncio
async def test_timeout_error():
    source = '''
import time

def test_timeout():
    time.sleep(2)  # This will timeout
    return "should not reach here"
'''
    result = await execute_async(source, entry_point='test_timeout', timeout=0.1, allowed_modules=['time'])
    assert result.error is not None
    assert "TimeoutError" in result.error


# Test special methods
@pytest.mark.asyncio
async def test_special_methods():
    source = '''
class SpecialClass:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return f"Special({self.value})"
    
    def __repr__(self):
        return f"SpecialClass({self.value!r})"
    
    def __len__(self):
        return len(str(self.value))
    
    def __getitem__(self, key):
        return str(self.value)[key]
    
    def __setitem__(self, key, value):
        s = list(str(self.value))
        s[key] = value
        self.value = ''.join(s)
    
    def __contains__(self, item):
        return item in str(self.value)
    
    def __call__(self, arg):
        return f"{self.value}({arg})"

def test_special():
    obj = SpecialClass("hello")
    return [
        str(obj),           # __str__
        repr(obj),          # __repr__
        len(obj),           # __len__
        obj[1],             # __getitem__
        'l' in obj,         # __contains__
        obj("world"),       # __call__
    ]
'''
    result = await execute_async(source, entry_point='test_special')
    assert result.result == [
        "Special(hello)",
        "SpecialClass('hello')",
        5,
        'e',
        True,
        "hello(world)"
    ]


# Test memory and recursion limits
@pytest.mark.asyncio
async def test_recursion_limit():
    source = '''
def recursive_func(n):
    if n <= 0:
        return 0
    return recursive_func(n - 1) + 1

def test_recursion():
    return recursive_func(50)  # Should work with small recursion
'''
    result = await execute_async(source, entry_point='test_recursion')
    assert result.result == 50


# Test delete operations
@pytest.mark.asyncio
async def test_delete_operations():
    source = '''
def test_delete():
    x = [1, 2, 3, 4, 5]
    del x[2]  # Delete by index
    
    d = {'a': 1, 'b': 2, 'c': 3}
    del d['b']  # Delete by key
    
    class Obj:
        def __init__(self):
            self.attr = "value"
    
    obj = Obj()
    del obj.attr  # Delete attribute
    
    return (x, d, hasattr(obj, 'attr'))
'''
    result = await execute_async(source, entry_point='test_delete')
    assert result.result == ([1, 2, 4, 5], {'a': 1, 'c': 3}, False)


# Test yield from
@pytest.mark.asyncio
async def test_yield_from():
    source = '''
def sub_generator():
    yield 1
    yield 2
    return "sub_done"

def main_generator():
    result = yield from sub_generator()
    yield result
    yield 3

def test_yield_from():
    gen = main_generator()
    return list(gen)
'''
    result = await execute_async(source, entry_point='test_yield_from')
    assert result.result == [1, 2, "sub_done", 3]


# Test walrus operator (assignment expressions)
@pytest.mark.asyncio
async def test_walrus_operator():
    source = '''
def test_walrus():
    data = [1, 2, 3, 4, 5]
    result = []
    
    # Walrus operator in list comprehension
    squares = [y*y for x in data if (y := x * 2) > 4]
    
    # Walrus operator in while loop
    while (n := len(result)) < 3:
        result.append(n)
    
    return (squares, result)
'''
    result = await execute_async(source, entry_point='test_walrus')
    assert result.result == ([16, 36, 64, 100], [0, 1, 2])


# Test f-string edge cases
@pytest.mark.asyncio
async def test_fstring_edge_cases():
    source = '''
def test_fstring():
    name = "world"
    number = 42
    pi = 3.14159
    
    return [
        f"Hello {name}!",                    # Basic
        f"Number: {number:04d}",             # Format specifier
        f"Pi: {pi:.2f}",                     # Float precision
        f"Expression: {2 + 3}",              # Expression
        f"Method: {name.upper()}",           # Method call
        f"Nested: {f'inner {name}'}",        # Nested f-string
    ]
'''
    result = await execute_async(source, entry_point='test_fstring')
    assert result.result == [
        "Hello world!",
        "Number: 0042", 
        "Pi: 3.14",
        "Expression: 5",
        "Method: WORLD",
        "Nested: inner world"
    ]


# Test more comprehensive error cases
@pytest.mark.asyncio
async def test_attribute_error():
    source = '''
def test_attr():
    obj = "string"
    return obj.nonexistent_method()
'''
    result = await execute_async(source, entry_point='test_attr')
    assert result.error is not None
    assert "AttributeError" in result.error


@pytest.mark.asyncio
async def test_key_error():
    source = '''
def test_key():
    d = {'a': 1, 'b': 2}
    return d['nonexistent']
'''
    result = await execute_async(source, entry_point='test_key')
    assert result.error is not None
    assert "KeyError" in result.error


@pytest.mark.asyncio
async def test_index_error():
    source = '''
def test_index():
    lst = [1, 2, 3]
    return lst[10]
'''
    result = await execute_async(source, entry_point='test_index')
    assert result.error is not None
    assert "IndexError" in result.error


@pytest.mark.asyncio
async def test_zero_division_error():
    source = '''
def test_zero_div():
    return 1 / 0
'''
    result = await execute_async(source, entry_point='test_zero_div')
    assert result.error is not None
    assert "ZeroDivisionError" in result.error


# Test more comprehensive slice utils
def test_custom_slice_comprehensive():
    # Test all possible combinations
    s1 = CustomSlice(1, 5, 2)
    assert s1.start == 1
    assert s1.stop == 5
    assert s1.step == 2
    assert s1[0] == 1  # start
    assert s1[1] == 5  # stop
    assert s1[2] == 2  # step
    
    # Test None values
    s2 = CustomSlice(None, None, None)
    assert s2[0] is None
    assert s2[1] is None
    assert s2[2] is None
    
    # Test mixed values
    s3 = CustomSlice(10, None, -1)
    assert s3[0] == 10
    assert s3[1] is None
    assert s3[2] == -1
    
    # Test repr
    assert repr(s1) == "CustomSlice(start=1, stop=5, step=2)"
    assert repr(s2) == "CustomSlice(start=None, stop=None, step=None)"
    
    # Test error cases
    with pytest.raises(IndexError):
        _ = s1[3]
    
    with pytest.raises(IndexError):
        _ = s1[-1]
    
    with pytest.raises(TypeError):
        _ = s1["invalid"]
