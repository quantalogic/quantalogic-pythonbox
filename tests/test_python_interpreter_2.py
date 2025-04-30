import pytest
from quantalogic_pythonbox import execute_async


@pytest.mark.asyncio
async def test_empty_function():
    """Test that an empty function implicitly returns None."""
    source = """
def compute():
    pass
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result is None


@pytest.mark.asyncio
async def test_multiple_return_statements():
    """Test that only the first return statement is executed."""
    source = """
def compute():
    return 1
    return 2
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 1


@pytest.mark.asyncio
async def test_syntax_error():
    """Test that a syntax error is caught and reported."""
    source = """
def compute():
    return 1 +
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.error is not None
    assert "SyntaxError" in result.error


@pytest.mark.asyncio
async def test_multiple_functions_without_call():
    """Test that an undefined entry point raises an error."""
    source = """
def foo():
    return 1
def bar():
    return 2
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.error is not None
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_string_multiplication_negative():
    """Test string repetition with a negative multiplier returns an empty string."""
    source = """
def compute():
    return "a" * -1
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == ""


@pytest.mark.asyncio
async def test_float_precision():
    """Test floating-point arithmetic precision."""
    source = """
def compute():
    return 0.1 + 0.2
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert abs(result.result - 0.3) < 1e-10


@pytest.mark.asyncio
async def test_integer_overflow():
    """Test that large integer operations are handled correctly."""
    source = """
def compute():
    return 2 ** 1000
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 2 ** 1000


@pytest.mark.asyncio
async def test_unicode_strings():
    """Test handling of Unicode strings."""
    source = """
def compute():
    return "こんにちは"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "こんにちは"


@pytest.mark.asyncio
async def test_multiline_string():
    """Test triple-quoted multiline strings."""
    source = """
def compute():
    return '''line1
line2'''
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "line1\nline2"


@pytest.mark.asyncio
async def test_raw_string():
    """Test raw strings with escape sequences."""
    source = """
def compute():
    return r"\\n"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "\\n"


@pytest.mark.asyncio
async def test_bytes_literal():
    """Test bytes literals."""
    source = """
def compute():
    return b"hello"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == b"hello"


@pytest.mark.asyncio
async def test_bytearray():
    """Test mutable bytearray operations."""
    source = """
def compute():
    b = bytearray(b"hi")
    b[0] = 72  # 'H'
    return b
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == bytearray(b"Hi")


@pytest.mark.asyncio
async def test_set_union_with_empty():
    """Test set union with an empty set."""
    source = """
def compute():
    return {1, 2} | set()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {1, 2}


@pytest.mark.asyncio
async def test_frozenset_operations():
    """Test frozenset intersection."""
    source = """
def compute():
    return frozenset([1, 2]) & frozenset([2, 3])
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == frozenset({2})


@pytest.mark.asyncio
async def test_dictionary_update():
    """Test dictionary update method."""
    source = """
def compute():
    d = {"a": 1}
    d.update({"b": 2})
    return d
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_list_extend():
    """Test list extend method."""
    source = """
def compute():
    lst = [1]
    lst.extend([2, 3])
    return lst
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [1, 2, 3]


@pytest.mark.asyncio
async def test_tuple_concatenation():
    """Test tuple concatenation."""
    source = """
def compute():
    return (1,) + (2, 3)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == (1, 2, 3)


@pytest.mark.asyncio
async def test_single_element_tuple():
    """Test single-element tuple syntax."""
    source = """
def compute():
    return (1,)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == (1,)


@pytest.mark.asyncio
async def test_empty_tuple():
    """Test empty tuple."""
    source = """
def compute():
    return ()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == ()


@pytest.mark.asyncio
async def test_nested_tuples():
    """Test accessing nested tuples."""
    source = """
def compute():
    return (1, (2, 3))[1][0]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 2


@pytest.mark.asyncio
async def test_list_reverse():
    """Test list reverse method."""
    source = """
def compute():
    lst = [1, 2, 3]
    lst.reverse()
    return lst
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [3, 2, 1]


@pytest.mark.asyncio
async def test_list_sort_with_key():
    """Test list sort with a key function."""
    source = """
def compute():
    lst = ["b", "aa", "ccc"]
    lst.sort(key=len)
    return lst
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == ["b", "aa", "ccc"]


@pytest.mark.asyncio
async def test_dictionary_get_with_default():
    """Test dictionary get method with default value."""
    source = """
def compute():
    d = {"a": 1}
    return d.get("b", 0)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 0


@pytest.mark.asyncio
async def test_set_add():
    """Test set add method."""
    source = """
def compute():
    s = {1, 2}
    s.add(3)
    return s
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {1, 2, 3}


@pytest.mark.asyncio
async def test_set_discard():
    """Test set discard method with non-existent element."""
    source = """
def compute():
    s = {1, 2}
    s.discard(3)
    return s
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {1, 2}


@pytest.mark.asyncio
async def test_string_join():
    """Test string join method."""
    source = """
def compute():
    return "-".join(["a", "b", "c"])
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "a-b-c"


@pytest.mark.asyncio
async def test_string_split():
    """Test string split method with custom separator."""
    source = """
def compute():
    return "a,b,c".split(",")
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_string_replace():
    """Test string replace method."""
    source = """
def compute():
    return "hello".replace("l", "w")
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "hewwo"


@pytest.mark.asyncio
async def test_string_strip():
    """Test string strip method."""
    source = """
def compute():
    return "  hello  ".strip()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "hello"


@pytest.mark.asyncio
async def test_string_casefold():
    """Test string casefold method for case-insensitive comparison."""
    source = """
def compute():
    return "HELLO".casefold()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "hello"


@pytest.mark.asyncio
async def test_string_find():
    """Test string find method."""
    source = """
def compute():
    return "hello".find("l")
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 2


@pytest.mark.asyncio
async def test_string_isdigit():
    """Test string isdigit method."""
    source = """
def compute():
    return "123".isdigit()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result is True


@pytest.mark.asyncio
async def test_list_index():
    """Test list index method."""
    source = """
def compute():
    return [1, 2, 3].index(2)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 1


@pytest.mark.asyncio
async def test_list_count():
    """Test list count method."""
    source = """
def compute():
    return [1, 2, 2, 3].count(2)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 2


@pytest.mark.asyncio
async def test_dictionary_items():
    """Test dictionary items method."""
    source = """
def compute():
    return list({"a": 1, "b": 2}.items())
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [("a", 1), ("b", 2)]


@pytest.mark.asyncio
async def test_dictionary_values():
    """Test dictionary values method."""
    source = """
def compute():
    return list({"a": 1, "b": 2}.values())
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [1, 2]


@pytest.mark.asyncio
async def test_set_difference_update():
    """Test set difference_update method."""
    source = """
def compute():
    s = {1, 2, 3}
    s.difference_update({2})
    return s
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {1, 3}


@pytest.mark.asyncio
async def test_boolean_not():
    """Test logical NOT operator."""
    source = """
def compute():
    return not True
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result is False


@pytest.mark.asyncio
async def test_identity_with_none():
    """Test identity comparison with None."""
    source = """
def compute():
    return None is None
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result is True


@pytest.mark.asyncio
async def test_chained_assignment():
    """Test chained assignment with multiple variables."""
    source = """
def compute():
    a = b = c = [1]
    a.append(2)
    return (a, b, c)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == ([1, 2], [1, 2], [1, 2])


@pytest.mark.asyncio
async def test_global_variable_read():
    """Test reading a global variable."""
    source = """
x = 5
def compute():
    return x
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 5


@pytest.mark.asyncio
async def test_nonlocal_variable_read():
    """Test reading a nonlocal variable."""
    source = """
def outer():
    x = 5
    def inner():
        return x
    return inner()
def compute():
    return outer()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 5


@pytest.mark.asyncio
async def test_function_no_return():
    """Test that a function with no return statement returns None."""
    source = """
def compute():
    x = 1
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result is None


@pytest.mark.asyncio
async def test_pass_in_loop():
    """Test pass statement in a loop."""
    source = """
def compute():
    for i in range(3):
        pass
    return i
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 2


@pytest.mark.asyncio
async def test_break_in_nested_loop():
    """Test break in a nested loop."""
    source = """
def compute():
    for i in range(3):
        for j in range(3):
            if j == 1:
                break
        if i == 1:
            break
    return (i, j)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == (1, 1)


@pytest.mark.asyncio
async def test_continue_in_nested_loop():
    """Test continue in a nested loop."""
    source = """
def compute():
    s = 0
    for i in range(3):
        for j in range(3):
            if j == 1:
                continue
            s += j
    return s
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6  # 0 + 2 for each i


@pytest.mark.asyncio
async def test_raise_base_exception():
    """Test catching a base Exception."""
    source = """
def compute():
    try:
        raise Exception("test")
    except Exception as e:
        return str(e)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "test"


@pytest.mark.asyncio
async def test_multiple_except_clauses():
    """Test multiple except clauses with specific exceptions."""
    source = """
def compute():
    try:
        [] + 1
    except TypeError:
        return "type"
    except ValueError:
        return "value"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "type"


@pytest.mark.asyncio
async def test_try_except_else():
    """Test else clause in try-except."""
    source = """
def compute():
    try:
        x = 1
    except Exception:
        return "error"
    else:
        return "success"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "success"


@pytest.mark.asyncio
async def test_finally_without_exception():
    """Test finally clause without an exception."""
    source = """
def compute():
    result = 0
    try:
        result = 1
    finally:
        result += 2
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 3


@pytest.mark.asyncio
async def test_raise_and_reraise():
    """Test re-raising an exception."""
    source = """
def compute():
    try:
        raise ValueError("test")
    except ValueError as e:
        raise
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert "ValueError" in result.error
    assert "test" in result.error


@pytest.mark.asyncio
async def test_generator_yield_from():
    """Test yield from with a range."""
    source = """
def gen():
    yield from range(3)
def compute():
    return list(gen())
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [0, 1, 2]


@pytest.mark.asyncio
async def test_generator_multiple_yields():
    """Test multiple yield statements in a generator."""
    source = """
def gen():
    yield 1
    yield 2
    yield 3
def compute():
    return list(gen())
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [1, 2, 3]


@pytest.mark.asyncio
async def test_empty_generator():
    """Test a generator with no yields returns an empty list."""
    source = """
def gen():
    if False:
        yield 1
def compute():
    return list(gen())
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == []


def test_generator_with_return():
    code = """
def gen():
    yield 1
    return 2

def main():
    g = gen()
    return next(g), next(g, None)
"""
    result = execute_async(code, entry_point="main")
    assert result.result == (1, None)  # Correct expectation per Python semantics


def test_built_in_next_with_default_iterator():
    code = """
def main():
    it = iter([1])
    return next(it), next(it, None)
"""
    result = execute_async(code, entry_point="main")
    assert result.result == (1, None)  # Expecting (1, None) as per standard Python behavior


@pytest.mark.asyncio
async def test_class_static_variable():
    """Test accessing a class-level static variable."""
    source = """
class A:
    x = 5
def compute():
    return A.x
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 5


@pytest.mark.asyncio
async def test_class_method_chaining():
    """Test method chaining in a class."""
    source = """
class A:
    def add(self, x):
        return x + 1
    def compute(self):
        return self.add(5)
def compute():
    return A().compute()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6


@pytest.mark.asyncio
async def test_property_setter():
    """Test property with getter and setter."""
    source = """
class A:
    def __init__(self):
        self._x = 0
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
def compute():
    a = A()
    a.x = 10
    return a.x
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 10


@pytest.mark.asyncio
async def test_multiple_inheritance():
    """Test multiple inheritance with method resolution order."""
    source = """
class A:
    def value(self):
        return 1
class B:
    def value(self):
        return 2
class C(A, B):
    pass
def compute():
    return C().value()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 1  # A comes first in MRO


@pytest.mark.asyncio
async def test_abc_import():
    """Test importing and using abc module."""
    source = """
from abc import ABC, abstractmethod
class A(ABC):
    @abstractmethod
    def f(self):
        pass
def compute():
    return True
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["abc"])
    assert result.result is True


@pytest.mark.asyncio
async def test_decorator_with_arguments():
    """Test a decorator that takes arguments."""
    source = """
def add_n(n):
    def deco(func):
        def wrapper(*args):
            return func(*args) + n
        return wrapper
    return deco
@add_n(3)
def add(a, b):
    return a + b
def compute():
    return add(1, 2)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6


@pytest.mark.asyncio
async def test_nested_decorator():
    """Test multiple nested decorators."""
    source = """
def deco1(func):
    def wrapper(*args):
        return func(*args) + 1
    return wrapper
def deco2(func):
    def wrapper(*args):
        return func(*args) * 2
    return wrapper
@deco1
@deco2
def add(a, b):
    return a + b
def compute():
    return add(2, 3)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 11  # (2 + 3) * 2 + 1


@pytest.mark.asyncio
async def test_lambda_multiple_arguments():
    """Test lambda with multiple arguments."""
    source = """
def compute():
    f = lambda x, y, z: x + y + z
    return f(1, 2, 3)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6


@pytest.mark.asyncio
async def test_list_comprehension_multiple_loops():
    """Test nested loops in a list comprehension."""
    source = """
def compute():
    return [(x, y) for x in range(2) for y in range(2)]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.asyncio
async def test_set_comprehension_with_condition():
    """Test set comprehension with a condition."""
    source = """
def compute():
    return {x for x in range(5) if x > 2}
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {3, 4}


@pytest.mark.asyncio
async def test_dict_comprehension_with_condition():
    """Test dictionary comprehension with a condition."""
    source = """
def compute():
    return {x: x*2 for x in range(5) if x % 2 == 0}
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == {0: 0, 2: 4, 4: 8}


@pytest.mark.asyncio
async def test_generator_expression_multiple_conditions():
    """Test generator expression with multiple conditions."""
    source = """
def compute():
    gen = (x for x in range(10) if x % 2 == 0 if x > 4)
    return list(gen)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [6, 8]


@pytest.mark.asyncio
async def test_slice_negative_step():
    """Test slicing with a negative step."""
    source = """
def compute():
    return [0, 1, 2, 3][::-1]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [3, 2, 1, 0]


@pytest.mark.asyncio
async def test_slice_out_of_bounds():
    """Test slicing beyond list bounds."""
    source = """
def compute():
    return [1, 2][5:10]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == []


@pytest.mark.asyncio
async def test_unpacking_too_few_values():
    """Test unpacking with too few values raises ValueError."""
    source = """
def compute():
    a, b, c = [1, 2]
    return a
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert "ValueError" in result.error
    assert "not enough values" in result.error


@pytest.mark.asyncio
async def test_unpacking_too_many_values():
    """Test unpacking with too many values raises ValueError."""
    source = """
def compute():
    a, b = [1, 2, 3]
    return a
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert "ValueError" in result.error
    assert "not enough values" in result.error


@pytest.mark.asyncio
async def test_star_unpacking_in_call():
    """Test unpacking with * in a function call."""
    source = """
def add(a, b, c):
    return a + b + c
def compute():
    lst = [1, 2, 3]
    return add(*lst)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6


@pytest.mark.asyncio
async def test_kwargs_unpacking_in_call():
    """Test unpacking with ** in a function call."""
    source = """
def func(x, y):
    return x - y
def compute():
    d = {"x": 5, "y": 2}
    return func(**d)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 3


@pytest.mark.asyncio
async def test_mixed_positional_keyword_args():
    """Test mixing positional and keyword arguments."""
    source = """
def func(a, b, c):
    return a + b + c
def compute():
    return func(1, c=3, b=2)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6


@pytest.mark.asyncio
async def test_positional_only_args():
    """Test positional-only parameters."""
    source = """
def func(a, /, b):
    return a + b
def compute():
    return func(1, 2)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 3


@pytest.mark.asyncio
async def test_keyword_only_args():
    """Test keyword-only parameters."""
    source = """
def func(a, *, b):
    return a + b
def compute():
    return func(1, b=2)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 3


@pytest.mark.asyncio
async def test_default_argument_mutation():
    """Test behavior of mutable default arguments."""
    source = """
def func(lst=[]):
    lst.append(1)
    return lst
def compute():
    return func()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == [1]


@pytest.mark.asyncio
async def test_eval_builtin():
    """Test that eval is restricted."""
    source = """
def compute():
    return eval("1 + 2")
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert "not allowed" in result.error


@pytest.mark.asyncio
async def test_exec_builtin():
    """Test that exec is restricted."""
    source = """
def compute():
    exec("x = 5")
    return x
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert "not allowed" in result.error


@pytest.mark.asyncio
async def test_import_submodule():
    """Test importing a submodule."""
    source = """
from math import sin
def compute():
    return sin(0)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["math"])
    assert result.result == 0.0


@pytest.mark.asyncio
async def test_aliased_import():
    """Test import with an alias."""
    source = """
import math as m
def compute():
    return m.sqrt(4)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["math"])
    assert result.result == 2.0


@pytest.mark.asyncio
async def test_star_import():
    """Test star import from an allowed module."""
    source = """
from math import *
def compute():
    return sqrt(9)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["math"])
    assert result.result == 3.0


@pytest.mark.asyncio
async def test_relative_import_attempt():
    """Test that relative imports are blocked."""
    source = """
from . import foo
def compute():
    return 1
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert "Relative imports are not supported" in result.error


@pytest.mark.asyncio
async def test_complex_conjugate():
    """Test complex number conjugate."""
    source = """
def compute():
    return (1 + 2j).conjugate()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == (1 - 2j)


@pytest.mark.asyncio
async def test_real_imag_parts():
    """Test accessing real and imaginary parts of a complex number."""
    source = """
def compute():
    c = 3 + 4j
    return (c.real, c.imag)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == (3.0, 4.0)


@pytest.mark.asyncio
async def test_bitwise_not():
    """Test bitwise NOT operator."""
    source = """
def compute():
    return ~5
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == -6


@pytest.mark.asyncio
async def test_modulo_negative():
    """Test modulo with a negative operand."""
    source = """
def compute():
    return -7 % 3
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 2


@pytest.mark.asyncio
async def test_floor_division_negative():
    """Test floor division with negative numbers."""
    source = """
def compute():
    return -7 // 3
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == -3


@pytest.mark.asyncio
async def test_power_negative_exponent():
    """Test exponentiation with a negative exponent."""
    source = """
def compute():
    return 2 ** -2
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 0.25


@pytest.mark.asyncio
async def test_octal_literal():
    """Test octal number literal."""
    source = """
def compute():
    return 0o10
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 8


@pytest.mark.asyncio
async def test_hex_literal():
    """Test hexadecimal number literal."""
    source = """
def compute():
    return 0xFF
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 255


@pytest.mark.asyncio
async def test_binary_literal():
    """Test binary number literal."""
    source = """
def compute():
    return 0b1010
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 10


@pytest.mark.asyncio
async def test_string_escape_sequences():
    """Test string with escape sequences."""
    source = """
def compute():
    return "\\t\\r\\n"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "\t\r\n"


@pytest.mark.asyncio
async def test_walrus_operator():
    """Test assignment expression (walrus operator)."""
    source = """
def compute():
    return (x := 5)
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 5


@pytest.mark.asyncio
async def test_walrus_in_if():
    """Test walrus operator in an if condition."""
    source = """
def compute():
    if (x := 5) > 3:
        return x
    return 0
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 5


@pytest.mark.asyncio
async def test_matrix_multiplication_operator():
    """Test matrix multiplication operator with custom class."""
    source = """
def compute():
    class Mat:
        def __matmul__(self, other):
            return 6
    return Mat() @ Mat()
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 6


@pytest.mark.asyncio
async def test_annotated_assignment():
    """Test annotated assignment (type hints ignored in execution)."""
    source = """
def compute():
    x: int = 5
    return x
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == 5