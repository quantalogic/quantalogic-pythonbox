import pytest
from quantalogic_pythonbox import execute_async

# Basic generator test to verify core functionality
@pytest.mark.asyncio
async def test_basic_generator():
    """Test basic generator functionality"""
    source = """
def gen():
    yield 1
    yield 2

def compute():
    g = gen()
    first = next(g)
    second = next(g)
    return [first, second]
"""
    result = await execute_async(source, entry_point="compute")
    assert result.result == [1, 2]

# Minimal async generator test
@pytest.mark.asyncio
async def test_minimal_async_generator():
    """Test minimal async generator functionality"""
    source = """
async def async_gen():
    yield 1

async def compute():
    gen = async_gen()
    value = await gen.__anext__()
    return value
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 1

# Isolated async generator return value test
@pytest.mark.asyncio
async def test_isolated_async_generator_return_value():
    source = """
async def async_gen():
    yield 42
    return 'done'

async def main():
    gen = async_gen()
    try:
        while True:
            value = await gen.__anext__()
            # Consume yielded values, but not necessary for this test
    except StopAsyncIteration as e:
        return e.value  # Capture return value
"""
    execution_result = await execute_async(source, entry_point="main", allowed_modules=["asyncio"])
    assert execution_result.result == 'done', f"Expected 'done', got {execution_result.result}"

# Test basic slicing behavior
@pytest.mark.asyncio
async def test_basic_slicing():
    """Test basic list slicing"""
    source = """
async def get_slice(arr):
    return arr[1:3]

async def main():
    arr = [10, 20, 30, 40]
    sliced = await get_slice(arr)
    return sliced
"""
    result = await execute_async(source, entry_point="main")
    assert result.result == [20, 30]

# Debug test to log execution flow
@pytest.mark.asyncio
async def test_debug_execution_flow():
    """Test to debug execution flow"""
    source = """
async def foo():
    return "success"

async def main():
    result = await foo()
    return result
"""
    result = await execute_async(source, entry_point="main")
    print(f"DEBUG - Execution result: {result}")  # Will show in pytest output
    assert result.result == "success"

# Focused tests for async generator behaviors that failed
@pytest.mark.asyncio
async def test_focused_async_generator_send_value():
    """Test specifically for asend() value propagation"""
    source = """
async def async_gen():
    x = yield 1
    yield x

async def compute():
    gen = async_gen()
    await gen.__anext__()  # First yield
    result = await gen.asend(42)  # Send value
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 42

@pytest.mark.asyncio
async def test_focused_async_generator_throw():
    """Test specifically for athrow() exception handling"""
    source = """
async def async_gen():
    try:
        yield 1
    except ValueError:
        yield "caught"

async def compute():
    gen = async_gen()
    await gen.__anext__()
    result = await gen.athrow(ValueError)
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == "caught"

@pytest.mark.asyncio
async def test_focused_async_generator_empty():
    """Test specifically for empty async generators"""
    source = """
async def async_gen():
    if False:
        yield

async def compute():
    gen = async_gen()
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        return "Empty generator"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == "Empty generator"

# Focused tests for custom object slicing that failed
@pytest.mark.asyncio
async def test_focused_custom_object_slicing():
    """Test specifically for custom object slicing behavior"""
    source = """
class Sliceable:
    def __getitem__(self, key):
        if isinstance(key, slice):
            return f"Slice({key.start},{key.stop},{key.step})"
        return key

async def get_custom_slice(obj):
    return obj[1:5:2]

async def main():
    obj = Sliceable()
    sliced = await get_custom_slice(obj)
    return f"Custom slice: {sliced}"
"""
    result = await execute_async(source, entry_point="main")
    assert result.result == "Custom slice: Slice(1,5,2)"

# Helper test for generator with return
@pytest.mark.asyncio
async def test_focused_generator_with_return():
    """Test specifically for generator with return statement"""
    source = """
def gen_with_return():
    yield 1
    yield 2
    return "done"

async def compute():
    g = gen_with_return()
    first = next(g)
    second = next(g)
    try:
        third = next(g)
    except StopIteration as e:
        return str(e.value)
    return "unexpected"
"""
    result = await execute_async(source, entry_point="compute")
    assert result.result == "done"

# Targeted test for async generator send value propagation
@pytest.mark.asyncio
async def test_targeted_async_generator_send_value():
    """Targeted test for async generator send value propagation"""
    source = """
async def async_gen():
    x = yield 1
    yield x

async def compute():
    gen = async_gen()
    await gen.__anext__()  # Consume first yield
    result = await gen.asend(42)  # Send value and check propagation
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 42, f"Expected 42, got {result.result}"

# Targeted tests for async generator exception handling and return value capture
@pytest.mark.asyncio
async def test_targeted_async_generator_exception_handling():
    """Targeted test for async generator exception handling with athrow"""
    source = """
async def async_gen():
    try:
        yield 1
    except ValueError:
        yield 'caught'

async def compute():
    gen = async_gen()
    await gen.__anext__()
    result = await gen.athrow(ValueError)
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 'caught', f"Expected 'caught', got {result.result}"

@pytest.mark.asyncio
async def test_targeted_async_generator_return_value():
    """Targeted test for async generator return value capture"""
    source = """
async def async_gen():
    yield 42
    return 'done'

async def compute():
    gen = async_gen()
    try:
        while True:
            value = await gen.__anext__()
    except StopAsyncIteration as e:
        return e.value
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 'done', f"Expected 'done', got {result.result}"

# Targeted test for yield assignment in async generator
@pytest.mark.asyncio
async def test_targeted_async_generator_yield_assignment():
    """Targeted test for yield assignment in async generator"""
    source = """
async def async_gen():
    x = yield 1
    return x

async def compute():
    gen = async_gen()
    await gen.__anext__()  # Consume first yield
    result = await gen.asend(42)
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 42, f"Expected 42, got {result.result}"

# Targeted test for send value with assignment in async generator
@pytest.mark.asyncio
async def test_targeted_async_generator_send_value_with_assignment():
    """Targeted test for send value with assignment in async generator"""
    source = """
async def async_gen():
    y = yield 'start'
    yield y + ' received'

async def compute():
    gen = async_gen()
    await gen.__anext__()  # Consume first yield
    result = await gen.asend('value')
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 'value received', f"Expected 'value received', got {result.result}"

# Targeted test for async for loop with async generator
@pytest.mark.asyncio
async def test_targeted_async_for_loop():
    """Targeted test for async for loop with async generator"""
    source = """
async def async_gen():
    yield 1
    yield 2

async def compute():
    sum_value = 0
    async for i in async_gen():
        sum_value += i
    return sum_value
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 3, f"Expected 3, got {result.result}"

# Test for UnboundLocalError with keyword-only parameters to diagnose binding issues.
@pytest.mark.asyncio
async def test_kwonly_param_error_handling():
    """Test for UnboundLocalError with keyword-only parameters to diagnose binding issues."""
    code = """
async def func(*, kwonly_param: int = 42) -> int:
    return kwonly_param
"""
    result = await execute_async(code, entry_point='func', namespace={})
    assert result.result == 42, f"Expected 42, got {result.result}. Error: {result.error}"
    assert result.error is None

# Test for async function with no parameters to diagnose UnboundLocalError and basic execution issues.
@pytest.mark.asyncio
async def test_no_param_function():
    """Test for async function with no parameters to ensure basic execution."""
    code = """
async def func() -> int:
    return 42
"""
    result = await execute_async(code, entry_point='func', namespace={})
    assert result.result == 42, f"Expected 42, got {result.result}. Error: {result.error}"
    assert result.error is None

# Test for async function with required keyword-only parameter to diagnose binding issues.
@pytest.mark.asyncio
async def test_required_kwonly_param():
    """Test for async function with required keyword-only parameter."""
    code = """
async def func(*, kwonly_param: int) -> int:
    return kwonly_param

async def main():
    return await func(kwonly_param=42)
"""
    result = await execute_async(code, entry_point='main', namespace={})
    assert result.result == 42, f"Expected 42, got {result.result}. Error: {result.error}"
    assert result.error is None

# Test for user-defined variable 'kwonly_params' to diagnose UnboundLocalError.
@pytest.mark.asyncio
async def test_user_defined_kwonly_params():
    """Test for user-defined variable 'kwonly_params' to diagnose UnboundLocalError."""
    code = """
async def func():
    kwonly_params = 42
    return kwonly_params
"""
    result = await execute_async(code, entry_point='func', namespace={})
    assert result.result == 42, f"Expected 42, got {result.result}. Error: {result.error}"
    assert result.error is None

# Test for function with keyword-only parameters to diagnose UnboundLocalError.
@pytest.mark.asyncio
async def test_function_with_kwonly_params():
    """Test function with keyword-only parameter to diagnose UnboundLocalError."""
    code = """
async def func(*, kwonly_param: int = 42) -> int:
    return kwonly_param
"""
    result = await execute_async(code, entry_point='func', namespace={})
    assert result.result == 42, f"Expected 42, got {result.result}. Error: {result.error}"
    assert result.error is None
