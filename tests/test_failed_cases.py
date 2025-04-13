import pytest
from quantalogic_pythonbox import execute_async
import asyncio

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
