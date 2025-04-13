import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
async def test_simple_async_generator_send():
    """Minimal test for async generator send functionality"""
    source = """
async def simple_gen():
    x = yield 1
    yield x
    yield 3

async def compute():
    gen = simple_gen()
    first = await gen.__anext__()  # First yield: 1
    second = await gen.asend(2)    # Send 2, yield x
    third = await gen.asend(None)  # Yield 3
    return [first, second, third]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Result: {result.result}, Error: {result.error}")
    return result.result
