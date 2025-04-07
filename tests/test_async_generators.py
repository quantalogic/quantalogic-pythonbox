import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
async def test_simple_async_generator():
    source = """
async def async_gen():
    for i in range(3):
        yield i
        await asyncio.sleep(0.01)

async def compute():
    results = []
    async for value in async_gen():
        results.append(value)
    return results
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [0, 1, 2]

@pytest.mark.asyncio
async def test_async_generator_with_exception():
    source = """
async def async_gen():
    yield 1
    raise ValueError("test error")

async def compute():
    gen = async_gen()
    first = await gen.__anext__()
    try:
        await gen.__anext__()
    except ValueError as e:
        return str(e)
    return "no error"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == "test error"