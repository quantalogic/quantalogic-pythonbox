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

@pytest.mark.asyncio
async def test_async_generator_send_value():
    source = """
async def async_gen():
    x = yield 1
    yield x
    yield 3

async def compute():
    gen = async_gen()
    first = await gen.__anext__()  # First yield: 1
    second = await gen.asend(2)    # Send 2, yield x
    third = await gen.asend(None)  # Yield 3
    return [first, second, third]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [1, 2, 3]

@pytest.mark.asyncio
async def test_async_generator_throw():
    source = """
async def async_gen():
    try:
        yield 1
        yield 2
    except ValueError:
        yield 3
    yield 4

async def compute():
    gen = async_gen()
    first = await gen.__anext__()   # First yield: 1
    await gen.athrow(ValueError)    # Throw ValueError
    last = await gen.__anext__()    # Yield 4
    return [first, last]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [1, 4]

@pytest.mark.asyncio
async def test_async_generator_close():
    source = """
async def async_gen():
    try:
        yield 1
        yield 2
    finally:
        return "Closed"

async def compute():
    gen = async_gen()
    first = await gen.__anext__()   # First yield: 1
    await gen.aclose()              # Close the generator
    return first
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == 1

@pytest.mark.asyncio
async def test_async_generator_empty():
    source = """
async def async_gen():
    # Empty generator
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