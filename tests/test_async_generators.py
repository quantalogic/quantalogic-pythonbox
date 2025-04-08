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

@pytest.mark.asyncio
async def test_async_generator_composition():
    source = """
async def inner_gen():
    yield 1
    yield 2

async def outer_gen():
    yield 0
    async for value in inner_gen():
        yield value
    yield 3

async def compute():
    results = []
    async for value in outer_gen():
        results.append(value)
    return results
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [0, 1, 2, 3]

@pytest.mark.asyncio
async def test_complex_exception_handling():
    source = """
async def async_gen():
    try:
        yield 1
        try:
            yield 2
            raise ValueError("inner error")
        except ValueError:
            yield 3
            raise RuntimeError("outer error")
    except RuntimeError as e:
        yield str(e)
    yield 4

async def compute():
    gen = async_gen()
    results = []
    results.append(await gen.__anext__())  # 1
    results.append(await gen.__anext__())  # 2
    results.append(await gen.__anext__())  # 3
    results.append(await gen.__anext__())  # "outer error"
    results.append(await gen.__anext__())  # 4
    return results
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [1, 2, 3, "outer error", 4]

@pytest.mark.asyncio
async def test_concurrent_generators():
    source = """
async def gen1():
    yield 1
    await asyncio.sleep(0.01)
    yield 2

async def gen2():
    yield "a"
    await asyncio.sleep(0.01)
    yield "b"

async def compute():
    g1 = gen1()
    g2 = gen2()
    results = []
    results.append(await g1.__anext__())  # 1
    results.append(await g2.__anext__())  # "a"
    results.append(await g1.__anext__())  # 2
    results.append(await g2.__anext__())  # "b"
    return results
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [1, "a", 2, "b"]

@pytest.mark.asyncio
async def test_generator_patterns_comparison():
    """Test comparing collection, generator and async generator patterns"""
    source = """
def get_collection():
    return [1, 2, 3]

def sync_generator():
    yield 1
    yield 2
    yield 3

async def async_generator():
    yield 1
    await asyncio.sleep(0.01)
    yield 2
    await asyncio.sleep(0.01)
    yield 3

async def compute():
    # Test collection
    collection = get_collection()
    
    # Test sync generator
    gen = sync_generator()
    sync_results = [next(gen), next(gen), next(gen)]
    
    # Test async generator
    async_gen = async_generator()
    async_results = [
        await async_gen.__anext__(),
        await async_gen.__anext__(),
        await async_gen.__anext__()
    ]
    
    return {
        'collection': collection,
        'sync_generator': sync_results,
        'async_generator': async_results
    }
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == {
        'collection': [1, 2, 3],
        'sync_generator': [1, 2, 3],
        'async_generator': [1, 2, 3]
    }

@pytest.mark.asyncio
async def test_generator_lifecycle():
    source = """
async def async_gen():
    try:
        yield 1
        yield 2
    finally:
        yield "cleanup"  # This is not recommended but tests edge case

async def compute():
    gen = async_gen()
    first = await gen.__anext__()
    await gen.aclose()
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        return [first, "closed"]
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == [1, "closed"]

@pytest.mark.asyncio
async def test_asend_behavior():
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
async def test_athrow_behavior():
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
async def test_aclose_behavior():
    """Test specifically for aclose() cleanup"""
    source = """
async def async_gen():
    try:
        yield 1
    finally:
        return "cleanup"

async def compute():
    gen = async_gen()
    await gen.__anext__()
    result = await gen.aclose()
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == "cleanup"

@pytest.mark.asyncio
async def test_nested_generator_flow():
    """Test control flow between nested generators"""
    source = """
async def inner():
    yield "inner"

async def outer():
    yield "outer-start"
    async for x in inner():
        yield x
    yield "outer-end"

async def compute():
    results = []
    async for x in outer():
        results.append(x)
    return results
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    assert result.result == ["outer-start", "inner", "outer-end"]