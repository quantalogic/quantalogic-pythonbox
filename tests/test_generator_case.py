import pytest
from quantalogic_pythonbox.interpreter import PythonBox

@pytest.mark.asyncio
async def test_sync_generator_return_in_try():
    code = """
def gen():
    yield 1
    yield 2
    return 'finished'

async def main():
    g = gen()
    result = None
    try:
        while True:
            next(g)
    except StopIteration as e:
        result = e.value
    return result
"""
    interpreter = PythonBox()
    result = await interpreter.execute_async(code)
    assert result.result == 'finished', f"Expected 'finished', got {result.result}"

@pytest.mark.asyncio
async def test_async_generator_return_after_yield():
    code = """
async def agen():
    yield 1
    yield 2
    return 'complete'

async def main():
    g = agen()
    async for _ in g:
        pass
    return g.agenerator.gi_frame.f_locals.get('__return__', None)
"""
    interpreter = PythonBox()
    result = await interpreter.execute_async(code)
    assert result.result == 'complete', f"Expected 'complete', got {result.result}"

@pytest.mark.asyncio
async def test_nested_generator_return():
    code = """
def inner_gen():
    yield 1
    return 'inner_done'

async def outer_gen():
    g = inner_gen()
    yield next(g)
    try:
        next(g)
    except StopIteration as e:
        return e.value

async def main():
    g = outer_gen()
    result = None
    async for x in g:
        result = x
    return result
"""
    interpreter = PythonBox()
    result = await interpreter.execute_async(code)
    assert result.result == 'inner_done', f"Expected 'inner_done', got {result.result}"

@pytest.mark.asyncio
async def test_async_generator_with_assignment_and_return():
    code = """
async def agen():
    x = yield 1
    return x * 2

async def main():
    g = agen()
    await g.asend(None)
    try:
        await g.asend(21)
    except StopAsyncIteration as e:
        return e.value
"""
    interpreter = PythonBox()
    result = await interpreter.execute_async(code)
    assert result.result == 42, f"Expected 42, got {result.result}"