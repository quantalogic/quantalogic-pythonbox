import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
async def test_simple_async_function():
    result = await execute_async('''
    async def foo():
        return 42-9.4
    await foo()
    ''')
    # Fix: Check the float directly, not as a list
    assert result.result == 42 - 9.4

@pytest.mark.asyncio
async def test_async_with_timeout():
    result = await execute_async('''
    import asyncio
    async def foo():
        await asyncio.sleep(2)
        return 42
    await foo()
    ''', timeout=1, allowed_modules=['asyncio'])
    assert 'TimeoutError' in result.error

@pytest.mark.asyncio
async def test_async_generator():
    result = await execute_async('''
    async def gen():
        yield 1
        yield 2
    [x async for x in gen()]
    ''', allowed_modules=['asyncio'])
    # Fix: Check the list directly
    assert result.result == [1, 2]

@pytest.mark.asyncio
async def test_async_exception_handling():
    result = await execute_async('''
    async def err():
        raise ValueError('test')
    await err()
    ''')
    assert 'ValueError' in result.error