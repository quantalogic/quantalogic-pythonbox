import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
async def test_positional_arguments():
    code = '''
async def add(a, b):
    return a + b

async def main():
    return await add(2, 3)
'''
    result = await execute_async(code, entry_point='main')
    assert result.error is None
    assert result.result == 5

@pytest.mark.asyncio
async def test_named_arguments():
    code = '''
async def greet(name, greeting='Hello'):
    return f"{greeting}, {name}!"

async def main():
    return await greet(name='World', greeting='Hi')
'''
    result = await execute_async(code, entry_point='main')
    assert result.error is None
    assert result.result == "Hi, World!"

@pytest.mark.asyncio
async def test_mixed_positional_and_named_arguments():
    code = '''
async def mix(a, b, c=0):
    return a * b + c

async def main():
    return await mix(2, b=4, c=3)
'''
    result = await execute_async(code, entry_point='main')
    assert result.error is None
    assert result.result == 11

@pytest.mark.asyncio
async def test_default_argument_behavior():
    code = '''
async def default(a, b=5):
    return a + b

async def main():
    return await default(7)
'''
    result = await execute_async(code, entry_point='main')
    assert result.error is None
    assert result.result == 12

@pytest.mark.asyncio
async def test_missing_positional_argument_error():
    code = '''
async def foo(a, b):
    return a + b

async def main():
    return await foo(1)
'''
    result = await execute_async(code, entry_point='main')
    assert result.error is not None
    assert 'missing 1 required positional argument' in result.error

@pytest.mark.asyncio
async def test_too_many_positional_arguments_error():
    code = '''
async def foo(a, b):
    return a + b

async def main():
    return await foo(1, 2, 3)
'''
    result = await execute_async(code, entry_point='main')
    assert result.error is not None
    assert 'positional arguments' in result.error
