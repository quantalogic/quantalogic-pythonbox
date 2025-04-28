import pytest
import inspect

pytest.skip("Example tests for async generators; skip by default.", allow_module_level=True)
pytestmark = pytest.mark.asyncio

# Example async generator tests to spot potential issues

async def test_simple_async_generator_list():
    async def gen():
        yield 1
        yield 2
        raise StopAsyncIteration(3)

    agen = gen()
    result = []
    async for v in agen:
        result.append(v)
    assert result == [1, 2]

    # After exhaustion, StopAsyncIteration.value should carry return value
    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

async def test_asend_functionality():
    async def gen():
        x = yield 'start'
        yield x
        raise StopAsyncIteration('done')

    agen = gen()
    first = await agen.__anext__()
    assert first == 'start'

    second = await agen.asend('foo')
    assert second == 'foo'

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

async def test_athrow_caught_in_generator():
    async def gen():
        try:
            yield 1
        except ValueError:
            yield 2
        yield 3

    agen = gen()
    assert await agen.__anext__() == 1
    second = await agen.athrow(ValueError('boom'))
    assert second == 2
    third = await agen.__anext__()
    assert third == 3

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

async def test_athrow_uncaught_propagates():
    async def gen():
        yield 1

    agen = gen()
    assert await agen.__anext__() == 1

    with pytest.raises(ValueError):
        await agen.athrow(ValueError('unexpected'))

async def test_aclose_triggers_finally():
    flags = {}

    async def gen():
        try:
            yield 'a'
        finally:
            flags['closed'] = True

    agen = gen()
    assert await agen.__anext__() == 'a'
    await agen.aclose()
    assert flags.get('closed', False) is True

async def test_async_enumerate_behavior():
    async def gen():
        yield 'a'
        yield 'b'

    result = []
    idx = 0
    async for val in gen():
        result.append((idx, val))
        idx += 1
    assert result == [(0, 'a'), (1, 'b')]

async def test_multiple_athrows_and_return():
    async def gen():
        try:
            yield 1
        except KeyError:
            yield 'caught_key'
        try:
            yield 2
        except ValueError:
            yield 'caught_value'
        raise StopAsyncIteration('end')

    agen = gen()
    assert await agen.__anext__() == 1
    assert await agen.athrow(KeyError('k')) == 'caught_key'
    assert await agen.__anext__() == 2
    assert await agen.athrow(ValueError('v')) == 'caught_value'

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

async def test_send_after_exhaustion_raises():
    async def gen():
        yield 1

    agen = gen()
    assert await agen.__anext__() == 1
    # Exhaust generator
    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

    with pytest.raises(RuntimeError):
        await agen.asend(None)

async def test_aclose_idempotent():
    flags = {'count': 0}

    async def gen():
        try:
            yield 1
        finally:
            flags['count'] += 1

    agen = gen()
    assert await agen.__anext__() == 1
    await agen.aclose()
    # Calling aclose again should not trigger finally twice
    await agen.aclose()
    assert flags['count'] == 1

# Test inspect API

def test_inspect_api():
    async def gen():
        yield 1

    assert inspect.isasyncgenfunction(gen)
    agen = gen()
    assert inspect.isasyncgen(agen)
