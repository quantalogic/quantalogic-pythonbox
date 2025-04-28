import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
class TestAsyncGeneratorFull:
    async def test_asend_value(self):
        code = """
async def gen():
    received = yield 1
    yield received

async def main():
    g = gen()
    first = await g.__anext__()
    second = await g.asend(42)
    return [first, second]
"""
        result = await execute_async(code, entry_point="main")
        assert result.result == [1, 42]
        assert result.error is None

    async def test_athrow_handling(self):
        code = """
async def gen():
    try:
        yield 1
    except ValueError as e:
        yield f"caught {e}"
    yield 2

async def main():
    g = gen()
    first = await g.__anext__()
    second = await g.athrow(ValueError("err"))
    third = await g.__anext__()
    return [first, second, third]
"""
        result = await execute_async(code, entry_point="main")
        assert result.result == [1, "caught err", 2]
        assert result.error is None

    async def test_return_without_value(self):
        code = """
async def gen():
    yield 1
    return
    yield 2

async def main():
    result = []
    async for x in gen():
        result.append(x)
    return result
"""
        result = await execute_async(code, entry_point="main")
        assert result.result == [1]
        assert result.error is None

    async def test_return_with_value_syntax_error(self):
        code = """
# return with value in async generator is invalid syntax
async def gen():
    yield 1
    return 2

async def main():
    return [x async for x in gen()]
"""
        result = await execute_async(code, entry_point="main")
        assert "SyntaxError" in result.error

    async def test_aclose_finalization(self):
        code = """
log = []
async def gen():
    try:
        yield 1
    finally:
        log.append("closed")

async def main():
    g = gen()
    first = await g.__anext__()
    await g.aclose()
    return [first, log]
"""
        result = await execute_async(code, entry_point="main")
        assert result.result == [1, ["closed"]]
        assert result.error is None

    async def test_inspect_api(self):
        code = """
import inspect
async def gen():
    yield 1
async def main():
    return (inspect.isasyncgenfunction(gen), inspect.isasyncgen(gen()))
"""
        result = await execute_async(code, entry_point="main")
        assert result.result == (True, True)
        assert result.error is None
