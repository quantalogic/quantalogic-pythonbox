import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
class TestAsyncFunctions:
    async def test_simple_async_function(self):
        code = """
async def foo():
    return 1 + 2
        """
        result = await execute_async(code, entry_point="foo")
        assert result.result == 3
        assert result.error is None

    @pytest.mark.parametrize("sleep_time, timeout, expect_timeout", [
        (1, 2, False),
        (2, 1, True),
    ])
    async def test_async_with_timeout(self, sleep_time, timeout, expect_timeout):
        code = """
import asyncio

async def foo():
    await asyncio.sleep({})
    return 42
        """.format(sleep_time)
        result = await execute_async(code, entry_point="foo", timeout=timeout)
        if expect_timeout:
            assert "TimeoutError" in result.error
            assert result.result is None
        else:
            assert result.result == 42
            assert result.error is None

    async def test_async_append_simple(self):
        code = """
async def main():
    lst = []
    lst.append(1)
    lst.append(2)
    return lst
        """
        result_obj = await execute_async(code, entry_point="main")
        assert result_obj.result == [1, 2], f"Expected [1, 2], but got {result_obj.result}"
        assert result_obj.error is None

@pytest.mark.asyncio
class TestAsyncGenerators:
    async def test_async_generator(self):
        code = """
async def gen():
    yield 1
    yield 2

async def main():
    result = []
    async for value in gen():
        result.append(value)
    return result
        """
        result = await execute_async(code, entry_point="main")
        assert result.result == [1, 2]
        assert result.error is None

    async def test_empty_async_generator(self):
        code = """
async def gen():
    if False:
        yield 1

async def main():
    return [x async for x in gen()]
        """
        result = await execute_async(code, entry_point="main")
        assert result.result == []
        assert result.error is None

    async def test_async_generator_append(self):
        code = """
async def gen():
    lst = []
    lst.append(1)  # Simple append
    yield lst
    lst.append(2)  # Append after yield
    yield lst

async def main():
    result = []
    async for value in gen():
        result.extend(value)
    return result
        """
        result_obj = await execute_async(code, entry_point="main")
        assert result_obj.result == [1, 1, 2], f"Expected [1, 1, 2], but got {result_obj.result}"
        assert result_obj.error is None

    async def test_async_generator_anext_manual(self):
        code = """
async def gen():
    yield 1
    yield 2

async def main():
    gen_obj = gen()
    result = []
    try:
        while True:
            value = await gen_obj.__anext__()
            result.append(value)
    except StopAsyncIteration:
        return result
        """
        result_obj = await execute_async(code, entry_point="main")
        assert result_obj.result == [1, 2], f"Expected [1, 2], but got {result_obj.result}"
        assert result_obj.error is None

    async def test_async_generator_anext_existence(self):
        code = """
async def gen():
    yield 1
    yield 2

async def main():
    gen_obj = gen()
    assert hasattr(gen_obj, '__anext__'), "Generator object missing __anext__ method"
    assert callable(gen_obj.__anext__), "__anext__ method is not callable"
    result = []
    try:
        while True:
            value = await gen_obj.__anext__()
            result.append(value)
    except StopAsyncIteration:
        return result
    """
        result_obj = await execute_async(code, entry_point="main")
        assert result_obj.result == [1, 2], f"Expected [1, 2], but got {result_obj.result}"
        assert result_obj.error is None

@pytest.mark.asyncio
class TestErrorHandling:
    @pytest.mark.parametrize("exception", [
        ValueError("test"),
        TypeError("type error"),
        RuntimeError("runtime error"),
    ])
    async def test_async_exception_handling(self, exception):
        exc_type = exception.__class__.__name__
        exc_msg = str(exception)
        code = """
async def foo():
    raise {}("{}")
        """.format(exc_type, exc_msg)
        result = await execute_async(code, entry_point="foo")
        assert result.result is None
        assert exc_msg in result.error
        assert exc_type in result.error

    async def test_undefined_variable(self):
        code = """
async def foo():
    return undefined_variable
        """
        result = await execute_async(code, entry_point="foo")
        assert result.result is None
        assert "NameError" in result.error
        assert "undefined_variable" in result.error

@pytest.mark.asyncio
class TestRealWorldScenarios:
    async def test_async_enumerate(self):
        code = """
async def async_enumerate(async_iterable, start=0):
    i = start
    async for item in async_iterable:
        yield (i, item)
        i += 1

async def gen():
    yield "a"
    yield "b"
    yield "c"

async def main():
    return [(i, x) async for i, x in async_enumerate(gen())]
        """
        result = await execute_async(code, entry_point="main")
        assert result.result == [(0, 'a'), (1, 'b'), (2, 'c')]
        assert result.error is None

    async def test_async_sort_with_lambda(self):
        code = """
import asyncio

async def async_key(item):
    await asyncio.sleep(0)
    return item

async def main():
    data = [4, 2, 3, 1]
    result = [x async for x in sorted(data, key=async_key)]
    return result
        """
        result = await execute_async(code, entry_point="main")
        assert result.result == [1, 2, 3, 4]
        assert result.error is None

    async def test_hn_processing(self):
        code = """
import asyncio

class HNItem:
    def __init__(self, title, score, url):
        self.title = title
        self.score = score
        self.url = url

async def fetch_item(item_id):
    await asyncio.sleep(0)
    return HNItem(f"Item {item_id}", item_id * 10, f"https://example.com/{item_id}")

async def process_items(item_ids):
    items = [await fetch_item(i) for i in item_ids]
    result = ""
    for item in sorted(items, key=lambda x: (-x.score, x.title)):
        result += f"Title: {item.title}, Score: {item.score}\\n"
        result += f"URL: {item.url}\\n"
    return result

async def main():
    result = await process_items([3, 1, 2])
    return result.splitlines()[0]
        """
        result = await execute_async(code, entry_point="main")
        assert result.result == "Title: Item 3, Score: 30"
        assert result.error is None

    async def test_async_sort_error_and_solution(self):
        error_code = """
import asyncio

class Item:
    def __init__(self, value):
        self.value = value

    async def get_value(self):
        await asyncio.sleep(0)
        return self.value

async def main():
    items = [Item(3), Item(1), Item(2)]
    sorted_items = sorted(items, key=lambda x: x.get_value())
    return [item.value for item in sorted_items]
        """
        error_result = await execute_async(error_code, entry_point="main")
        assert "'<' not supported between instances of 'coroutine' and 'coroutine'" in error_result.error

        fixed_code = """
import asyncio

class Item:
    def __init__(self, value):
        self.value = value

    async def get_value(self):
        await asyncio.sleep(0)
        return self.value

async def main():
    items = [Item(3), Item(1), Item(2)]
    values = [await item.get_value() for item in items]
    sorted_items = sorted(items, key=lambda x: values[items.index(x)])
    return [item.value for item in sorted_items]
        """
        result = await execute_async(fixed_code, entry_point="main")
        assert result.result == [1, 2, 3]
        assert result.error is None