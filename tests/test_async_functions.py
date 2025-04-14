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
    assert result.result == 32.6

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
        
    async def compute():
        # Instead of a list comprehension, explicitly collect items
        results = []
        # Use a generator once to avoid duplicates
        async for value in gen():
            results.append(value)
        return results
    ''', entry_point="compute", allowed_modules=['asyncio'])
    assert result.result == [1, 2]

@pytest.mark.asyncio
async def test_async_exception_handling():
    result = await execute_async('''
    async def err():
        raise ValueError('test')
    await err()
    ''')
    assert 'ValueError' in result.error

@pytest.mark.asyncio
async def test_async_enumerate():
    result = await execute_async('''
    async def async_enumerate(async_iterable, start=0):
        i = start
        async for item in async_iterable:
            yield i, item
            i += 1

    async def gen():
        yield 'a'
        yield 'b'
        yield 'c'
    
    [(i, x) async for i, x in async_enumerate(gen())]
    ''')
    assert result.result == [(0, 'a'), (1, 'b'), (2, 'c')]

@pytest.mark.asyncio
async def test_async_sort_with_lambda():
    result = await execute_async('''
    import asyncio
    async def get_value(x):
        await asyncio.sleep(0.01)
        return x
    
    items = [3, 1, 4, 2]
    keys = [await get_value(x) for x in items]
    sorted_items = [x for _, x in sorted(zip(keys, items), key=lambda pair: pair[0])]
    sorted_items
    ''', allowed_modules=['asyncio'])
    assert result.result == [1, 2, 3, 4]

@pytest.mark.asyncio
async def test_async_hn_processing():
    result = await execute_async(r'''
import asyncio

class HNItem:
    def __init__(self, title, url, score):
        self.title = title
        self.url = url
        self.score = score

async def get_hn_items(item_type, num_items):
    # Mock data for testing
    mock_items = [
        HNItem("AI Breakthrough", "https://example.com/ai", 250),
        HNItem("Python 4.0 Released", "https://example.com/python", 180),
        HNItem("Quantum Computing", "https://example.com/quantum", 300)
    ]
    await asyncio.sleep(0.01)
    return mock_items[:num_items]

async def main():
    step1_item_type = "top"
    step1_num_items = 3
    step1_hn_items = await get_hn_items(item_type=step1_item_type, num_items=step1_num_items)
    
    # Sort items by score in descending order
    step1_sorted_items = sorted(step1_hn_items, key=lambda item: item.score, reverse=True)[:step1_num_items]
    
    # Create a markdown report
    step1_markdown_report = "# Top Hacker News Articles\n\n"
    for item in step1_sorted_items:
        step1_markdown_report += f"## [{item.title}]({item.url})\n"
        step1_markdown_report += f"Score: {item.score}\n\n"
    
    return step1_markdown_report

await main()
''', allowed_modules=['asyncio'])
    
    # Debug output if the test fails
    if result.error:
        print(f"Error: {result.error}")
    
    # Check if the report contains expected content
    assert result.result is not None, "Result should not be None"
    assert "# Top Hacker News Articles" in result.result
    assert "Quantum Computing" in result.result
    assert "Score: 300" in result.result

@pytest.mark.asyncio
async def test_async_sort_error_and_solution():
    # First show the error case
    error_result = await execute_async('''
    import asyncio
    
    class HNItem:
        def __init__(self, title, score):
            self.title = title
            self.score = score
    
        async def get_score(self):
            await asyncio.sleep(0.01)
            return self.score
    
    items = [HNItem("Item A", 300), HNItem("Item B", 200), HNItem("Item C", 400)]
    
    # This will fail because the lambda returns a coroutine
    try:
        sorted(items, key=lambda x: x.get_score(), reverse=True)
    except TypeError as e:
        error = str(e)
    
    error
    ''', allowed_modules=['asyncio'])
    
    assert "'<' not supported between instances of 'coroutine' and 'coroutine'" in error_result.result
    
    # Then show the correct solution
    solution_result = await execute_async('''
    import asyncio
    
    class HNItem:
        def __init__(self, title, score):
            self.title = title
            self.score = score
    
        async def get_score(self):
            await asyncio.sleep(0.01)
            return self.score
    
    items = [HNItem("Item A", 300), HNItem("Item B", 200), HNItem("Item C", 400)]
    scores = [await item.get_score() for item in items]
    sorted_pairs = sorted(zip(scores, items), key=lambda pair: pair[0], reverse=True)
    sorted_items = [pair[1] for pair in sorted_pairs]
    [item.title for item in sorted_items]
    ''', allowed_modules=['asyncio'])
    
    assert solution_result.result == ["Item C", "Item A", "Item B"]