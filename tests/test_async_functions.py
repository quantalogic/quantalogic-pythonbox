import pytest
from quantalogic_pythonbox import execute_async
import math

# Fixtures for common test components
@pytest.fixture
def simple_async_code():
    return '''
async def foo():
    return 42-9.4
await foo()
'''

@pytest.fixture
def timeout_code():
    return '''
import asyncio
async def foo():
    await asyncio.sleep(2)
    return 42
await foo()
'''

@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test suite for basic async function execution"""
    
    async def test_simple_async_function(self, simple_async_code):
        """Verify basic async function execution returns correct value"""
        result = await execute_async(simple_async_code)
        assert result.result == pytest.approx(32.6)
        assert result.error is None

    @pytest.mark.parametrize("sleep_time,timeout,should_timeout", [
        (1, 2, False),
        (2, 1, True)
    ])
    async def test_async_with_timeout(self, timeout_code, sleep_time, timeout, should_timeout):
        """Verify timeout behavior for async functions"""
        modified_code = timeout_code.replace('sleep(2)', f'sleep({sleep_time})')
        result = await execute_async(modified_code, timeout=timeout, allowed_modules=['asyncio'])
        
        if should_timeout:
            assert 'TimeoutError' in result.error
        else:
            assert result.result == 42
            assert result.error is None

@pytest.mark.asyncio
class TestAsyncGenerators:
    """Test suite for async generator functionality"""
    
    async def test_async_generator(self):
        """Verify async generators properly yield values"""
        result = await execute_async('''
async def gen():
    yield 1
    yield 2
    
async def compute():
    results = []
    async for value in gen():
        results.append(value)
    return results
''', entry_point="compute", allowed_modules=['asyncio'])
        assert result.result == [1, 2]
        assert result.error is None

    async def test_empty_async_generator(self):
        """Verify empty async generators behave correctly"""
        result = await execute_async('''
async def gen():
    if False:
        yield
    
async def compute():
    return [x async for x in gen()]
''', entry_point="compute")
        assert result.result == []
        assert result.error is None

@pytest.mark.asyncio
class TestErrorHandling:
    """Test suite for error handling in async execution"""
    
    @pytest.mark.parametrize("exception_type", [
        "ValueError('test')",
        "TypeError('type error')",
        "RuntimeError('runtime error')"
    ])
    async def test_async_exception_handling(self, exception_type):
        """Verify different exception types are properly caught"""
        result = await execute_async(f'''
async def err():
    raise {exception_type}
await err()
''')
        assert exception_type.split('(')[0] in result.error

    async def test_undefined_variable(self):
        """Verify proper error reporting for undefined variables"""
        result = await execute_async('''
async def foo():
    return undefined_var
await foo()
''')
        assert 'NameError' in result.error
        assert 'undefined_var' in result.error

# Additional test classes for more complex scenarios...

@pytest.mark.asyncio
class TestRealWorldScenarios:
    """Test more complex real-world async patterns"""
    
    async def test_async_enumerate(self):
        """Verify async enumerate implementation"""
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

    async def test_async_sort_with_lambda(self):
        """Verify async sorting with lambda functions"""
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

    async def test_hn_processing(self):
        """Verify complex async data processing pipeline"""
        result = await execute_async(r'''
import asyncio

class HNItem:
    def __init__(self, title, url, score):
        self.title = title
        self.url = url
        self.score = score

async def get_hn_items(item_type, num_items):
    mock_items = [
        HNItem("AI Breakthrough", "https://example.com/ai", 250),
        HNItem("Python 4.0 Released", "https://example.com/python", 180),
        HNItem("Quantum Computing", "https://example.com/quantum", 300)
    ]
    await asyncio.sleep(0.01)
    return mock_items[:num_items]

async def main():
    items = await get_hn_items("top", 3)
    sorted_items = sorted(items, key=lambda item: item.score, reverse=True)
    report = "# Top Hacker News Articles\n\n"
    for item in sorted_items:
        report += f"## [{item.title}]({item.url})\n"
        report += f"Score: {item.score}\n\n"
    return report

await main()
''', allowed_modules=['asyncio'])
        
        assert "# Top Hacker News Articles" in result.result
        assert "Quantum Computing" in result.result
        assert "Score: 300" in result.result
        assert result.error is None

    async def test_async_sort_error_and_solution(self):
        """Verify proper async sorting with async key functions"""
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
sorted_items = sorted(items, key=lambda x: x.get_score(), reverse=True)
[item.title for item in sorted_items]
''', allowed_modules=['asyncio'])
        assert error_result.result == ["Item C", "Item A", "Item B"]
        
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

    async def test_hn_items_pipeline(self):
        """Verify Hacker News items pipeline and table formatting"""
        code = r'''
import heapq

class FakeItem:
    def __init__(self, title, score, by):
        self.title = title
        self.score = score
        self.by = by

class quantalogic_hacker_news:
    @staticmethod
    async def get_hn_items(item_type, per_page, page):
        items = []
        for i in range(1, per_page + 1):
            items.append(FakeItem(f"Title{page}_{i}", page * 100 + i, f"Author{page}_{i}"))
        return items

async def main():
    step1_items = []
    for page in range(1, 3):
        step1_response = await quantalogic_hacker_news.get_hn_items(item_type='top', per_page=2, page=page)
        if step1_response:
            step1_items.extend(step1_response)
        else:
            return {
                'status': 'inprogress',
                'result': 'Failed to retrieve Hacker News items.',
                'next_step': 'Retry fetching Hacker News items.'
            }
    step2_sorted_items = heapq.nlargest(4, step1_items, key=lambda item: item.score)

    step3_table_header = "| Title | Score | Author |\n|---|---|---|"
    step3_table_rows = []
    for item in step2_sorted_items:
        step3_table_rows.append(f"| {item.title} | {item.score} | {item.by} |")
    step4_table = step3_table_header + "\n" + "\n".join(step3_table_rows)

    return {
        'status': 'completed',
        'result': step4_table,
    }
'''
        result = await execute_async(code, entry_point='main', allowed_modules=['heapq'])
        assert result.error is None
        assert result.result['status'] == 'completed'
        table = result.result['result']
        assert table.startswith("| Title | Score | Author |")
        assert "| Title1_1 | 101 | Author1_1 |" in table
        assert "| Title2_2 | 202 | Author2_2 |" in table