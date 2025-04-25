import pytest
from quantalogic_pythonbox import execute_async
import datetime

@pytest.mark.asyncio
async def test_step3_sort_by_datetime():
    """Verify sorting of items by date in descending order using async key function."""
    code = '''
import datetime

async def get_datetime(item):
    return datetime.datetime.strptime(item["date"], "%Y-%m-%d")

async def main():
    step3_all_items = [
        {"name": "A", "date": "2025-01-01"},
        {"name": "B", "date": "2024-12-31"},
        {"name": "C", "date": "2025-02-15"}
    ]
    step3_all_items.sort(key=get_datetime, reverse=True)
    step3_table = step3_all_items
    return step3_table
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={'datetime': datetime}
    )
    expected = [
        {"name": "C", "date": "2025-02-15"},
        {"name": "A", "date": "2025-01-01"},
        {"name": "B", "date": "2024-12-31"}
    ]
    assert result.result == expected

@pytest.mark.asyncio
async def test_async_comparison_numeric():
    """Verify numeric comparison operations in async context."""
    code = '''
async def get_num(x):
    return x

async def main():
    a = await get_num(10)
    b = await get_num(5)
    return [a > b, a < b, a == 10, b != 10]
'''
    result = await execute_async(
        code,
        entry_point='main'
    )
    assert result.result == [True, False, True, True]

@pytest.mark.asyncio
async def test_async_comparison_string():
    """Verify string comparison operations in async context."""
    code = '''
async def get_str(s):
    return s

async def main():
    a = await get_str("apple")
    b = await get_str("banana")
    return a < b
'''
    result = await execute_async(
        code,
        entry_point='main'
    )
    assert result.result is True
