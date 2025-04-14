import pytest
from quantalogic_pythonbox import execute_async
import math

# Test fixtures
@pytest.fixture
def simple_async_code():
    return '''
async def sinus(x: float) -> float:
    return float(math.sin(x))

async def compute() -> float:
    result = await sinus(10) + 8.4
    return result

async def main() -> float:
    return await compute()
'''

@pytest.fixture
def simple_sync_code():
    return '''
async def sinus(x: float) -> float:
    return float(math.sin(x))

async def main():
    step5_sin_value = await sinus(x=4.7)
    step5_result = step5_sin_value + 8.1
    return f"Task completed: {step5_result}"
'''

@pytest.mark.asyncio
class TestBasicFunctionality:
    """Tests for basic async/sync function execution"""
    
    async def test_simple_async_function(self, simple_async_code):
        """Verify basic async function execution with math operations"""
        result = await execute_async(
            simple_async_code,
            entry_point='main',
            namespace={'math': math}
        )
        assert result.result == pytest.approx(math.sin(10) + 8.4)

    async def test_simple_sync_function(self, simple_sync_code):
        """Verify sync function execution with string formatting"""
        result = await execute_async(
            simple_sync_code,
            entry_point='main',
            namespace={'math': math}
        )
        expected = f"Task completed: {math.sin(4.7) + 8.1}"
        assert result.result == expected

@pytest.mark.asyncio
class TestSearchEnumerations:
    """Tests for search result enumeration scenarios"""
    
    @pytest.mark.parametrize("input_data,expected", [
        ([
            {"title": "Story 1", "score": 42},
            {"title": "Story 2", "score": 31},
            {"title": "Story 3", "score": 27}
        ], """
        1. Story 1 (Score: 42)
        2. Story 2 (Score: 31)
        3. Story 3 (Score: 27)"""),
        ([
            {"title": "Item A", "score": 99},
            {"title": "Item B", "score": 101}
        ], """
        1. Item A (Score: 99)
        2. Item B (Score: 101)""")
    ])
    async def test_enumerate_search_results(self, input_data, expected):
        """Verify proper enumeration of search results"""
        code = f'''
async def search_stories(query: str):
    return {input_data}

async def main():
    results = await search_stories("python")
    report = ""
    for index, story in enumerate(results, 1):
        report += f"\n{index}. {story['title']} (Score: {story['score']})"
    return report
'''
        result = await execute_async(code, entry_point='main')
        assert result.result.strip() == expected.strip()

    async def test_enumerate_empty_results(self):
        """Verify handling of empty search results"""
        result = await execute_async('''
async def search_stories(query: str):
    return []

async def main():
    results = await search_stories("python")
    return "No results found" if not results else ""
''', entry_point='main')
        assert result.result == "No results found"

    async def test_enumerate_single_result(self):
        """Verify single result enumeration"""
        result = await execute_async('''
async def search_stories(query: str):
    return [{"title": "Lone Story", "score": 99}]

async def main():
    results = await search_stories("python")
    report = ""
    for idx, story in enumerate(results, 1):
        report += f"\n{idx}. {story['title']} (Score: {story['score']})"
    return report
''', entry_point='main')
        assert result.result.strip() == """
1. Lone Story (Score: 99)""".strip()

    async def test_enumerate_missing_fields(self):
        """Verify graceful handling of missing fields"""
        result = await execute_async('''
async def search_stories(query: str):
    return [
        {"title": "Complete Story", "score": 50},
        {"title": "Missing Score"},
        {"score": 30},
        {}
    ]

async def main():
    results = await search_stories("python")
    report = ""
    for idx, story in enumerate(results, 1):
        report += f"\n{idx}. {story.get('title', 'Untitled')} (Score: {story.get('score', 'N/A')})"
    return report
''', entry_point='main')
        expected = """
1. Complete Story (Score: 50)
2. Missing Score (Score: N/A)
3. Untitled (Score: 30)
4. Untitled (Score: N/A)"""
        assert result.result.strip() == expected.strip()

    async def test_enumerate_large_results(self):
        """Verify handling of large result sets with truncation"""
        result = await execute_async('''
async def search_stories(query: str):
    return [{"title": f"Story {i}", "score": i} for i in range(100)]

async def main():
    results = await search_stories("python")
    report = f"Found {len(results)} stories"
    for idx, story in enumerate(results[:5], 1):
        report += f"\n{idx}. {story['title']} (Score: {story['score']})"
    if len(results) > 5:
        report += "\n... (truncated)"
    return report
''', entry_point='main')
        assert "Found 100 stories" in result.result
        assert "1. Story 0 (Score: 0)" in result.result
        assert "... (truncated)" in result.result