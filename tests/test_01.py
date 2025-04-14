import pytest
from quantalogic_pythonbox import execute_async
import math

@pytest.mark.asyncio
async def test_simple_async_function():
    code = '''
async def sinus(x: float) -> float:
    return float(math.sin(x))

async def compute() -> float:
    result = await sinus(10) + 8.4
    return result

async def main() -> float:
    return await compute()
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={'math': math}
    )
    assert result.result == math.sin(10) + 8.4

@pytest.mark.asyncio
async def test_simple_sync_function():
    code = '''
async def sinus(x: float) -> float:
    return float(math.sin(x))

async def main():
    # Calculate sin(4.7)
    step5_sin_value = await sinus(x=4.7)
    step5_result = step5_sin_value + 8.1
    return f"Task completed: {step5_result}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={'math': math}
    )
    assert result.result == f"Task completed: {math.sin(4.7) + 8.1}"

@pytest.mark.asyncio
async def test_enumerate_search_results():
    code = '''
async def search_stories(query: str):
    # Simulate search results
    return [
        {"title": "Story 1", "score": 42},
        {"title": "Story 2", "score": 31},
        {"title": "Story 3", "score": 27}
    ]

async def main():
    step3_search_results = await search_stories("python")
    step3_report = ""
    for index, story in enumerate(step3_search_results):
        step3_report += f"""
        {index + 1}. {story['title']} (Score: {story['score']})"""
    return step3_report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    expected = """
        1. Story 1 (Score: 42)
        2. Story 2 (Score: 31)
        3. Story 3 (Score: 27)"""
    assert result.result.strip() == expected.strip()

@pytest.mark.asyncio
async def test_enumerate_sorted_by_score():
    code = '''
async def search_stories(query: str):
    return [
        {"title": "Low Score", "score": 10},
        {"title": "High Score", "score": 95},
        {"title": "Medium Score", "score": 42}
    ]

async def main():
    results = await search_stories("python")
    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    report = ""
    for idx, story in enumerate(sorted_results, 1):
        report += f"""
        {idx}. {story['title']} (Score: {story['score']})"""
    return report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    expected = """
        1. High Score (Score: 95)
        2. Medium Score (Score: 42)
        3. Low Score (Score: 10)"""
    assert result.result.strip() == expected.strip()

@pytest.mark.asyncio
async def test_enumerate_sorted_by_title():
    code = '''
async def search_stories(query: str):
    return [
        {"title": "Zebra", "score": 30},
        {"title": "Apple", "score": 50},
        {"title": "Banana", "score": 20}
    ]

async def main():
    results = await search_stories("python")
    # Sort by title ascending
    sorted_results = sorted(results, key=lambda x: x['title'])
    report = ""
    for idx, story in enumerate(sorted_results, 1):
        report += f"""
        {idx}. {story['title']} (Score: {story['score']})"""
    return report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    expected = """
        1. Apple (Score: 50)
        2. Banana (Score: 20)
        3. Zebra (Score: 30)"""
    assert result.result.strip() == expected.strip()

@pytest.mark.asyncio
async def test_enumerate_empty_results():
    code = '''
async def search_stories(query: str):
    return []

async def main():
    results = await search_stories("python")
    report = "No results found" if not results else ""
    for idx, story in enumerate(results, 1):
        report += f"""
        {idx}. {story.get('title', 'Untitled')} (Score: {story.get('score', 'N/A')})"""
    return report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "No results found"

@pytest.mark.asyncio
async def test_enumerate_single_result():
    code = '''
async def search_stories(query: str):
    return [{"title": "Lone Story", "score": 99}]

async def main():
    results = await search_stories("python")
    report = ""
    for idx, story in enumerate(results, 1):
        report += f"""
        {idx}. {story['title']} (Score: {story['score']})"""
    return report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    expected = """
        1. Lone Story (Score: 99)"""
    assert result.result.strip() == expected.strip()

@pytest.mark.asyncio
async def test_enumerate_missing_fields():
    code = '''
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
        report += f"""
        {idx}. {story.get('title', 'Untitled')} (Score: {story.get('score', 'N/A')})"""
    return report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    expected = """
        1. Complete Story (Score: 50)
        2. Missing Score (Score: N/A)
        3. Untitled (Score: 30)
        4. Untitled (Score: N/A)"""
    assert result.result.strip() == expected.strip()

@pytest.mark.asyncio
async def test_enumerate_large_results():
    code = '''
async def search_stories(query: str):
    return [{"title": f"Story {i}", "score": i} for i in range(100)]

async def main():
    results = await search_stories("python")
    report = f"Found {len(results)} stories"
    for idx, story in enumerate(results[:5], 1):  # Only show first 5
        report += f"""
        {idx}. {story['title']} (Score: {story['score']})"""
    if len(results) > 5:
        report += """
        ... (truncated)"""
    return report
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert "Found 100 stories" in result.result
    assert "1. Story 0 (Score: 0)" in result.result
    assert "... (truncated)" in result.result