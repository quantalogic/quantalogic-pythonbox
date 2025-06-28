#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_large_results():
    code = '''
async def search_stories(query: str):
    return [{"title": f"Story {i}", "score": i} for i in range(100)]

async def main():
    results = await search_stories("python")
    report = f"Found {len(results)} stories"
    for idx, story in enumerate(results[:5], 1):
        report += f"\\n{idx}. {story['title']} (Score: {story['score']})"
    if len(results) > 5:
        report += "\\n... (truncated)"
    return report
'''
    result = await execute_async(code, entry_point='main')
    print(f"Result: {result}")
    print(f"Result.result: {repr(result.result)}")
    print(f"Result.error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_large_results())
