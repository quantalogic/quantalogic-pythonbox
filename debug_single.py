#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_single_result():
    code = '''
async def search_stories(query: str):
    return [{"title": "Lone Story", "score": 99}]

async def main():
    results = await search_stories("python")
    report = ""
    for idx, story in enumerate(results, 1):
        report += f"\\n{idx}. {story['title']} (Score: {story['score']})"
    return report
'''
    result = await execute_async(code, entry_point='main')
    print(f"Result: {result}")
    print(f"Result.result: {result.result}")
    print(f"Result.error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_single_result())
