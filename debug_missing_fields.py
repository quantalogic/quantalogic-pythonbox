#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_missing_fields():
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
        report += f"\\n{idx}. {story.get('title', 'Untitled')} (Score: {story.get('score', 'N/A')})"
    return report
'''
    result = await execute_async(code, entry_point='main')
    print(f"Result: {result}")
    print(f"Result.result: {repr(result.result)}")
    print(f"Result.error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_missing_fields())
