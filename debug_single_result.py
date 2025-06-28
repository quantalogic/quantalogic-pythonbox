#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
    print("Testing single result case...")
    
    result = await execute_async('''
async def search_stories(query: str):
    return [{"title": "Lone Story", "score": 99}]

async def main():
    results = await search_stories("python")
    report = ""
    for idx, story in enumerate(results, 1):
        report += f"\\n{idx}. {story['title']} (Score: {story['score']})"
    return report
''', entry_point='main')
    
    print(f"Result object: {result}")
    print(f"Result.result: {result.result}")
    print(f"Result.error: {result.error}")
    print(f"Result.local_variables: {result.local_variables}")

if __name__ == "__main__":
    asyncio.run(main())
