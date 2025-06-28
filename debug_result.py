#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_basic():
    # Test with the exact data from the failing test
    input_data = [
        {"title": "Story 1", "score": 42},
        {"title": "Story 2", "score": 31},
        {"title": "Story 3", "score": 27}
    ]
    
    code = f'''
async def search_stories(query: str):
    return {input_data}

async def main():
    results = await search_stories("python")
    report = ""
    for index, story in enumerate(results, 1):
        report += "\\n"
        report += str(index)
        report += ". "
        report += story['title']
        report += " "
        report += f"(Score: "
        report += str(story['score'])
        report += ")"
    return report
'''
    
    print("Generated code:")
    print(repr(code))
    print("Parsed code:")
    print(code)
    
    result = await execute_async(code, entry_point='main')
    print(f"Result: {result}")
    print(f"Result.result: {result.result}")
    print(f"Result.error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_basic())
