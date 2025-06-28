#!/usr/bin/env python3
"""Debug the async enumerate issue to understand the yield coordination problem."""

import asyncio
from quantalogic_pythonbox import execute_async

async def main():
    print("=== Testing async enumerate issue ===")
    
    # Test case that's failing
    result = await execute_async('''
async def async_enumerate(async_iterable, start=0):
    i = start
    async for item in async_iterable:
        print(f"Processing item: {item}, index: {i}")
        yield i, item
        print(f"After yield for {item}, index now: {i}")
        i += 1
        print(f"Incremented i to: {i}")

async def gen():
    print("Gen yielding 'a'")
    yield 'a'
    print("Gen yielding 'b'")
    yield 'b'
    print("Gen yielding 'c'")
    yield 'c'
    print("Gen completed")

print("Starting list comprehension")
result = [(i, x) async for i, x in async_enumerate(gen())]
print(f"Final result: {result}")
result
''')
    
    print("Result:", result.result)
    print("Expected: [(0, 'a'), (1, 'b'), (2, 'c')]")

if __name__ == "__main__":
    asyncio.run(main())
