#!/usr/bin/env python3
"""Debug the async enumerate minimal case."""

import asyncio
from quantalogic_pythonbox import execute_async

async def main():
    print("=== Testing minimal async enumerate ===")
    
    result = await execute_async('''
async def simple_gen():
    yield 'a'
    yield 'b'
    yield 'c'

async def async_enumerate(async_iterable):
    i = 0
    async for item in async_iterable:
        print(f"yielding ({i}, {item})")
        yield (i, item)
        i += 1

[x async for x in async_enumerate(simple_gen())]
''')
    
    print("Result:", result.result)
    print("Expected: [(0, 'a'), (1, 'b'), (2, 'c')]")

if __name__ == "__main__":
    asyncio.run(main())
