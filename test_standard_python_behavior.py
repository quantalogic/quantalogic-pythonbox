#!/usr/bin/env python3
"""Test to verify the issue with standard Python."""

import asyncio

async def simple_gen():
    print("Gen yielding 'a'")
    yield 'a'
    print("Gen yielding 'b'")  
    yield 'b'
    print("Gen yielding 'c'")
    yield 'c'

async def async_enumerate(async_iterable):
    i = 0
    async for item in async_iterable:
        print(f"yielding ({i}, {item})")
        yield (i, item)
        i += 1

async def main():
    print("=== Standard Python behavior ===")
    result = [x async for x in async_enumerate(simple_gen())]
    print("Result:", result)
    print("Expected: [(0, 'a'), (1, 'b'), (2, 'c')]")

if __name__ == "__main__":
    asyncio.run(main())
