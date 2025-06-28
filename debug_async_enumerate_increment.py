#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from quantalogic_pythonbox.execution import interpret_code

code = """
async def simple_gen():
    yield 'a'
    yield 'b'
    yield 'c'

async def debug_async_enumerate(async_iterable, start=0):
    print(f"Starting with i = {start}")
    i = start
    async for item in async_iterable:
        print(f"Before yield: i = {i}, item = {item}")
        yield i, item
        print(f"After yield, before increment: i = {i}")
        i += 1
        print(f"After increment: i = {i}")

async def test_function():
    result = []
    async for i, x in debug_async_enumerate(simple_gen()):
        print(f"Received: ({i}, {x})")
        result.append((i, x))
    return result

import asyncio
result = await test_function()
"""

def main():
    print("Testing async_enumerate with debug output...")
    result = interpret_code(code, allowed_modules=['asyncio'])
    print(f"Final result: {result}")

if __name__ == "__main__":
    main()
