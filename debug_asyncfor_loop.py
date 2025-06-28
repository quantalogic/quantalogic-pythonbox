#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from quantalogic_pythonbox.execution import interpret_code

code = """
async def simple_async_gen():
    yield 1
    yield 2
    yield 3

async def test_function():
    count = 0
    async for value in simple_async_gen():
        print(f"Got value: {value}")
        count += 1
        if count >= 3:  # Safety break
            break
    return count

import asyncio
result = await test_function()
"""

def main():
    print("Testing simple async for loop...")
    result = interpret_code(code, allowed_modules=['asyncio'])
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
