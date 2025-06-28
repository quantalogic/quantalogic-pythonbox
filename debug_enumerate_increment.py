#!/usr/bin/env python3

import asyncio
import sys

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    test_code = '''
async def async_enumerate(async_iterable, start=0):
    i = start
    print(f"Starting with i={i}")
    async for item in async_iterable:
        print(f"About to yield: ({i}, {item})")
        yield i, item
        print(f"After yield, incrementing i from {i} to {i+1}")
        i += 1
        print(f"i is now {i}")

async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

# Test the async enumerate directly
result = []
async for item in async_enumerate(gen()):
    print(f"Collected: {item}")
    result.append(item)

print(f"Final result: {result}")
result
'''
    
    print("Testing async enumerate directly...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
