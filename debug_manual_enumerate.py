#!/usr/bin/env python3

import asyncio
import sys

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    test_code = '''
async def async_enumerate_manual(async_iterable, start=0):
    """Manual implementation that doesn't rely on async for"""
    i = start
    print(f"Starting with i={i}")
    
    # Get the async iterator manually
    async_iter = async_iterable.__aiter__()
    
    while True:
        try:
            print(f"Getting next item with i={i}")
            item = await async_iter.__anext__()
            print(f"Got item: {item}, yielding ({i}, {item})")
            yield i, item
            print(f"After yield, incrementing i from {i} to {i+1}")
            i += 1
            print(f"i is now {i}")
        except StopAsyncIteration:
            print("StopAsyncIteration caught, ending generator")
            break

async def gen():
    yield 'a'
    yield 'b' 
    yield 'c'

# Test the manual async enumerate
result = []
async for item in async_enumerate_manual(gen()):
    print(f"Collected: {item}")
    result.append(item)

print(f"Final result: {result}")
result
'''
    
    print("Testing manual async enumerate...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
