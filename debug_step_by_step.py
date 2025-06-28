#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    # Test step 1: simple async_enumerate
    test_code1 = '''
async def async_enumerate(async_iterable, start=0):
    i = start
    async for item in async_iterable:
        yield i, item
        i += 1

async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

# Test the async_enumerate function directly  
async def test():
    result = []
    async for pair in async_enumerate(gen()):
        result.append(pair)
    return result

test()
'''
    
    print("Testing async_enumerate function...")
    result = await execute_async(test_code1, entry_point="test")
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    
    # Test step 2: async_enumerate with comprehension
    test_code2 = '''
async def async_enumerate(async_iterable, start=0):
    i = start
    async for item in async_iterable:
        yield i, item
        i += 1

async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

# Use manual iteration to collect results
async def test2():
    values = []
    async for i, x in async_enumerate(gen()):
        values.append((i, x))
    return values

test2()
'''
    
    print("\nTesting manual iteration...")
    result = await execute_async(test_code2, entry_point="test2")
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
