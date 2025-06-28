#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    test_code = '''
async def async_enumerate(async_iterable, start=0):
    i = start
    async for item in async_iterable:
        yield i, item
        i += 1

async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

[(i, x) async for i, x in async_enumerate(gen())]
'''
    
    print("Testing async enumerate...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    print(f"Local vars: {result.local_variables}")
    print(f"Type of result: {type(result.result)}")

if __name__ == "__main__":
    asyncio.run(main())
