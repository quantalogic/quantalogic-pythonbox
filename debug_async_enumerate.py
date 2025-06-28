#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

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
    async for item in async_iterable:
        print(f"async_enumerate yielding: ({i}, {item})")
        yield i, item
        i += 1

async def gen():
    print("gen yielding: a")
    yield 'a'
    print("gen yielding: b") 
    yield 'b'
    print("gen yielding: c")
    yield 'c'

print("Creating generator...")
generator = gen()

print("Creating async_enumerate...")
enum_gen = async_enumerate(generator)

print("Starting list comprehension...")
result = [item async for item in enum_gen]
print(f"Final result: {result}")
result
'''
    
    print("Testing async enumerate...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
    print(f"Error: {result.error}")
    print(f"Local vars: {result.local_variables}")
    print(f"Type of result: {type(result.result)}")

if __name__ == "__main__":
    asyncio.run(main())
