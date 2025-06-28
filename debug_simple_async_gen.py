#!/usr/bin/env python3

import asyncio
import sys

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    test_code = '''
async def simple_gen():
    print("Yielding 1")
    yield 1
    print("Yielding 2")
    yield 2
    print("Yielding 3")
    yield 3

print("Starting manual iteration:")
gen = simple_gen()
values = []
try:
    while True:
        val = await gen.__anext__()
        print(f"Got value: {val}")
        values.append(val)
except StopAsyncIteration:
    print("StopAsyncIteration caught")

print(f"Manual result: {values}")

print("Starting list comprehension:")
gen2 = simple_gen()
result2 = [x async for x in gen2]
print(f"List comp result: {result2}")

result2
'''
    
    print("Testing simple async generator...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
