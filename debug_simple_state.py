#!/usr/bin/env python3

import asyncio
import sys

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    test_code = '''
async def simple_async_gen():
    print("Before first yield")
    yield 1
    print("After first yield")
    yield 2
    print("After second yield")
    yield 3
    print("Done")

result = []
async for value in simple_async_gen():
    print(f"Got value: {value}")
    result.append(value)

print(f"Final result: {result}")
result
'''
    
    print("Testing simple async generator state...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
