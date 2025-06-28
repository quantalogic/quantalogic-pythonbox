#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to sys.path to import the module
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox')

from quantalogic_pythonbox.execution import execute_async

async def main():
    # Minimal reproduction
    test_code = '''
async def simple_gen():
    yield 'a'
    yield 'b'

[x async for x in simple_gen()]
'''
    
    print("Testing simple async comprehension...")
    result = await execute_async(test_code)
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    print(f"Type of result: {type(result.result)}")

if __name__ == "__main__":
    asyncio.run(main())
