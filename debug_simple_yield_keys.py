#!/usr/bin/env python3
"""Debug the yield key generation issue."""

import asyncio
from quantalogic_pythonbox import execute_async

async def main():
    print("=== Testing yield key generation ===")
    
    # Simpler test case
    result = await execute_async('''
async def simple_gen():
    print("About to yield 1")
    yield 1
    print("About to yield 2") 
    yield 2
    print("About to yield 3")
    yield 3

[x async for x in simple_gen()]
''')
    
    print("Simple result:", result.result)
    print("Expected: [1, 2, 3]")

if __name__ == "__main__":
    asyncio.run(main())
