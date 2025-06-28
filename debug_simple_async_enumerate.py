#!/usr/bin/env python3

import asyncio
import sys
sys.path.insert(0, '.')

from quantalogic_pythonbox.execution import execute_async

async def test_simple_async_enumerate():
    """Test a very simplified version of async_enumerate to isolate the issue"""
    
    # Test 1: Simple async generator that should yield 'a', 'b', 'c'
    print("=== Test 1: Simple async generator ===")
    result = await execute_async('''
async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

result = []
async for x in gen():
    result.append(x)
result
''')
    print(f"Result: {result.result}")
    print(f"Expected: ['a', 'b', 'c']")
    print(f"Match: {result.result == ['a', 'b', 'c']}")
    
    # Test 2: Simple async enumerate (no counter, just pass through)
    print("\n=== Test 2: Simple async enumerate (pass through) ===")
    result = await execute_async('''
async def simple_async_enumerate(async_iterable):
    async for item in async_iterable:
        yield item

async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

result = []
async for x in simple_async_enumerate(gen()):
    result.append(x)
result
''')
    print(f"Result: {result.result}")
    print(f"Expected: ['a', 'b', 'c']")
    print(f"Match: {result.result == ['a', 'b', 'c']}")
    
    # Test 3: Simple async enumerate with counter
    print("\n=== Test 3: Simple async enumerate with counter ===")
    result = await execute_async('''
async def simple_async_enumerate(async_iterable):
    i = 0
    async for item in async_iterable:
        yield (i, item)
        i += 1

async def gen():
    yield 'a'
    yield 'b'
    yield 'c'

result = []
async for pair in simple_async_enumerate(gen()):
    result.append(pair)
result
''')
    print(f"Result: {result.result}")
    print(f"Expected: [(0, 'a'), (1, 'b'), (2, 'c')]")
    print(f"Match: {result.result == [(0, 'a'), (1, 'b'), (2, 'c')]}")

if __name__ == "__main__":
    asyncio.run(test_simple_async_enumerate())
