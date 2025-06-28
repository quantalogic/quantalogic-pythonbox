#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox import execute_async

async def test_nested_generators():
    code = '''
async def gen():
    print("gen: Before yield a")
    yield 'a'
    print("gen: Before yield b") 
    yield 'b'
    print("gen: Before yield c")
    yield 'c'
    print("gen: After yield c")

async def async_enumerate(async_iterable, start=0):
    print(f"async_enumerate: Starting with start={start}")
    i = start
    print(f"async_enumerate: About to start async for loop")
    async for item in async_iterable:
        print(f"async_enumerate: Got item {item}, will yield ({i}, {item})")
        yield i, item
        print(f"async_enumerate: After yield ({i}, {item}), incrementing i")
        i += 1
    print("async_enumerate: Async for loop completed")

async def test():
    results = []
    print("test: Starting async for over async_enumerate")
    async for idx_item in async_enumerate(gen()):
        print(f"test: Got idx_item: {idx_item}")
        results.append(idx_item)
    print("test: Completed async for")
    return results

await test()
'''
    
    print("Executing nested async generator test:")
    result = await execute_async(code, allowed_modules=[])
    print(f"Final result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_nested_generators())
