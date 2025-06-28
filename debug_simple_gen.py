#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox import execute_async

async def test_simple_generator():
    code = '''
async def gen():
    print("Before yield a")
    yield 'a'
    print("Before yield b") 
    yield 'b'
    print("Before yield c")
    yield 'c'
    print("After yield c")

async def test():
    results = []
    async for item in gen():
        print(f"Got item: {item}")
        results.append(item)
    return results

await test()
'''
    
    print("Executing async generator test:")
    result = await execute_async(code, allowed_modules=[])
    print(f"Final result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_simple_generator())
