#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
    code = '''
import asyncio

# Test 1: Check if asyncio.sleep returns a coroutine when called directly
sleep_coro = asyncio.sleep(0.01)
print(f"sleep_coro: {sleep_coro}")

# Test 2: Check asyncio.sleep in await context
async def test_func():
    coro = asyncio.sleep(0.01)
    print(f"Inside function - coro: {coro}")
    result = await coro
    print(f"Await result: {result}")
    return "done"

result = await test_func()
print(f"Final result: {result}")
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
