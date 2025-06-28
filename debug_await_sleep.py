#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
    code = '''
import asyncio

async def test_sleep():
    await asyncio.sleep(0.01)
    return "done"

result = await test_sleep()
result
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
