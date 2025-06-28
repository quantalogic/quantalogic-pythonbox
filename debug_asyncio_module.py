#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
    code = '''
import asyncio

# Test what asyncio.sleep actually is
sleep_func = asyncio.sleep
print(f"asyncio.sleep: {sleep_func}")
print(f"Type: {type(sleep_func)}")

# Test calling it without await
sleep_result = asyncio.sleep(0.01)
print(f"sleep_result: {sleep_result}")
print(f"Type: {type(sleep_result)}")
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
