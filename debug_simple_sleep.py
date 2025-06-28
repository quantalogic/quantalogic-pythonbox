#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
    code = '''
import asyncio

# Test calling asyncio.sleep without await
sleep_result = asyncio.sleep(0.01)
sleep_result
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print(f"Result: {result.result}")
    print(f"Result type: {type(result.result)}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
