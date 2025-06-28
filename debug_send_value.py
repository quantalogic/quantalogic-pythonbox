#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_send_value():
    """Debug the asend issue"""
    source = """
async def async_gen():
    x = yield 1  # This should receive the sent value
    yield x

async def compute():
    gen = async_gen()
    await gen.__anext__()  # First yield (returns 1)
    result = await gen.asend(42)  # Send value (should get 42)
    return result
"""
    
    print("=== Testing async generator send value ===")
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    print(f"Execution time: {result.execution_time}")

if __name__ == "__main__":
    asyncio.run(test_send_value())
