#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_throw_value():
    """Debug the athrow issue"""
    source = """
async def async_gen():
    try:
        yield 1
    except ValueError as e:
        yield "caught"

async def compute():
    gen = async_gen()
    await gen.__anext__()  # First yield (returns 1)
    result = await gen.athrow(ValueError("test error"))  # Throw exception (should get "caught")
    return result
"""
    
    print("=== Testing async generator throw exception ===")
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    print(f"Execution time: {result.execution_time}")

if __name__ == "__main__":
    asyncio.run(test_throw_value())
