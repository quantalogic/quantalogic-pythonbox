#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "quantalogic_pythonbox"))

from quantalogic_pythonbox.execution import execute_async


async def test_asend_issue():
    """Test the asend issue"""
    print("=== Testing async generator asend issue ===")
    
    source = """
async def async_gen():
    print("Generator started")
    x = yield 1
    print(f"Received value: {x}")
    yield x

async def compute():
    print("Starting computation")
    gen = async_gen()
    first_value = await gen.__anext__()  # First yield
    print(f"First yielded value: {first_value}")
    result = await gen.asend(42)  # Send value
    print(f"Second yielded value: {result}")
    return result
"""
    
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Final result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_asend_issue())
