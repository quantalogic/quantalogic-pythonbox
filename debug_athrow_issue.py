#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "quantalogic_pythonbox"))

from quantalogic_pythonbox.execution import execute_async


async def test_athrow_issue():
    """Test the athrow issue"""
    print("=== Testing async generator athrow issue ===")
    
    source = """
async def async_gen():
    try:
        yield 1
    except ValueError:
        yield "caught"

async def compute():
    gen = async_gen()
    first_value = await gen.__anext__()
    print(f"First value: {first_value}")
    result = await gen.athrow(ValueError)
    print(f"athrow result: {result}")
    return result
"""
    
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Final result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_athrow_issue())
