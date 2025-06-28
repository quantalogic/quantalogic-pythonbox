#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "quantalogic_pythonbox"))

from quantalogic_pythonbox.execution import execute_async


async def test_simple_asend():
    """Test a simpler asend case"""
    print("=== Testing simple async generator asend ===")
    
    # First test: just yielding without assignment
    source1 = """
async def async_gen():
    print("Generator started")
    yield 1
    yield 2

async def compute():
    gen = async_gen()
    first = await gen.__anext__()
    second = await gen.__anext__()
    return second
"""
    
    result = await execute_async(source1, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Simple yield result: {result.result}")
    
    # Second test: yield with assignment
    source2 = """
async def async_gen():
    print("Generator started - assignment test")
    x = yield 1
    print(f"x after yield: {x}")
    return x

async def compute():
    gen = async_gen()
    first = await gen.__anext__()
    print(f"First value: {first}")
    try:
        result = await gen.asend(42)
        print(f"asend result: {result}")
        return result
    except StopAsyncIteration as e:
        print(f"StopAsyncIteration with value: {e.value}")
        return e.value
"""
    
    result = await execute_async(source2, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Assignment yield result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_simple_asend())
