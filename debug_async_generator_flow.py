#!/usr/bin/env python3
"""
Debug script to understand the exact execution flow needed
"""

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_simple_case():
    """Test a simpler async generator case first"""
    source = """
async def async_gen():
    yield 1
    yield 2

async def compute():
    gen = async_gen()
    first = await gen.__anext__()
    second = await gen.__anext__()
    return f"{first},{second}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Simple case result: {result.result}")
    print(f"Simple case error: {result.error}")

async def test_assignment_case():
    """Test the problematic assignment case"""
    source = """
async def async_gen():
    print("Before yield 1")
    x = yield 1  # This should pause and wait for a value
    print(f"Received x = {x}")
    yield x

async def compute():
    gen = async_gen()
    print("Calling __anext__")
    first = await gen.__anext__()  # Should get 1
    print(f"Got first: {first}")
    print("Calling asend(42)")
    second = await gen.asend(42)   # Should send 42 to x and get 42 back
    print(f"Got second: {second}")
    return second
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=["asyncio"])
    print(f"Assignment case result: {result.result}")
    print(f"Assignment case error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_simple_case())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_assignment_case())
