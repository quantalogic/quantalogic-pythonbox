#!/usr/bin/env python3
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "quantalogic_pythonbox"))

from quantalogic_pythonbox.execution import execute_async


async def test_exception_handling():
    """Test basic exception handling in try-except"""
    print("=== Testing basic exception handling ===")
    
    source = """
async def test():
    try:
        raise ValueError("test error")
    except ValueError as e:
        return f"caught: {e}"
    return "not caught"

result = await test()
result
"""
    
    result = await execute_async(source, entry_point=None, allowed_modules=["asyncio"])
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")


async def test_yield_in_try():
    """Test yield inside try-except"""
    print("=== Testing yield inside try-except ===")
    
    source = """
async def async_gen():
    try:
        print("Before yield")
        yield 1
        print("After yield")
    except ValueError as e:
        print(f"Caught exception: {e}")
        yield "caught"
    print("Generator finished")

async def test():
    gen = async_gen()
    first = await gen.__anext__()
    print(f"First value: {first}")
    # Just continue to see what happens after yield
    try:
        second = await gen.__anext__()
        print(f"Second value: {second}")
        return second
    except StopAsyncIteration:
        print("Generator exhausted")
        return "exhausted"

result = await test()
result
"""
    
    result = await execute_async(source, entry_point=None, allowed_modules=["asyncio"])
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(test_exception_handling())
    print()
    asyncio.run(test_yield_in_try())
