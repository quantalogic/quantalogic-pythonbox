#!/usr/bin/env python3
"""Debug script to test async generator asend/athrow behavior in standard Python"""

import asyncio

async def test_standard_python():
    """Test how asend works in standard Python"""
    print("=== Standard Python async generator behavior ===")
    
    async def async_gen():
        print("Generator: Starting")
        x = yield 1
        print(f"Generator: Received x = {x}")
        result = yield x
        print(f"Generator: yield x returned {result}")
    
    gen = async_gen()
    
    # First call to __anext__ starts the generator and returns first yield
    first_value = await gen.__anext__()
    print(f"First yield: {first_value}")
    
    # Send 42 to the generator, it should assign to x and yield x (42)
    second_value = await gen.asend(42)
    print(f"Second yield after asend(42): {second_value}")
    
    try:
        # Try to send None and see what happens
        third_value = await gen.asend(None)
        print(f"Third yield: {third_value}")
    except StopAsyncIteration:
        print("Generator finished")

async def test_athrow():
    """Test how athrow works in standard Python"""
    print("\n=== Standard Python async generator athrow behavior ===")
    
    async def async_gen():
        try:
            yield 1
        except ValueError as e:
            print(f"Generator: Caught exception: {e}")
            yield "caught"
        except BaseException as e:
            print(f"Generator: Caught other exception: {e}")
            yield f"other: {e}"
    
    gen = async_gen()
    
    # First call to __anext__ starts the generator and returns first yield
    first_value = await gen.__anext__()
    print(f"First yield: {first_value}")
    
    # Throw ValueError into the generator
    try:
        second_value = await gen.athrow(ValueError("test error"))
        print(f"Second yield after athrow: {second_value}")
    except StopAsyncIteration:
        print("Generator finished")

if __name__ == "__main__":
    asyncio.run(test_standard_python())
    asyncio.run(test_athrow())
