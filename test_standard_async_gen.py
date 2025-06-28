#!/usr/bin/env python3

import asyncio

async def test_standard_python():
    """Test how standard Python handles async generator send"""
    
    async def async_gen():
        x = yield 1  # This should receive the sent value
        yield x

    gen = async_gen()
    print("First yield:", await gen.__anext__())  # Should return 1
    try:
        result = await gen.asend(42)  # Should return 42
        print("Second yield:", result)
    except StopAsyncIteration:
        print("Generator completed")

async def test_athrow():
    """Test how standard Python handles async generator athrow"""
    
    async def async_gen():
        try:
            yield 1
        except ValueError:
            yield "caught"

    gen = async_gen()
    print("First yield:", await gen.__anext__())  # Should return 1
    try:
        result = await gen.athrow(ValueError)  # Should return "caught"
        print("Caught exception, yielded:", result)
    except StopAsyncIteration:
        print("Generator completed")

if __name__ == "__main__":
    asyncio.run(test_standard_python())
    print("---")
    asyncio.run(test_athrow())
