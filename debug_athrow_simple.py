import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
    code = '''
async def async_gen():
    try:
        x = yield 1
        yield f"sent: {x}"
    except ValueError:
        yield "caught"

async def test():
    gen = async_gen()
    await gen.asend(None)  # Start and get first value
    try:
        result = await gen.athrow(ValueError)
        return result
    except StopAsyncIteration:
        pass

result = await test()
'''
    
    print("=== Running athrow test ===")
    result = await execute_async(code)
    print(f"Result: {result.result}")
    if result.error:
        print(f"Error: {result.error}")
        
    print("\n=== Testing Python native behavior ===")
    # Let's see what Python itself does
    async def async_gen():
        try:
            x = yield 1
            yield f"sent: {x}"
        except ValueError:
            yield "caught"

    async def test():
        gen = async_gen()
        await gen.asend(None)  # Start and get first value
        try:
            result = await gen.athrow(ValueError)
            return result
        except StopAsyncIteration:
            pass

    result = await test()
    print(f"Python native result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
