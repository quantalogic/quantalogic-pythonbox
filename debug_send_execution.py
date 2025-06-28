#!/usr/bin/env python3

"""
Debug script to understand async generator send execution flow
"""

import asyncio
import logging
from quantalogic_pythonbox.execution import execute_async

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)')
logger = logging.getLogger(__name__)

async def test_send_execution():
    """Test async generator send execution step by step"""
    
    source = """
async def async_gen():
    print("Generator started")
    x = yield 1
    print(f"Received value: {x}")
    yield x
    print("Generator finished")

async def test():
    gen = async_gen()
    print("Created generator")
    
    # First call to get first yield
    first_value = await gen.__anext__()
    print(f"First yield: {first_value}")
    
    # Send value and get second yield
    second_value = await gen.asend(42)
    print(f"Second yield: {second_value}")
    
    return second_value
"""
    
    try:
        result = await execute_async(source, entry_point="test")
        print(f"Final result: {result.result}")
        print(f"Error: {result.error}")
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_send_execution())
