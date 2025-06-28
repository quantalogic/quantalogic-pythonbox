#!/usr/bin/env python3
"""
Debug script to check function types
"""

import asyncio
import inspect
from quantalogic_pythonbox.execution import execute_async

async def test_function_types():
    """Test what types functions have"""
    source = """
class A:
    def add(self, x):
        return x + 1

def compute():
    a = A()
    add_method = a.add
    
    # Check the function types
    print(f"add_method type: {type(add_method)}")
    print(f"iscoroutinefunction: {__import__('asyncio').iscoroutinefunction(add_method)}")
    print(f"inspect.iscoroutinefunction: {__import__('inspect').iscoroutinefunction(add_method)}")
    
    return "done"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_function_types())
