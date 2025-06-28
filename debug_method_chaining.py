#!/usr/bin/env python3
"""
Debug script to understand method chaining issue
"""

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_method_chaining():
    """Test method chaining step by step"""
    source = """
class A:
    def add(self, x):
        return x + 1
    def compute(self):
        return self.add(5)

def compute():
    a = A()
    print(f"a = {a}")
    add_method = a.add
    print(f"add_method = {add_method}")
    add_result = add_method(5)
    print(f"add_result = {add_result}")
    return add_result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_method_chaining())
