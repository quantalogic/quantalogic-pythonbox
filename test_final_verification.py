#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to the path
project_root = "/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox"
sys.path.insert(0, project_root)

from quantalogic_pythonbox.execution import execute_async

async def test_both_cases():
    print("=== Testing Error Case ===")
    error_code = """
import asyncio

class Item:
    def __init__(self, value):
        self.value = value

    async def get_value(self):
        await asyncio.sleep(0)
        return self.value

async def main():
    items = [Item(3), Item(1), Item(2)]
    sorted_items = sorted(items, key=lambda x: x.get_value())
    return [item.value for item in sorted_items]
    """
    
    error_result = await execute_async(error_code, entry_point="main")
    print(f"Error result: {error_result.result}")
    print(f"Error message: {error_result.error}")
    print("✅ Contains expected error:", "'<' not supported between instances of 'coroutine' and 'coroutine'" in error_result.error)
    
    print("\n=== Testing Solution Case ===")
    fixed_code = """
import asyncio

class Item:
    def __init__(self, value):
        self.value = value

    async def get_value(self):
        await asyncio.sleep(0)
        return self.value

async def main():
    items = [Item(3), Item(1), Item(2)]
    values = [await item.get_value() for item in items]
    sorted_items = sorted(items, key=lambda x: values[items.index(x)])
    return [item.value for item in sorted_items]
    """
    
    result = await execute_async(fixed_code, entry_point="main")
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    print(f"✅ Solution works: {result.result == [1, 2, 3] and result.error is None}")

if __name__ == "__main__":
    asyncio.run(test_both_cases())
