#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to the path
project_root = "/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox"
sys.path.insert(0, project_root)

from quantalogic_pythonbox.execution import execute_async

async def debug_solution():
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
    print(f"values: {values}")
    print(f"items: {items}")
    for i, item in enumerate(items):
        index = items.index(item)
        print(f"item {i}: {item}, index: {index}, values[index]: {values[index]}")
    
    # Test the lambda function
    test_lambda = lambda x: values[items.index(x)]
    for item in items:
        result = test_lambda(item)
        print(f"Lambda result for {item}: {result} (type: {type(result)})")
    
    sorted_items = sorted(items, key=lambda x: values[items.index(x)])
    return [item.value for item in sorted_items]
    """
    
    result = await execute_async(fixed_code, entry_point="main")
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_solution())
