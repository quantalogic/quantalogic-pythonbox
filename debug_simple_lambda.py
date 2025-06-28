#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to the path
project_root = "/Users/raphaelmansuy/Github/03-working/quantalogic-pythonbox"
sys.path.insert(0, project_root)

from quantalogic_pythonbox.execution import execute_async

async def debug_simple_lambda():
    simple_code = """
async def main():
    values = [3, 1, 2]
    items = ["a", "b", "c"]
    
    # Simple lambda that should work
    simple_lambda = lambda x: values[items.index(x)]
    
    # Test the lambda directly
    result = simple_lambda("b")
    print(f"Simple lambda result: {result}")
    
    # Try sorted with this lambda
    sorted_items = sorted(items, key=lambda x: values[items.index(x)])
    return sorted_items
    """
    
    result = await execute_async(simple_code, entry_point="main")
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_simple_lambda())
