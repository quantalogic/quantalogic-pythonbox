#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def debug_bound_method_type():
    print("Testing bound method type...")
    
    # Test the bound method type
    result = await execute_async('''
import asyncio

class HNItem:
    def __init__(self, title, score):
        self.title = title
        self.score = score

    async def get_score(self):
        await asyncio.sleep(0.01)
        return self.score

item = HNItem("Item A", 300)

# Get the bound method
method = item.get_score

# Check if it has async_func attribute
result = []
result.append(hasattr(method, 'async_func'))
if hasattr(method, 'async_func'):
    result.append('AsyncFunction' in str(type(method.async_func)))
result
''', allowed_modules=['asyncio'])
    
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_bound_method_type())
