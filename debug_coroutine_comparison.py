#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def debug_coroutine_comparison():
    print("Testing coroutine comparison...")
    
    # Test the comparison directly
    result = await execute_async('''
import asyncio

class HNItem:
    def __init__(self, title, score):
        self.title = title
        self.score = score

    async def get_score(self):
        await asyncio.sleep(0.01)
        return self.score

items = [HNItem("Item A", 300), HNItem("Item B", 200)]

# Get coroutines
coro1 = items[0].get_score()
coro2 = items[1].get_score()

# Try to compare them directly - this should fail
try:
    result = coro1 < coro2
    "comparison_succeeded"
except TypeError as e:
    str(e)
''', allowed_modules=['asyncio'])
    
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_coroutine_comparison())
