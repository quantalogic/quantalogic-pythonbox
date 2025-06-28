#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def debug_sorted_behavior():
    print("Testing sorted behavior with coroutines...")
    
    # Test the sorted call directly
    result = await execute_async('''
import asyncio

class HNItem:
    def __init__(self, title, score):
        self.title = title
        self.score = score

    async def get_score(self):
        await asyncio.sleep(0.01)
        return self.score

items = [HNItem("Item A", 300), HNItem("Item B", 200), HNItem("Item C", 400)]

# This should fail in normal Python
sorted(items, key=lambda x: x.get_score(), reverse=True)
''', allowed_modules=['asyncio'])
    
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
    print(f"Result type: {type(result.result)}")

if __name__ == "__main__":
    asyncio.run(debug_sorted_behavior())
