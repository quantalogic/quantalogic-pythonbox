#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def debug_get_score_return():
    print("Testing what get_score() returns...")
    
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

item = HNItem("Item A", 300)

# Get what get_score returns
coro = item.get_score()
type(coro).__name__
''', allowed_modules=['asyncio'])
    
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_get_score_return())
