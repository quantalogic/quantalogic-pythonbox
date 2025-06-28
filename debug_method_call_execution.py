#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def debug_method_call_execution():
    print("Testing method call execution...")
    
    # Test the method call directly
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

# Call the bound method and see what it returns
coro = item.get_score()
coro
''', allowed_modules=['asyncio'])
    
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_method_call_execution())
