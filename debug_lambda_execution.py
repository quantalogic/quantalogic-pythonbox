#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def debug_lambda_execution():
    print("Testing lambda execution...")
    
    # Test what the lambda is getting
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

# What does the lambda return?
key_func = lambda x: x.get_score()
first_result = key_func(items[0])
first_result
''', allowed_modules=['asyncio'])
    
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(debug_lambda_execution())
