#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_sorting_issue():
    print("Testing async sort issue...")
    
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

# Test what get_score() returns
item = items[0]
score_call = item.get_score()
print("Value of item.get_score():", score_call)

# Test the lambda key function  
key_func = lambda x: x.get_score()
key_result = key_func(item)
print("Value of key_func(item):", key_result)

# Try sorted - this should raise an error
try:
    result = sorted(items, key=lambda x: x.get_score(), reverse=True)
    print("Sorted succeeded")
except Exception as e:
    print("Sorted failed with:", str(e))
    raise e

"Test completed"
''', allowed_modules=['asyncio'])
    
    print("Result:", result.result)
    print("Error:", result.error)
    if result.error:
        print("Error type:", type(result.error))

if __name__ == "__main__":
    asyncio.run(test_sorting_issue())
