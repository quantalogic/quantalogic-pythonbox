#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_lambda_coroutine():
    print("Testing lambda coroutine behavior...")
    
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

# Test the lambda key function directly
item = items[0]
key_func = lambda x: x.get_score()
key_result = key_func(item)

print("Key result repr:", repr(key_result))
print("Key result str:", str(key_result))

# Test comparison between two key results
item2 = items[1]
key_result2 = key_func(item2)

print("Key result2 repr:", repr(key_result2))

# Try comparing the results
try:
    comparison = key_result < key_result2
    print("Comparison succeeded:", comparison)
except Exception as e:
    print("Comparison failed with:", type(e).__name__, str(e))

"Test completed"
''', allowed_modules=['asyncio'])
    
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_lambda_coroutine())
