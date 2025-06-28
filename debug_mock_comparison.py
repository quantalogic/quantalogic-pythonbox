#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_mock_coroutine_comparison():
    print("Testing MockCoroutine comparison error...")
    
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

# Test lambda function returns
key_func = lambda x: x.get_score()
result1 = key_func(items[0])
result2 = key_func(items[1])

print("Result1 str representation:", str(result1))
print("Result2 str representation:", str(result2))

# Try direct comparison
try:
    comparison_result = result1 < result2
    print("Direct comparison succeeded:", comparison_result)
except Exception as e:
    print("Direct comparison failed with:", str(e))
    
# Test what happens in sorted
try:
    sorted_result = sorted(items, key=lambda x: x.get_score(), reverse=True)
    print("Sorted succeeded - this should not happen!")
except Exception as e:
    print("Sorted failed with:", str(e))

"Test completed"
''', allowed_modules=['asyncio'])
    
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_mock_coroutine_comparison())
