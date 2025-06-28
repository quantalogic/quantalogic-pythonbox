"""
Debug script to understand the sorting issue with zip
"""
import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_sorting_with_zip():
    code = '''
import asyncio

class HNItem:
    def __init__(self, title, score):
        self.title = title
        self.score = score

    async def get_score(self):
        await asyncio.sleep(0.01)
        return self.score

items = [HNItem("Item A", 300), HNItem("Item B", 200), HNItem("Item C", 400)]
scores = [await item.get_score() for item in items]
print(f"scores: {scores}")

# Create zip pairs
pairs = list(zip(scores, items))
print(f"pairs: {pairs}")

# Sort them
sorted_pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
print(f"sorted_pairs worked: {sorted_pairs is not None}")

sorted_items = [pair[1] for pair in sorted_pairs]
result = [item.title for item in sorted_items]
print(f"Final result: {result}")
result
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_sorting_with_zip())
