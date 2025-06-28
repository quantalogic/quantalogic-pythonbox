"""
Debug script to understand the solution part failing
"""
import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_solution_code():
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
print(f"scores = {scores}")
print(f"type of scores[0] = {type(scores[0])}")

sorted_pairs = sorted(zip(scores, items), key=lambda pair: pair[0], reverse=True)
print(f"sorted_pairs = {sorted_pairs}")

sorted_items = [pair[1] for pair in sorted_pairs]
print(f"sorted_items titles = {[item.title for item in sorted_items]}")

[item.title for item in sorted_items]
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_solution_code())
