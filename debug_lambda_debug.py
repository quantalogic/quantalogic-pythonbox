"""
Debug what's happening inside the lambda
"""
import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_lambda_debug():
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

# Create zip pairs
pairs = list(zip(scores, items))
print(f"First pair: {pairs[0]}")
print(f"pair[0][0]: {pairs[0][0]}")

# Test lambda manually
test_lambda = lambda pair: pair[0]
test_result = test_lambda(pairs[0])
print(f"Lambda result: {test_result}")
print(f"Lambda result type: {type(test_result)}")

test_result
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_lambda_debug())
