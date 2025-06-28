"""
Debug what the lambda returns when calling async method
"""
import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_lambda_async_method():
    code = '''
import asyncio

class HNItem:
    def __init__(self, title, score):
        self.title = title
        self.score = score

    async def get_score(self):
        await asyncio.sleep(0.01)
        return self.score

items = [HNItem("Item A", 300)]

# Test lambda with async method call
test_lambda = lambda x: x.get_score()
test_result = test_lambda(items[0])
print(f"Lambda result: {test_result}")
print(f"Lambda result type: {type(test_result)}")

# Test calling the method directly
direct_call = items[0].get_score()
print(f"Direct call result: {direct_call}")
print(f"Direct call type: {type(direct_call)}")

test_result
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_lambda_async_method())
