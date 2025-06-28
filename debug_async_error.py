"""
Test that async method calls in lambdas still raise the error correctly
"""
import asyncio
from quantalogic_pythonbox.execution import execute_async

async def test_async_method_error():
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

try:
    sorted(items, key=lambda x: x.get_score(), reverse=True)
    result = "No error"
except TypeError as e:
    result = str(e)

result
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(test_async_method_error())
