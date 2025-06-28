#!/usr/bin/env python3

import asyncio
from quantalogic_pythonbox.execution import execute_async

async def main():
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

sorted(items, key=lambda x: x.get_score(), reverse=True)
'''
    
    result = await execute_async(code, allowed_modules=['asyncio'])
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
