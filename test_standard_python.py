#!/usr/bin/env python3

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
    result = sorted(items, key=lambda x: x.get_score(), reverse=True)
    print(f"Sorted result: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
