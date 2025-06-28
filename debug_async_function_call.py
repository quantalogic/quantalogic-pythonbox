#!/usr/bin/env python3

"""Debug script to understand async function call behavior"""

import asyncio
from quantalogic_pythonbox import execute_async

async def main():
    source = """
async def get_slice(arr):
    return arr[1:3]

async def main():
    arr = [10, 20, 30, 40]
    sliced = await get_slice(arr)
    return sliced
"""
    
    result = await execute_async(source, entry_point="main")
    print("Result:", result.result)
    print("Error:", result.error)

if __name__ == "__main__":
    asyncio.run(main())
