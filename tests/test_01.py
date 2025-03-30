import pytest
from quantalogic_pythonbox import execute_async
import math

@pytest.mark.asyncio
async def test_simple_async_function():
    code = '''
async def sinus(x: float) -> float:
    return float(math.sin(x))

async def compute() -> float:
    result = await sinus(10) + 8.4
    return result

async def main() -> float:
    return await compute()
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={'math': math}
    )
    assert result.result == math.sin(10) + 8.4

@pytest.mark.asyncio
async def test_simple_sync_function():
    code = '''
async def sinus(x: float) -> float:
    return float(math.sin(x))

async def main():
    # Calculate sin(4.7)
    step5_sin_value = await sinus(x=4.7)
    step5_result = step5_sin_value + 8.1
    return f"Task completed: {step5_result}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={'math': math}
    )
    assert result.result == f"Task completed: {math.sin(4.7) + 8.1}"