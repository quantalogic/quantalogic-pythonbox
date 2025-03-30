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