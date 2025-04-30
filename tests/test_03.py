import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
async def test_zero_division_error():
    """Verify ZeroDivisionError is reported."""
    code = '''
async def main():
    return 1 / 0
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'ZeroDivisionError' in result.error

@pytest.mark.asyncio
async def test_type_error_invalid_operand():
    """Verify TypeError for invalid operands."""
    code = '''
async def main():
    return "text" + 5
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'TypeError' in result.error

@pytest.mark.asyncio
async def test_value_error_int_conversion():
    """Verify ValueError for invalid int conversion."""
    code = '''
async def main():
    return int("not_an_int")
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'ValueError' in result.error

@pytest.mark.asyncio
async def test_key_error_missing_key():
    """Verify KeyError for missing dict key."""
    code = '''
async def main():
    d = {"a": 1}
    return d["b"]
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'KeyError' in result.error

@pytest.mark.asyncio
async def test_index_error_list_out_of_range():
    """Verify IndexError for out-of-range list access."""
    code = '''
async def main():
    lst = [1, 2, 3]
    return lst[5]
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'IndexError' in result.error

@pytest.mark.asyncio
async def test_attribute_error_nonexistent_attribute():
    """Verify AttributeError for nonexistent attribute access."""
    code = '''
async def main():
    obj = object()
    return obj.nonexistent
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'AttributeError' in result.error

@pytest.mark.asyncio
async def test_custom_exception_propagation():
    """Verify propagation of custom exceptions."""
    code = '''
class CustomError(Exception):
    pass

async def main():
    raise CustomError("error occurred")
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'CustomError' in result.error
    assert 'error occurred' in result.error
