import pytest
from quantalogic_pythonbox import execute_async


@pytest.mark.asyncio
async def test_positive_slicing():
    # Test positive index slicing on a list.
    source = """
async def compute():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    result = arr[1:4]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: ['b', 'c', 'd']"


@pytest.mark.asyncio
async def test_negative_slicing():
    # Test negative index slicing on a list.
    source = """
async def compute():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    result = arr[-3:]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: ['d', 'e', 'f']"


@pytest.mark.asyncio
async def test_slicing_with_negative_step():
    # Test slicing with a negative step on a list.
    source = """
async def compute():
    arr = [1, 2, 3, 4, 5]
    result = arr[::-1]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [5, 4, 3, 2, 1]"


@pytest.mark.asyncio
async def test_complex_slicing_operations():
    # Test complex slicing operations with computed indices.
    source = """
async def compute():
    arr = [10, 20, 30, 40, 50, 60, 70, 80]
    result = arr[2:len(arr)*3//4]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [30, 40, 50, 60]"


@pytest.mark.asyncio
async def test_complex_object_arrays_with_async_list():
    # Test filtering complex objects in a list with async operations.
    source = """
async def filter_admins(users):
    if not users:
        return []
    return [user for user in users if 'admin' in user['roles']]

async def compute():
    users = [
        {'id': 1, 'name': 'Alice', 'roles': ['admin', 'user']},
        {'id': 2, 'name': 'Bob', 'roles': ['user']},
        {'id': 3, 'name': 'Charlie', 'roles': ['editor', 'user']}
    ]
    admins = await filter_admins(users)
    admin_names = [admin['name'] for admin in admins]
    return f"Admins: {admin_names[0] if admin_names else 'None'}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Admins: Alice"


@pytest.mark.asyncio
async def test_nested_array_operations_with_async_list():
    # Test counting roles in nested array operations with async list.
    source = """
async def count_total_roles(users):
    role_counts = [len(user['roles']) for user in users]
    return sum(role_counts)

async def compute():
    users = [
        {'id': 1, 'name': 'Alice', 'roles': ['admin', 'user']},
        {'id': 2, 'name': 'Bob', 'roles': ['user']},
        {'id': 3, 'name': 'Charlie', 'roles': ['editor', 'user', 'reviewer']}
    ]
    total_roles = await count_total_roles(users)
    return f"Total roles: {total_roles}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Total roles: 6"


@pytest.mark.asyncio
async def test_array_mapping_operations():
    # Test mapping operations on an array of dictionaries.
    source = """
async def get_names(users):
    return [user['name'] for user in users]

async def compute():
    users = [
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'name': 'Bob'},
        {'id': 3, 'name': 'Charlie'}
    ]
    names = await get_names(users)
    return f"Names: {', '.join(names)}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Names: Alice, Bob, Charlie"


@pytest.mark.asyncio
async def test_nested_array_operations():
    # Test nested array operations with role counting.
    source = """
async def count_total_roles(users):
    role_counts = [len(user['roles']) for user in users]
    return sum(role_counts)

async def compute():
    users = [
        {'id': 1, 'name': 'Alice', 'roles': ['admin', 'user']},
        {'id': 2, 'name': 'Bob', 'roles': ['user']},
        {'id': 3, 'name': 'Charlie', 'roles': ['editor', 'user', 'reviewer']}
    ]
    total_roles = await count_total_roles(users)
    return f"Total roles: {total_roles}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Total roles: 6"


@pytest.mark.asyncio
async def test_complex_array_filtering():
    # Test filtering complex arrays with multiple conditions.
    source = """
async def filter_active_users(users):
    return [user for user in users if user.get('active') and len(user['roles']) > 0]

async def compute():
    users = [
        {'id': 1, 'name': 'Alice', 'roles': ['admin'], 'active': True},
        {'id': 2, 'name': 'Bob', 'roles': [], 'active': True},
        {'id': 3, 'name': 'Charlie', 'roles': ['user'], 'active': False}
    ]
    active_users = await filter_active_users(users)
    return f"Active users: {len(active_users)}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Active users: 1"


@pytest.mark.asyncio
async def test_debug_slice_calculation():
    # Test debug slice calculation (placeholder).
    source = """
async def compute():
    arr = [1, 2, 3, 4, 5]
    result = arr[1:3]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [2, 3]"


@pytest.mark.asyncio
async def test_debug_async_generator_iteration():
    # Test async generator iteration (placeholder).
    source = """
async def gen():
    for i in range(3):
        yield i

async def compute():
    result = [x async for x in gen()]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [0, 1, 2]"


@pytest.mark.asyncio
async def test_debug_string_join_with_async():
    # Test string join with async (placeholder).
    source = """
async def compute():
    arr = ['a', 'b', 'c']
    result = ''.join([x async for x in arr])
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: abc"


@pytest.mark.asyncio
async def test_precise_slice_calculation():
    # Test precise slice calculation.
    source = """
async def compute():
    arr = [1, 2, 3, 4, 5]
    result = arr[0:5:2]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [1, 3, 5]"


@pytest.mark.asyncio
async def test_async_generator_to_list():
    # Test converting async generator to list.
    source = """
async def gen():
    for i in range(4):
        yield i

async def compute():
    result = [x async for x in gen()]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [0, 1, 2, 3]"


@pytest.mark.asyncio
async def test_async_string_join():
    # Test async string join operation.
    source = """
async def compute():
    arr = ['x', 'y', 'z']
    result = ''.join([x async for x in arr])
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: xyz"


@pytest.mark.asyncio
async def test_empty_array_operations():
    # Test operations on empty arrays.
    source = """
async def compute():
    arr = []
    result = arr[:]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: []"


@pytest.mark.asyncio
async def test_array_edge_cases():
    # Test array slicing edge cases.
    source = """
async def compute():
    arr = ['a', 'b', 'c']
    result = arr[5:10]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: []"


@pytest.mark.asyncio
async def test_async_array_transformations():
    # Test async array transformations.
    source = """
async def transform(arr):
    return [x * 2 async for x in arr]

async def compute():
    arr = [1, 2, 3]
    result = await transform(arr)
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [2, 4, 6]"


@pytest.mark.asyncio
async def test_nested_async_operations():
    # Test nested async operations on arrays.
    source = """
async def double(arr):
    return [x * 2 async for x in arr]

async def compute():
    arr = [1, 2, 3]
    result = await double(await double(arr))
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: [4, 8, 12]"


@pytest.mark.asyncio
async def test_debug_async_generator_iteration_with_error_handling():
    # Test async generator with error handling (placeholder).
    source = """
async def gen():
    for i in range(3):
        if i == 2:
            raise ValueError("Error")
        yield i

async def compute():
    try:
        result = [x async for x in gen()]
        return f"Result: {result}"
    except ValueError:
        return "Caught error"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Caught error"


@pytest.mark.asyncio
async def test_debug_string_join_with_async_error_handling():
    # Test string join with async and error handling (placeholder).
    source = """
async def gen():
    for x in ['a', 'b']:
        yield x
    raise ValueError("Error")

async def compute():
    try:
        result = ''.join([x async for x in gen()])
        return f"Result: {result}"
    except ValueError:
        return "Caught error"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Caught error"


@pytest.mark.asyncio
async def test_step_slicing():
    # Test slicing with a step value.
    source = """
async def compute():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    result = arr[::2]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: ['a', 'c', 'e']"


@pytest.mark.asyncio
async def test_mixed_slicing():
    # Test mixed positive and negative index slicing.
    source = """
async def compute():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    result = arr[1:-2]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: ['b', 'c', 'd']"


@pytest.mark.asyncio
async def test_empty_slice():
    # Test slicing with empty range.
    source = """
async def compute():
    arr = ['a', 'b', 'c']
    result = arr[3:1]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: []"


@pytest.mark.asyncio
async def test_string_slicing():
    # Test string slicing.
    source = """
async def compute():
    s = "abcdefgh"
    result = s[2:5]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: cde"


@pytest.mark.asyncio
async def test_tuple_slicing():
    # Test tuple slicing.
    source = """
async def compute():
    t = (10, 20, 30, 40, 50)
    result = t[1:3]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: (20, 30)"


@pytest.mark.asyncio
async def test_custom_object_slicing():
    # Test slicing on a custom object with __getitem__. Note: Fails due to interpreter bug in handling async __getitem__ return.
    # Expected "Custom slice: slice(1, 5, 2)" per intended semantics, but interpreter returns None.
    source = """
class Sliceable:
    def __getitem__(self, s):
        return f"Custom slice: {s}"

async def compute():
    obj = Sliceable()
    result = obj[1:5:2]
    return result
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Custom slice: slice(1, 5, 2)"


@pytest.mark.asyncio
async def test_omitted_indices_slicing():
    # Test slicing with omitted indices.
    source = """
async def compute():
    arr = ['a', 'b', 'c']
    result = arr[:]
    return f"Result: {result}"
"""
    result = await execute_async(source, entry_point="compute", allowed_modules=[])
    assert result.result == "Result: ['a', 'b', 'c']"