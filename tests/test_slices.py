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

@pytest.mark.asyncio
async def test_array_operations():
    code = '''
async def create_array() -> list:
    return [1, 2, 3, 4, 5]

async def sum_array(arr: list) -> float:
    return sum(arr)

async def main():
    arr = await create_array()
    total = await sum_array(arr)
    return f"Array sum: {total}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Array sum: 15"

@pytest.mark.asyncio
async def test_array_slicing():
    code = '''
async def get_slice(arr: list, start: int, end: int) -> list:
    return arr[start:end]

async def main():
    arr = [10, 20, 30, 40, 50]
    sliced = await get_slice(arr, 1, 4)
    return f"Slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Slice: [20, 30, 40]"

@pytest.mark.asyncio
async def test_array_slicing_with_step():
    code = '''
async def get_every_second_item(arr: list) -> list:
    return arr[::2]

async def main():
    arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    sliced = await get_every_second_item(arr)
    return f"Every second item: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Every second item: [0, 2, 4, 6, 8]"

@pytest.mark.asyncio
async def test_negative_slicing():
    code = '''
async def get_last_three(arr: list) -> list:
    return arr[-3:]

async def main():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    sliced = await get_last_three(arr)
    return f"Last three: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Last three: ['d', 'e', 'f']"

@pytest.mark.asyncio
async def test_slicing_with_negative_step():
    code = '''
async def reverse_array(arr: list) -> list:
    return arr[::-1]

async def main():
    arr = [1, 2, 3, 4, 5]
    reversed_arr = await reverse_array(arr)
    return f"Reversed: {reversed_arr}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Reversed: [5, 4, 3, 2, 1]"

@pytest.mark.asyncio
async def test_complex_slicing_operations():
    code = '''
async def get_middle_chunk(arr: list) -> list:
    return arr[len(arr)//4 : 3*len(arr)//4]

async def main():
    arr = [10, 20, 30, 40, 50, 60, 70, 80]
    middle = await get_middle_chunk(arr)
    return f"Middle chunk: {middle}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Middle chunk: [30, 40, 50, 60]"

@pytest.mark.asyncio
async def test_complex_object_arrays_with_async_list():
    """Test complex object arrays with explicit async list conversion"""
    code = '''
async def create_users() -> list:
    return [
        {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
        {"id": 2, "name": "Bob", "roles": ["user"]},
        {"id": 3, "name": "Charlie", "roles": ["editor", "user"]}
    ]

async def filter_admins(users: list) -> list:
    if not users:
        return []
    return [user for user in users if "admin" in user.get("roles", [])]

async def main():
    users = await create_users()
    admins = await filter_admins(users)
    admin_names = [user['name'] for user in admins]
    return f"Found {len(admins)} admin(s): {', '.join(admin_names) if admins else 'none'}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Found 1 admin(s): Alice"

@pytest.mark.asyncio
async def test_nested_array_operations_with_async_list():
    """Test nested array operations with explicit async list conversion"""
    code = '''
async def get_users() -> list:
    return [
        {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
        {"id": 2, "name": "Bob", "roles": ["user"]},
        {"id": 3, "name": "Charlie", "roles": ["editor", "user", "reviewer"]}
    ]

async def count_total_roles(users: list) -> int:
    roles = [len(user['roles']) for user in users]
    return sum(roles)

async def main():
    users = await get_users()
    total_roles = await count_total_roles(users)
    return f"Total roles: {total_roles}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Total roles: 6"

@pytest.mark.asyncio
async def test_array_mapping_operations():
    code = '''
async def get_names(users: list) -> list:
    return [user['name'] for user in users]

async def main():
    users = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"}
    ]
    names = await get_names(users)
    return f"Names: {', '.join(names)}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Names: Alice, Bob, Charlie"

@pytest.mark.asyncio
async def test_nested_array_operations():
    code = '''
async def get_users() -> list:
    return [
        {"id": 1, "name": "Alice", "roles": ["admin", "user"]},
        {"id": 2, "name": "Bob", "roles": ["user"]},
        {"id": 3, "name": "Charlie", "roles": ["editor", "user", "reviewer"]}
    ]

async def count_total_roles(users: list) -> int:
    roles = [len(user['roles']) for user in users]
    return sum(roles)

async def main():
    users = await get_users()
    total_roles = await count_total_roles(users)
    return f"Total roles: {total_roles}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Total roles: 6"

@pytest.mark.asyncio
async def test_complex_array_filtering():
    code = '''
async def filter_active_users(users: list) -> list:
    return [
        user for user in users 
        if user.get('active', False) 
        and len(user.get('roles', [])) > 0
    ]

async def main():
    users = [
        {"id": 1, "name": "Alice", "roles": ["admin"], "active": True},
        {"id": 2, "name": "Bob", "roles": [], "active": True},
        {"id": 3, "name": "Charlie", "roles": ["user"], "active": False}
    ]
    active_users = await filter_active_users(users)
    return f"Active users: {len(active_users)}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Active users: 1"

@pytest.mark.asyncio
async def test_debug_slice_calculation():
    """Test the slice calculation logic directly"""
    arr = [10, 20, 30, 40, 50, 60, 70, 80]
    length = len(arr)
    start = length // 4
    end = 3 * length // 4
    
    # Debug print the values
    print(f"Length: {length}, Start: {start}, End: {end}")
    
    # Verify the slice
    assert arr[start:end] == [30, 40, 50, 60]

@pytest.mark.asyncio
async def test_debug_async_generator_iteration():
    """Test if async generators can be iterated"""
    async def gen_users():
        users = [
            {'name': 'Alice', 'roles': ['admin'], 'active': True},
            {'name': 'Bob', 'roles': ['user'], 'active': False},
            {'name': 'Charlie', 'roles': ['user', 'editor'], 'active': True}
        ]
        for user in users:
            yield user
    
    # Try iterating the async generator
    users = []
    async for user in gen_users():
        users.append(user)
    
    assert len(users) == 3
    assert users[0]['name'] == 'Alice'

@pytest.mark.asyncio
async def test_debug_string_join_with_async():
    """Test string join operation with async data"""
    async def get_names():
        return ['Alice', 'Bob', 'Charlie']
    
    names = await get_names()
    result = ", ".join(names)
    assert result == "Alice, Bob, Charlie"

@pytest.mark.asyncio
async def test_precise_slice_calculation():
    """Verify exact slice indices"""
    arr = [10, 20, 30, 40, 50, 60, 70, 80]
    length = len(arr)
    
    # Expected slice indices
    expected_start = 1  # length//4 = 2, but we want 20 included
    expected_end = 6    # 3*length//4 = 6
    
    print(f"Expected slice: {arr[expected_start:expected_end]}")
    assert arr[expected_start:expected_end] == [20, 30, 40, 50, 60]

@pytest.mark.asyncio
async def test_async_generator_to_list():
    """Test converting async generator to list"""
    async def gen_users():
        users = [
            {'name': 'Alice', 'roles': ['admin'], 'active': True},
            {'name': 'Bob', 'roles': ['user'], 'active': False}
        ]
        for user in users:
            yield user
    
    # Convert async generator to list
    users = [user async for user in gen_users()]
    assert len(users) == 2
    assert isinstance(users, list)

@pytest.mark.asyncio
async def test_async_string_join():
    """Test string join with async list conversion"""
    async def get_admins():
        users = [
            {'name': 'Alice', 'roles': ['admin']},
            {'name': 'Bob', 'roles': ['user']}
        ]
        return [u['name'] for u in users if 'admin' in u['roles']]
    
    admins = await get_admins()
    result = ", ".join(admins)
    assert result == "Alice"

@pytest.mark.asyncio
async def test_empty_array_operations():
    """Test operations on empty arrays"""
    async def process_empty(arr):
        return {
            'length': len(arr),
            'slice': arr[1:3],
            'sum': sum(arr) if arr else 0
        }
    
    result = await process_empty([])
    assert result == {'length': 0, 'slice': [], 'sum': 0}

@pytest.mark.asyncio
async def test_array_edge_cases():
    """Test array edge cases"""
    arr = [1, 2, 3, 4, 5]
    
    # Test various slice scenarios
    assert arr[:100] == arr  # Slice beyond array length
    assert arr[-100:] == arr  # Negative slice beyond array length
    assert arr[100:] == []    # Start beyond array length
    assert arr[::-2] == [5, 3, 1]  # Negative step

@pytest.mark.asyncio
async def test_async_array_transformations():
    """Test chained async array transformations"""
    async def get_data():
        return [1, 2, 3, 4, 5]
    
    async def process(data):
        return [x * 2 for x in data if x % 2 == 0]
    
    data = await get_data()
    processed = await process(data)
    assert processed == [4, 8]

@pytest.mark.asyncio
async def test_nested_async_operations():
    """Test nested async operations with arrays"""
    async def outer():
        async def inner(arr):
            return [x * x for x in arr]
        
        data = [1, 2, 3]
        return await inner(data)
    
    result = await outer()
    assert result == [1, 4, 9]

@pytest.mark.asyncio
async def test_debug_async_generator_iteration_with_error_handling():
    """Test async generator iteration with error handling"""
    async def gen_users():
        users = [
            {'name': 'Alice', 'roles': ['admin'], 'active': True},
            {'name': 'Bob', 'roles': ['user'], 'active': False},
            {'name': 'Charlie', 'roles': ['user', 'editor'], 'active': True}
        ]
        for user in users:
            yield user
    
    # Test iteration with try/except
    users = []
    try:
        async for user in gen_users():
            users.append(user)
    except Exception as e:
        pytest.fail(f"Async iteration failed: {str(e)}")
    
    assert len(users) == 3
    assert users[0]['name'] == 'Alice'

@pytest.mark.asyncio
async def test_debug_string_join_with_async_error_handling():
    """Test string join with async data and error handling"""
    async def get_names():
        return ['Alice', 'Bob', 'Charlie']
    
    try:
        names = await get_names()
        result = ", ".join(names)
        assert result == "Alice, Bob, Charlie"
    except Exception as e:
        pytest.fail(f"String join failed: {str(e)}")

@pytest.mark.asyncio
async def test_positive_slicing():
    code = '''
async def get_middle_three(arr: list) -> list:
    return arr[1:4]

async def main():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    sliced = await get_middle_three(arr)
    return f"Middle three: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Middle three: ['b', 'c', 'd']"

@pytest.mark.asyncio
async def test_step_slicing():
    code = '''
async def get_every_second(arr: list) -> list:
    return arr[::2]

async def main():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    sliced = await get_every_second(arr)
    return f"Every second: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Every second: ['a', 'c', 'e']"

@pytest.mark.asyncio
async def test_mixed_slicing():
    code = '''
async def get_mixed_slice(arr: list) -> list:
    return arr[1:-2]

async def main():
    arr = ['a', 'b', 'c', 'd', 'e', 'f']
    sliced = await get_mixed_slice(arr)
    return f"Mixed slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Mixed slice: ['b', 'c', 'd']"

@pytest.mark.asyncio
async def test_empty_slice():
    code = '''
async def get_empty_slice(arr: list) -> list:
    return arr[10:20]

async def main():
    arr = ['a', 'b', 'c']
    sliced = await get_empty_slice(arr)
    return f"Empty slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Empty slice: []"

@pytest.mark.asyncio
async def test_string_slicing():
    code = '''
async def get_string_slice(s: str) -> str:
    return s[2:5]

async def main():
    s = "abcdefgh"
    sliced = await get_string_slice(s)
    return f"String slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "String slice: cde"

@pytest.mark.asyncio
async def test_tuple_slicing():
    code = '''
async def get_tuple_slice(t: tuple) -> tuple:
    return t[1:3]

async def main():
    t = (10, 20, 30, 40, 50)
    sliced = await get_tuple_slice(t)
    return f"Tuple slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Tuple slice: (20, 30)"

@pytest.mark.asyncio
async def test_custom_object_slicing():
    code = '''
class Sliceable:
    def __getitem__(self, key):
        if isinstance(key, slice):
            return f"Slice({key.start},{key.stop},{key.step})"
        return key

async def get_custom_slice(obj) -> str:
    return obj[1:5:2]

async def main():
    obj = Sliceable()
    sliced = await get_custom_slice(obj)
    return f"Custom slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Custom slice: Slice(1,5,2)"

@pytest.mark.asyncio
async def test_omitted_indices_slicing():
    code = '''
async def get_full_slice(arr: list) -> list:
    return arr[:]

async def main():
    arr = ['a', 'b', 'c']
    sliced = await get_full_slice(arr)
    return f"Full slice: {sliced}"
'''
    result = await execute_async(
        code,
        entry_point='main',
        namespace={}
    )
    assert result.result == "Full slice: ['a', 'b', 'c']"