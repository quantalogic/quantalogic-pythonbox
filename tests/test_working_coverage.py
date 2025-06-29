"""
Working coverage tests for quantalogic_pythonbox - focusing on straightforward scenarios
"""
import pytest
import asyncio
from quantalogic_pythonbox import execute_async


class TestBasicCoverage:
    """Basic tests that work with current interpreter limitations"""
    
    @pytest.mark.asyncio
    async def test_simple_arithmetic(self):
        """Test basic arithmetic operations"""
        code = """
x = 10
y = 20
result = x + y * 2
"""
        result = await execute_async(code)
        assert result.local_variables["result"] == 50
    
    @pytest.mark.asyncio 
    async def test_list_operations(self):
        """Test list operations and comprehensions"""
        code = """
numbers = [1, 2, 3, 4, 5]
squares = [x * x for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
result = len(squares) + len(evens)
"""
        result = await execute_async(code)
        assert result.local_variables["squares"] == [1, 4, 9, 16, 25]
        assert result.local_variables["evens"] == [2, 4]
        assert result.local_variables["result"] == 7
        
    @pytest.mark.asyncio
    async def test_dictionary_operations(self):
        """Test dictionary comprehensions and operations"""
        code = """
data = {'a': 1, 'b': 2, 'c': 3}
doubled = {k: v * 2 for k, v in data.items()}
result = sum(doubled.values())
"""
        result = await execute_async(code)
        assert result.local_variables["doubled"] == {'a': 2, 'b': 4, 'c': 6}
        assert result.local_variables["result"] == 12
    
    @pytest.mark.asyncio
    async def test_function_definitions(self):
        """Test function definitions and calls"""
        code = """
def add(a, b):
    return a + b

def multiply(x, y=2):
    return x * y

result1 = add(3, 4)
result2 = multiply(5)
result3 = multiply(5, 3)
"""
        result = await execute_async(code)
        assert result.local_variables["result1"] == 7
        assert result.local_variables["result2"] == 10
        assert result.local_variables["result3"] == 15
    
    @pytest.mark.asyncio
    async def test_control_flow(self):
        """Test if/else and loops"""
        code = """
results = []
for i in range(5):
    if i % 2 == 0:
        results.append(i)
    else:
        results.append(i * 2)

total = 0
x = 0
while x < 3:
    total += x
    x += 1
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [0, 2, 2, 6, 4]
        assert result.local_variables["total"] == 3
    
    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test try/except blocks"""
        code = """
results = []
try:
    x = 10 / 2
    results.append(x)
except ZeroDivisionError:
    results.append("error")

try:
    y = 10 / 0
    results.append(y)
except ZeroDivisionError:
    results.append("caught_error")
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [5.0, "caught_error"]
    
    @pytest.mark.asyncio
    async def test_nested_structures(self):
        """Test nested lists and dictionaries"""
        code = """
data = [
    {"name": "Alice", "scores": [85, 90, 78]},
    {"name": "Bob", "scores": [92, 88, 95]}
]

averages = []
for person in data:
    avg = sum(person["scores"]) / len(person["scores"])
    averages.append(avg)

result = len(averages)
"""
        result = await execute_async(code)
        assert len(result.local_variables["averages"]) == 2
        assert result.local_variables["result"] == 2
    
    @pytest.mark.asyncio
    async def test_string_operations(self):
        """Test string operations and methods"""
        code = """
text = "Hello World"
upper_text = text.upper()
lower_text = text.lower()
words = text.split()
joined = "-".join(words)
result = len(joined)
"""
        result = await execute_async(code)
        assert result.local_variables["upper_text"] == "HELLO WORLD"
        assert result.local_variables["lower_text"] == "hello world"
        assert result.local_variables["words"] == ["Hello", "World"]
        assert result.local_variables["joined"] == "Hello-World"
        assert result.local_variables["result"] == 11
    
    @pytest.mark.asyncio
    async def test_lambda_functions(self):
        """Test lambda functions"""
        code = """
square = lambda x: x * x
add = lambda a, b: a + b

numbers = [1, 2, 3, 4, 5]
squared = list(map(square, numbers))
result = add(10, 20)
"""
        result = await execute_async(code)
        assert result.local_variables["squared"] == [1, 4, 9, 16, 25]
        assert result.local_variables["result"] == 30
    
    @pytest.mark.asyncio
    async def test_set_operations(self):
        """Test set operations"""
        code = """
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
union = set1 | set2
intersection = set1 & set2
difference = set1 - set2
result = len(union)
"""
        result = await execute_async(code)
        assert result.local_variables["union"] == {1, 2, 3, 4, 5, 6}
        assert result.local_variables["intersection"] == {3, 4}
        assert result.local_variables["difference"] == {1, 2}
        assert result.local_variables["result"] == 6


class TestAdvancedCoverage:
    """Advanced tests to cover more complex scenarios"""
    
    @pytest.mark.asyncio
    async def test_class_definitions(self):
        """Test class definitions and methods"""
        code = """
class Counter:
    def __init__(self, start=0):
        self.value = start
    
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value

counter = Counter(5)
result1 = counter.increment()
result2 = counter.get_value()
"""
        result = await execute_async(code)
        assert result.local_variables["result1"] == 6
        assert result.local_variables["result2"] == 6
    
    @pytest.mark.asyncio
    async def test_generators(self):
        """Test generator functions"""
        code = """
def number_generator(n):
    for i in range(n):
        yield i * 2

gen = number_generator(3)
results = list(gen)
"""
        result = await execute_async(code)
        assert result.local_variables["results"] == [0, 2, 4]
    
    @pytest.mark.asyncio
    async def test_nested_functions(self):
        """Test nested function definitions"""
        code = """
def outer(x):
    def inner(y):
        return x + y
    return inner

add_5 = outer(5)
result = add_5(10)
"""
        result = await execute_async(code)
        assert result.local_variables["result"] == 15
    
    @pytest.mark.asyncio
    async def test_multiple_assignment(self):
        """Test multiple assignment patterns"""
        code = """
a, b = 10, 20
x, y, z = [1, 2, 3]
first, *rest = [10, 20, 30, 40]
result = a + b + x + y + z + first + sum(rest)
"""
        result = await execute_async(code)
        assert result.local_variables["a"] == 10
        assert result.local_variables["b"] == 20
        assert result.local_variables["x"] == 1
        assert result.local_variables["y"] == 2
        assert result.local_variables["z"] == 3
        assert result.local_variables["first"] == 10
        assert result.local_variables["rest"] == [20, 30, 40]
        assert result.local_variables["result"] == 126
    
    @pytest.mark.asyncio
    async def test_complex_expressions(self):
        """Test complex expressions and operator precedence"""
        code = """
x = 2
y = 3
z = 4

result1 = x + y * z  # 2 + 12 = 14
result2 = (x + y) * z  # 5 * 4 = 20
result3 = x ** y + z  # 8 + 4 = 12
result4 = not (x > y and y < z)  # not (False and True) = not False = True
"""
        result = await execute_async(code)
        assert result.local_variables["result1"] == 14
        assert result.local_variables["result2"] == 20
        assert result.local_variables["result3"] == 12
        assert result.local_variables["result4"] == True
