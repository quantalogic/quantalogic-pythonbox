import pytest
from quantalogic_pythonbox import execute_async

@pytest.mark.asyncio
async def test_manual_stopiteration_in_generator():
    """PEP-479: StopIteration inside generator becomes RuntimeError."""
    code = '''
 def gen():
     raise StopIteration("boom")
 def main():
     try:
         next(gen())
     except Exception as e:
         return type(e).__name__
 '''
    result = await execute_async(code, entry_point='main')
    assert result.result == "RuntimeError"

@pytest.mark.asyncio
async def test_return_value_in_generator():
    """PEP-479: return in generator yields StopIteration with value."""
    code = '''
 def gen():
     if False:
         yield 1
     return "finished"
 def main():
     g = gen()
     try:
         next(g)
     except StopIteration as e:
         return e.value
 '''
    result = await execute_async(code, entry_point='main')
    assert result.result == "finished"

@pytest.mark.asyncio
async def test_stopiteration_after_yield():
    """PEP-479: manual StopIteration after yield carries value."""
    code = '''
 def gen():
     yield 1
     raise StopIteration("done")
 def main():
     g = gen()
     out = []
     try:
         while True:
             out.append(next(g))
     except StopIteration as e:
         val = e.value
     return (out, val)
 '''
    result = await execute_async(code, entry_point='main')
    assert result.result == ([1], "done")

@pytest.mark.asyncio
async def test_exception_chaining_sets_cause():
    """PEP-3134: raising from another exception sets __cause__."""
    code = '''
async def main():
    try:
        raise ValueError("orig")
    except ValueError as e:
        try:
            raise RuntimeError("new") from e
        except RuntimeError as re:
            cause = re.__cause__
            return (type(re).__name__, type(cause).__name__, cause.args[0])
'''
    result = await execute_async(code, entry_point='main')
    assert result.result == ('RuntimeError', 'ValueError', 'orig')

@pytest.mark.asyncio
async def test_exception_chaining_suppresses_context():
    """PEP-409: raising from None suppresses context and __cause__ is None."""
    code = '''
async def main():
    try:
        1/0
    except ZeroDivisionError:
        try:
            raise KeyError("key") from None
        except KeyError as ke:
            return (ke.__cause__, ke.__suppress_context__)
'''
    result = await execute_async(code, entry_point='main')
    assert result.result == (None, True)

@pytest.mark.asyncio
async def test_old_raise_syntax_error():
    """PEP-3109: old raise syntax removed."""
    code = '''
def main():
    raise ValueError, "msg"
'''
    result = await execute_async(code, entry_point='main')
    assert result.result is None
    assert 'SyntaxError' in result.error

@pytest.mark.asyncio
async def test_bare_raise_reraises_exception():
    """PEP-312: bare raise re-raises current exception."""
    code = '''
async def main():
    try:
        1/0
    except ZeroDivisionError:
        try:
            raise
        except ZeroDivisionError as e:
            return type(e).__name__
'''
    result = await execute_async(code, entry_point='main')
    assert result.result == 'ZeroDivisionError'

@pytest.mark.asyncio
async def test_exception_chaining_default_context():
    """PEP-3134: default exception chaining sets __context__ and no __cause__."""
    code = '''
async def main():
    try:
        1/0
    except ZeroDivisionError as e:
        try:
            raise KeyError("key")
        except KeyError as ke:
            return (ke.__cause__, isinstance(ke.__context__, ZeroDivisionError), ke.__suppress_context__)
'''
    result = await execute_async(code, entry_point='main')
    assert result.result == (None, True, False)

@pytest.mark.asyncio
async def test_async_generator_return_value_pep479():
    """PEP-479: return with value in async generator yields StopAsyncIteration.value."""
    code = '''
async def gen():
    yield 1
    return "done"

async def main():
    agen = gen()
    # consume first yield
    first = await agen.__anext__()
    try:
        await agen.__anext__()
    except StopAsyncIteration as e:
        return (first, e.value)
'''
    result = await execute_async(code, entry_point='main')
    assert result.result == (1, "done")

@pytest.mark.asyncio
async def test_sync_generator_throw_pep342():
    """PEP-342: throw() into sync generator propagates exception inside generator."""
    code = '''
 def gen():
     try:
         yield 1
     except ValueError as e:
         yield f"caught {e}"
 def main():
     g = gen()
     first = next(g)
     second = g.throw(ValueError("err"))
     return (first, second)
 '''
    result = await execute_async(code, entry_point='main')
    assert result.result == (1, "caught err")

@pytest.mark.asyncio
async def test_yield_from_exception_propagation_pep380():
    """PEP-380: yield from propagates exceptions from sub-generator."""
    code = '''
 def gen1():
     yield 1
     raise KeyError("k")
 def gen2():
     yield from gen1()
 def main():
     g = gen2()
     first = next(g)
     try:
         next(g)
     except KeyError as e:
         return (first, e.args[0])
 '''
    result = await execute_async(code, entry_point='main')
    assert result.result == (1, "k")
