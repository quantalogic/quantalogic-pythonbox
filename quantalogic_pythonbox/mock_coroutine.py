"""
Mock coroutine implementation for proper error simulation.
"""

import collections.abc

class MockCoroutine:
    """A mock coroutine that raises appropriate errors when compared"""
    
    def __init__(self, async_func, args, kwargs):
        self.async_func = async_func
        self.args = args
        self.kwargs = kwargs
        
    def __lt__(self, other):
        raise TypeError("'<' not supported between instances of 'coroutine' and 'coroutine'")
        
    def __le__(self, other):
        raise TypeError("'<=' not supported between instances of 'coroutine' and 'coroutine'")
        
    def __gt__(self, other):
        raise TypeError("'>' not supported between instances of 'coroutine' and 'coroutine'")
        
    def __ge__(self, other):
        raise TypeError("'>=' not supported between instances of 'coroutine' and 'coroutine'")
        
    def __eq__(self, other):
        return False  # coroutines are never equal unless they are the same object
        
    def __ne__(self, other):
        return True
        
    def __str__(self):
        func_name = "unknown"
        if hasattr(self.async_func, '__name__'):
            func_name = self.async_func.__name__
        elif hasattr(self.async_func, 'node') and hasattr(self.async_func.node, 'name'):
            func_name = self.async_func.node.name
        return f"<coroutine object {func_name} at {hex(id(self))}>"
        
    def __repr__(self):
        return self.__str__()
        
    def close(self):
        pass
        
    def send(self, value):
        raise StopIteration
        
    def throw(self, exc_type, exc_value=None, traceback=None):
        raise exc_type(exc_value) if exc_value else exc_type()
        
    def __await__(self):
        async def coro():
            return await self.async_func(*self.args, **self.kwargs)
        return coro().__await__()

# Register MockCoroutine as a coroutine type so asyncio.iscoroutine() recognizes it
collections.abc.Coroutine.register(MockCoroutine)
