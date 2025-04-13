"""
Async-aware versions of built-in functions for the PythonBox interpreter.
These are designed to handle coroutines properly in various contexts.
"""

import asyncio
from typing import Any, Callable, Iterable, List, Optional, TypeVar

T = TypeVar('T')

async def async_sorted(iterable: Iterable[T], *, key: Optional[Callable[[T], Any]] = None, reverse: bool = False) -> List[T]:
    """
    Async-aware version of the sorted() built-in function.
    
    Properly handles key functions that return coroutines by awaiting them.
    This allows sorting to work correctly in async contexts.
    
    Args:
        iterable: The iterable to sort
        key: Optional key function that may return a coroutine
        reverse: Whether to sort in reverse order
        
    Returns:
        A sorted list of the items from iterable
    """
    items = list(iterable)
    
    if key is not None:
        # Get all the keys (awaiting any coroutines that are returned)
        keys = []
        for item in items:
            key_value = key(item)
            if asyncio.iscoroutine(key_value):
                key_value = await key_value
            keys.append(key_value)
        
        # Create (key, index) pairs for stable sorting
        pairs = [(keys[i], i, items[i]) for i in range(len(items))]
        
        # Sort by key, then by original index for stability
        pairs.sort(key=lambda x: (x[0], x[1]), reverse=reverse)
        
        # Extract the sorted items
        return [pair[2] for pair in pairs]
    else:
        # No key function, just regular sort
        return sorted(items, reverse=reverse)
