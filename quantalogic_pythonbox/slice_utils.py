"""
Custom slice implementation for PythonBox interpreter.
"""

class CustomSlice:
    """
    A custom implementation of Python's slice object that formats its string representation
    as 'Slice(start,stop,step)' without spaces after commas and with capital 'S'.
    """
    
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
    
    def __str__(self):
        """
        Return string representation of the slice with custom formatting.
        """
        return f"Slice({self.start},{self.stop},{self.step})"
    
    def __repr__(self):
        """
        Return formal representation of the slice with custom formatting.
        """
        return self.__str__()
    
    def __eq__(self, other):
        """
        Compare this slice with another slice or CustomSlice.
        """
        if isinstance(other, CustomSlice):
            return (self.start == other.start and
                    self.stop == other.stop and
                    self.step == other.step)
        elif isinstance(other, slice):
            return (self.start == other.start and
                    self.stop == other.stop and
                    self.step == other.step)
        return False
    
    def indices(self, length):
        """
        Return a tuple (start, stop, step) representing the slice adjusted for a sequence of the specified length.
        """
        # This mimics the behavior of slice.indices()
        start, stop, step = self.start, self.stop, self.step
        
        # Handle None values
        if start is None:
            start = 0 if step > 0 else length - 1
        elif start < 0:
            start += length
            if start < 0:
                start = 0 if step > 0 else -1
        elif start >= length:
            start = length if step > 0 else length - 1
            
        if stop is None:
            stop = length if step > 0 else -1
        elif stop < 0:
            stop += length
            if stop < 0:
                stop = 0 if step > 0 else -1
        elif stop >= length:
            stop = length
            
        return (start, stop, step)
