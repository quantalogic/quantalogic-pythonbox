# quantalogic_pythonbox/generator_wrapper.py
"""
Generator wrapper for handling synchronous generators in the PythonBox interpreter.
"""

class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen
        self.closed = False
        self.return_value = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration(self.return_value)
        try:
            return next(self.gen)
        except StopIteration as e:
            self.closed = True
            # Make sure to capture the return value from the generator
            if hasattr(e, 'value'):
                self.return_value = e.value
            raise StopIteration(self.return_value)

    def send(self, value):
        if self.closed:
            raise StopIteration(self.return_value)
        try:
            return self.gen.send(value)
        except StopIteration as e:
            self.closed = True
            if hasattr(e, 'value'):
                self.return_value = e.value
            raise StopIteration(self.return_value)

    def throw(self, exc_type, exc_val=None, exc_tb=None):
        if self.closed:
            raise StopIteration(self.return_value)
        try:
            if exc_val is None:
                if isinstance(exc_type, type):
                    exc_val = exc_type()
                else:
                    exc_val = exc_type
            elif isinstance(exc_val, type):
                exc_val = exc_val()
            
            return self.gen.throw(exc_type, exc_val, exc_tb)
        except StopIteration as e:
            self.closed = True
            if hasattr(e, 'value'):
                self.return_value = e.value
            raise StopIteration(self.return_value)

    def close(self):
        if not self.closed:
            try:
                self.gen.close()
            except Exception:
                pass
            self.closed = True