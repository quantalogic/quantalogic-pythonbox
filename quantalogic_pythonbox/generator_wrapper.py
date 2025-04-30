# quantalogic_pythonbox/generator_wrapper.py
"""
Generator wrapper for handling synchronous generators in the PythonBox interpreter.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen
        self.closed = False
        self.return_value = None
        self.yielded_values = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration(self.return_value)
        try:
            value = next(self.gen)
            self.yielded_values.append(value)
            return value
        except StopIteration as e:
            self.closed = True
            self.return_value = e.value if hasattr(e, 'value') else None
            raise StopIteration(self.return_value)

    def send(self, value):
        if self.closed:
            raise StopIteration(self.return_value)
        try:
            sent_value = self.gen.send(value)
            self.yielded_values.append(sent_value)
            return sent_value
        except StopIteration as e:
            self.closed = True
            self.return_value = e.value if hasattr(e, 'value') else None
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
            
            thrown_value = self.gen.throw(exc_type, exc_val, exc_tb)
            self.yielded_values.append(thrown_value)
            return thrown_value
        except StopIteration as e:
            self.closed = True
            self.return_value = e.value if hasattr(e, 'value') else None
            raise StopIteration(self.return_value)

    def close(self):
        if not self.closed:
            try:
                self.gen.close()
            except Exception:
                pass
            self.closed = True