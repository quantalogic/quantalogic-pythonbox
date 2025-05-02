# quantalogic_pythonbox/generator_wrapper.py
"""
Generator wrapper for handling synchronous generators in the PythonBox interpreter.
"""

import logging
from quantalogic_pythonbox.exceptions import ReturnException

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GeneratorWrapper:
    def __init__(self, gen):
        self.gen = gen
        self.closed = False
        self.return_value = None
        self.yielded_values = []
        logging.debug("GeneratorWrapper initialized")

    def __iter__(self):
        return self

    def __next__(self):
        logging.debug(f"Debug: Calling next on self.gen, type: {type(self.gen).__name__}")
        if self.closed:
            e = StopIteration()
            e.value = self.return_value
            raise e
        try:
            value = next(self.gen)
            self.yielded_values.append(value)
            return value
        except StopIteration as e:
            # Extract original StopIteration value, falling back to args if needed
            orig_val = getattr(e, 'value', None)
            if orig_val is None and e.args:
                orig_val = e.args[0]
            logging.debug(f"Debug: Caught StopIteration in __next__ with value: {orig_val}")
            self.closed = True
            self.return_value = orig_val
            e = StopIteration()
            e.value = self.return_value
            raise e
        except Exception as e:
            logging.error(f"Debug: Exception in __next__: {type(e).__name__}, message: {str(e)}")
            raise

    def send(self, value):
        if self.closed:
            e = StopIteration()
            e.value = self.return_value
            raise e
        try:
            logging.debug(f"Debug: Calling self.gen.send with value: {value}, type of self.gen: {type(self.gen).__name__}")
            sent_value = self.gen.send(value)
            self.yielded_values.append(sent_value)
            return sent_value
        except ReturnException as e:
            logging.debug(f"Debug: Caught ReturnException in send with value: {e.value}")
            self.closed = True
            self.return_value = e.value
            e = StopIteration()
            e.value = self.return_value
            raise e
        except StopIteration as e:
            # Extract original StopIteration value, falling back to args if needed
            orig_val = getattr(e, 'value', None)
            if orig_val is None and e.args:
                orig_val = e.args[0]
            logging.debug(f"Debug: Caught StopIteration in send with value: {orig_val}")
            self.closed = True
            self.return_value = orig_val
            e = StopIteration()
            e.value = self.return_value
            raise e
        except Exception as e:
            logging.debug(f"Debug: Exception in send: {type(e).__name__}: {str(e)}")
            raise

    def throw(self, exc_type, exc_val=None, exc_tb=None):
        logging.debug(f"Debug: GeneratorWrapper.throw called with exc_type: {exc_type}, exc_val: {exc_val}, exc_tb: {exc_tb}")
        if self.closed:
            e = StopIteration()
            e.value = self.return_value
            raise e
        logging.debug(f"Throwing exception of type {exc_type} with value {exc_val} into generator")
        try:
            if exc_val is None:
                exc_val = exc_type() if isinstance(exc_type, type) else exc_type
            elif isinstance(exc_val, type):
                exc_val = exc_val()
            thrown_value = self.gen.throw(exc_type, exc_val, exc_tb)
            self.yielded_values.append(thrown_value)
            return thrown_value
        except ReturnException as e:
            logging.debug(f"Debug: Caught ReturnException in throw with value: {e.value}")
            self.closed = True
            self.return_value = e.value
            e = StopIteration()
            e.value = self.return_value
            raise e
        except StopIteration as e:
            # Extract original StopIteration value, falling back to args if needed
            orig_val = getattr(e, 'value', None)
            if orig_val is None and e.args:
                orig_val = e.args[0]
            logging.debug(f"Debug: Caught StopIteration in throw with value: {orig_val}")
            self.closed = True
            self.return_value = orig_val
            e = StopIteration()
            e.value = self.return_value
            raise e
        except Exception as e:
            raise

    def close(self):
        if not self.closed:
            try:
                self.gen.close()
            except Exception:
                pass
            self.closed = True