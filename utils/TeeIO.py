import contextlib
import sys

class TeeIO:
    def __init__(self, original, target):
        self.original = original
        self.target = target

    def write(self, text):
        self.original.write(text)
        self.target.write(text)

    def flush(self):
        self.original.flush()
        self.target.flush()

@contextlib.contextmanager
def tee_stdout(target):
    tee = TeeIO(sys.stdout, target)
    with contextlib.redirect_stdout(tee):
        yield