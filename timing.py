
# timing.py



from contextlib import contextmanager
from timeit import default_timer



@contextmanager
def Timer(logger=None, callback='Total time: {time:.3f}s'):

    timer = _Timer()

    yield timer

    if logger is not None:

        logger.debug(callback.format(**{'time': timer.age}))



class _Timer(object):


    def __init__(self):

        self._start, self._checkpoint = self.now, self.now


    @property
    def now(self):

        # get current time

        return default_timer()    


    @property
    def start(self):

        # get timer start time

        return self._start


    @property
    def elapsed(self):

        # get elapsed time from self.checkpoint and reset self.checkpoint

        now = self.now
        elapsed = now - self._checkpoint
        self._checkpoint = now
        return elapsed


    @property
    def age(self):

        # get time since self.start

        return self.now - self.start


    def restart(self):

        # reset current checkpoint time

        self._checkpoint = self.now


    @property:
    def checkpoint(self):

        # get checkpoint time

        return self._checkpoint
