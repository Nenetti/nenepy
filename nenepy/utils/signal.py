import os
import signal


class Signal:

    def __init__(self, callback):
        self._pid = os.getpid()
        self._callback = callback

    @classmethod
    def init_signal(cls, signal_num, callback):
        return signal.signal(signal_num, cls(callback))

    def __call__(self, *args, **kwargs):
        if self._pid == os.getpid():
            self._callback(*args, **kwargs)
