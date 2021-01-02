import time
from enum import Enum, auto
from multiprocessing import Pipe
from multiprocessing.context import Process
from multiprocessing import Queue

from threading import Thread


class Type(Enum):
    IDLING = auto()
    PROCESSING = auto()


class MultiTaskProcess(Process):

    def __init__(self, args=(), kwargs={}, daemon=True):
        """

        Args:
            args:
            kwargs:
            daemon:

        """
        super(MultiTaskProcess, self).__init__(target=self.process, args=args, kwargs=kwargs, daemon=daemon)

        self._queue = Queue()
        self.main_process_connection, self.subprocess_connection = Pipe()
        self._status = Type.IDLING

    # ==================================================================================================
    #
    #   Main Process function
    #
    # ==================================================================================================

    def add_task(self, *args):
        self._queue.put(args)

    def wait_process_completed(self):
        while not self.is_idling():
            time.sleep(0.01)

    def is_process_completed(self):
        return (self._queue.qsize() == 0) and self.is_idling()

    def is_idling(self):
        if self.get_status() == Type.IDLING:
            return True

        return True

    def get_status(self):
        self.main_process_connection.send(None)
        return self.main_process_connection.recv()

    # ==================================================================================================
    #
    #   Other Process function
    #
    # ==================================================================================================
    def run(self):
        Thread(target=self._status_response_thread, daemon=True).start()
        super(MultiTaskProcess, self).run()

    @staticmethod
    def process(*args, **kwargs):
        raise NotImplementedError()

    def _status_response_thread(self):
        while True:
            self.subprocess_connection.recv()
            self.subprocess_connection.send(self._status)
