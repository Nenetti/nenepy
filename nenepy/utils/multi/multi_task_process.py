import os
import signal
import sys
import time
from enum import Enum, auto
from multiprocessing import Pipe
from multiprocessing.context import Process
from multiprocessing import Queue

from threading import Thread


class Status(Enum):
    IDLING = auto()
    PROCESSING = auto()


class Request(Enum):
    STATUS = auto()
    CLOSE = auto()


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
        self._main_process_connection, self._subprocess_connection = Pipe()
        self._status = Status.IDLING
        self._to_close = False

    # ==================================================================================================
    #
    #    Shared function
    #
    # ==================================================================================================
    @property
    def qsize(self):
        return self._queue.qsize()

    @property
    def status(self):
        self._main_process_connection.send(Request.STATUS)
        return self._main_process_connection.recv()

    @property
    def need_to_process(self):
        return self.qsize > 0

    @property
    def is_closed(self):
        return self._closed

    # ==================================================================================================
    #
    #   Main Process function
    #
    # ==================================================================================================

    def add_task(self, *args):
        if not self.is_closed:
            self._queue.put(args)
        else:
            raise ValueError("process object is closed")

    def wait_process_completed(self):
        while not self.is_process_completed():
            time.sleep(0.01)

    def is_process_completed(self):
        return (self.qsize == 0) and (self.status == Status.IDLING)

    def close_with_waiting(self):
        if self.is_closed:
            return

        if self.exitcode == 1:
            self.close()
            return

        if not self.is_process_completed():
            self.wait_process_completed()

        self.close()

    def close(self):
        if not self.is_closed:
            self._main_process_connection.send(Request.CLOSE)
            self.join()
            super(MultiTaskProcess, self).close()

    # ==================================================================================================
    #
    #   Sub Process function
    #
    # ==================================================================================================

    def on_start(self):
        pass

    def on_exit(self):
        pass

    def process(self, *task):
        raise NotImplementedError()

    def run(self):
        signal.signal(signal.SIGINT, self._exit_signal)
        self.on_start()
        thread = Thread(target=self._status_response_thread, daemon=True)
        thread.start()
        while not self._to_close:
            if self.need_to_process:
                self._status = Status.PROCESSING
                task = self._queue.get()
                self.process(*task)
                if not self.need_to_process:
                    self._status = Status.IDLING
            time.sleep(0.01)

        self.on_exit()
        if thread.is_alive():
            thread.join()

        print(f"PID {os.getpid()}: {self.__class__.__name__} {id(self)}: closed")

    def _status_response_thread(self):
        while not self._to_close:
            request = self._subprocess_connection.recv()
            if request is Request.STATUS:
                self._subprocess_connection.send(self._status)
            elif request is Request.CLOSE:
                self._to_close = True

    def _exit_signal(self, *args, **kwargs):
        self._to_close = True
