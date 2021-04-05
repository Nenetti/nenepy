import os
import signal
import sys
import time
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from multiprocessing import Pipe
from multiprocessing.context import Process
from multiprocessing import Queue

from threading import Thread


class Status(Enum):
    IDLING = auto()
    PROCESSING = auto()
    CLOSED = auto()


class Request(Enum):
    STATUS = auto()
    CLOSE = auto()
    KILL = auto()


class MultiProcess(Process, metaclass=ABCMeta):

    def __init__(self, args=(), kwargs={}, is_daemon=True):
        """

        Args:
            args:
            kwargs:
            is_daemon:

        """
        super(MultiProcess, self).__init__(target=self.process, args=args, kwargs=kwargs, daemon=is_daemon)
        self._queue = Queue()
        self._main_process_connection, self._subprocess_connection = Pipe()
        self._status = Status.IDLING
        self._needs_kill = False
        self._needs_close = False

    # ==================================================================================================
    #
    #   Property
    #
    # ==================================================================================================
    @property
    def qsize(self):
        return self._queue.qsize()

    @property
    def status(self):
        if self.is_closed:
            return Status.CLOSED
        else:
            self._main_process_connection.send(Request.STATUS)
            return self._main_process_connection.recv()

    @property
    def needs_process(self):
        return self.qsize > 0

    @property
    def is_closed(self):
        return self._closed

    @property
    def is_process_completed(self):
        return (self.qsize == 0) and (self.status == Status.IDLING or self.status == Status.CLOSED)

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def add_task(self, *args):
        if (not self.is_closed) and (not self._needs_kill) and (not self._needs_kill):
            self._queue.put(args)
        else:
            raise ValueError("process object is closed")

    def close_with_waiting(self):
        self.close()
        if not self.is_closed:
            self.join()
            super(MultiProcess, self).close()

    def kill_with_waiting(self):
        self.kill()
        if not self.is_closed:
            self.join()
            super(MultiProcess, self).close()

    def close(self):
        if (not self.is_closed) and (not self._main_process_connection.closed):
            self._main_process_connection.send(Request.CLOSE)
            self._main_process_connection.close()

    def kill(self):
        if (not self.is_closed) and (not self._main_process_connection.closed):
            self._main_process_connection.send(Request.KILL)
            self._main_process_connection.close()

    # ==================================================================================================
    #
    #   Sub Process function
    #
    # ==================================================================================================
    def on_start(self):
        pass

    def on_exit(self):
        pass

    @abstractmethod
    def process(self, *task):
        raise NotImplementedError()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.on_start()
        thread = Thread(target=self._status_response_thread, daemon=False)
        thread.start()

        while True:
            if self.qsize > 0:
                task = self._queue.get()
                self.process(*task)

            if (self._needs_close and (self.qsize == 0)) or self._needs_kill:
                while self.qsize > 0:
                    self._queue.get()
                break

            time.sleep(0.01)

        self.on_exit()
        if thread.is_alive():
            thread.join()

        if self._needs_kill:
            print(f"PID {os.getpid()}: {self.__class__.__name__} ID({id(self)}) killed")
        else:
            print(f"PID {os.getpid()}: {self.__class__.__name__} ID({id(self)}) closed")

    def _status_response_thread(self):
        while True:
            request = self._subprocess_connection.recv()

            if request is Request.STATUS:
                status = Status.PROCESSING if self.needs_process else Status.IDLING
                self._subprocess_connection.send(status)

            elif request is Request.CLOSE:
                self._needs_close = True
                break

            elif request is Request.KILL:
                self._needs_kill = True
                break
