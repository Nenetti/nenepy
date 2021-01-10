import multiprocessing
import threading
import time
from multiprocessing.connection import Pipe
from multiprocessing.context import Process
from multiprocessing.queues import Queue as Q
from multiprocessing.process import active_children

from queue import Queue

from threading import Thread


class MultiProcessManager:

    def __init__(self, n_processes=1):
        self._queue = Queue()
        self._processes = [None] * n_processes
        self._n_available_processes = n_processes
        self._n_max_processes = n_processes

        Thread(target=self._loop_thread, daemon=True).start()

    def add_task(self, func, args, callback=None):
        task = (func, args, callback)
        self._queue.put(task)

    def wait_process_completed(self):
        while not self.is_all_process_completed():
            time.sleep(0.01)

    def is_all_process_completed(self):
        return (self._n_available_processes == self._n_max_processes) and (self._queue.qsize() == 0)

    def _loop_thread(self):
        while True:
            for i, process in enumerate(self._processes):
                if self._is_process_completed(process):
                    self._processes[i] = None
                    self._n_available_processes += 1

                if self._is_process_available(i) and (self._queue.qsize() > 0):
                    task = self._queue.get()
                    self._processes[i] = self._execute_task(*task)
                    self._n_available_processes -= 1

            time.sleep(0.01)

    @staticmethod
    def _is_process_completed(process):
        if process is None:
            return False

        if process.is_alive():
            return False

        return True

    def _is_process_available(self, index):
        if self._processes[index] is None:
            return True

        return False

    @staticmethod
    def _execute_task(*thread_args):

        def generate_process(target_func, target_args, callback):
            def process_func(func, args, connection):
                output = func(*args)
                connection.send(output)

            c1, c2 = Pipe()
            process = Process(target=process_func, args=(target_func, target_args, c2), daemon=True)
            process.start()
            result = c1.recv()
            c2.close()
            c1.close()
            if callback is not None:
                callback(result)

        thread = Thread(target=generate_process, args=thread_args, daemon=True)
        thread.start()
        return thread
