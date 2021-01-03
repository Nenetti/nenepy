import time

from nenepy.utils.multi.multi_task_process import MultiTaskProcess


class MultiTaskProcessManager:

    def __init__(self, target_cls, args=(), kwargs={}, n_processes=1, auto_start=False):
        """

        Args:
            target_cls (MultiTaskProcess cls):
            args:
            kwargs:
            daemon:

        """
        self._n_processes = n_processes
        self._index = 0

        self._processes = [None] * n_processes
        for i in range(n_processes):
            self._processes[i] = target_cls(*args, **kwargs)
            assert isinstance(self._processes[i], MultiTaskProcess)

        self._target_process = self._processes[0]

        if auto_start:
            self.start()

    # ==================================================================================================
    #
    #   Pubic function
    #
    # ==================================================================================================
    def start(self):
        for process in self._processes:
            process.start()

    def add_task(self, *args):
        self._target_process.add_task(*args)
        self._change_next_process()

    def close(self):
        for process in self._processes:
            process.close()

    def wait_process_completed(self):
        while not self.is_processes_completed():
            time.sleep(0.1)

    def is_processes_completed(self):
        for process in self._processes:
            if not process.is_process_completed():
                return False

        return True

    def close_with_waiting(self):
        for process in self._processes:
            process.close_with_waiting()

    def kill(self):
        for process in self._processes:
            process.close()

    # ==================================================================================================
    #
    #   Private function
    #
    # ==================================================================================================
    def _change_next_process(self):
        if self._index != self._n_processes - 1:
            self._index += 1
        else:
            self._index = 0

        self._target_process = self._processes[self._index]
