import time

from nenepy.utils.multi.multi_process import MultiProcess


class MultiProcessManager:

    def __init__(self, target_cls, args=(), kwargs={}, n_processes=1, is_auto_starting=False):
        """

        Args:
            target_cls (MultiTaskProcess cls):
            args:
            kwargs:

        """
        self._n_processes = n_processes
        self._index = 0

        self._processes = [None] * n_processes
        for i in range(n_processes):
            process = target_cls(*args, **kwargs)
            if not isinstance(process, MultiProcess):
                raise TypeError(f"The type must be 'MultiProcess', but given '{type(process)}'")
            self._processes[i] = process

        self._target_process = self._processes[0]

        if is_auto_starting:
            self.start()

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def start(self):
        for process in self._processes:
            process.start()

    def add_task(self, *args):
        if not self._target_process.is_closed:
            self._target_process.add_task(*args)
            self._change_next_process()

    def wait_process_completed(self):
        while not self.is_processes_completed():
            time.sleep(0.1)

    def is_processes_completed(self):
        for process in self._processes:
            if not process.is_process_completed():
                return False

        return True

    def close(self):
        for process in self._processes:
            process.close()

    def kill(self):
        for process in self._processes:
            process.kill()

    def close_with_waiting(self):
        for process in self._processes:
            process.close_with_waiting()

    def kill_with_waiting(self):
        for process in self._processes:
            process.kill_with_waiting()

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    def _change_next_process(self):
        if self._index != self._n_processes - 1:
            self._index += 1
        else:
            self._index = 0

        self._target_process = self._processes[self._index]
