import time


class Timer:
    """

    Measure processing time.

    Examples:
        >>> timer = Timer()
        >>> timer.start()

        target code...

        >>> timer.start()
        >>> print(timer.elapsed_time)
        0.123...

        >>> print(timer.formatted_elapsed_time())
        00:01:23

    """

    def __init__(self):
        self._start_time = time.time()
        self._elapsed_time = 0

    # ==================================================================================================
    #
    #   Public Method
    #
    # ==================================================================================================

    @property
    def start_time(self):
        return self._start_time

    @property
    def elapsed_time(self):
        return self._elapsed_time

    def start(self):
        self._start_time = time.time()

    def stop(self):
        self._elapsed_time = time.time() - self._start_time

    def formatted_elapsed_time(self):
        seconds = int(self._elapsed_time % 60)
        minutes = int((self._elapsed_time % 3600) // 60)
        hours = int(self._elapsed_time // 3600)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        print(self.elapsed_time)

    def __str__(self):
        return f"{time.time() - self._start_time}"
