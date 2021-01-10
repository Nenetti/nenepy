from threading import Thread


class MultiThread(Thread):

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, callback=None):
        super(MultiThread, self).__init__(target=target, args=args, kwargs=kwargs, daemon=daemon)
        self._callback = callback

    def run(self):
        super(MultiThread, self).run()
        self._callback()
