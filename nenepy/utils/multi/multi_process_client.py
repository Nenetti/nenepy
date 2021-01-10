from enum import Enum, auto
from multiprocessing.connection import PipeConnection


class Type(Enum):
    IDLING = auto()
    PROCESSING = auto()
    CLOSED = auto()


class MultiProcessClient:

    def __init__(self, connection):
        """

        Args:
            connection (_ConnectionBase):

        """
        self._connection = connection

    def get_server_status(self):
        self._connection.send("")
        return self._connection.recv()
