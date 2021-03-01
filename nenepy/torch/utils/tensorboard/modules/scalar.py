class Scalar:

    def __init__(self, value):
        """

        Args:
            value (ScalarEvent):

        """
        self.value = value.value
        self.step = value.step
        self.wall_time = value.wall_time
