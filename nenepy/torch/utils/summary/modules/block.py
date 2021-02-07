class Block:

    def __init__(self):
        self.module = None
        self.child_blocks = []
        self.processing_time = 0
        self.depth = 0
        self.is_root = False
        self.is_bottom = False
        self.is_last_module_in_sequential = False
        self.architecture = None

    def has_children(self):
        return len(self.child_blocks) > 0

    # ==================================================================================================
    #
    #   Static Method
    #
    # ==================================================================================================
    @staticmethod
    def calc_depth(roots):
        """
        Args:
            roots (list[Block]):

        """

        def recursive(block, depth):
            block.depth = depth
            if len(block.child_blocks) > 0:
                block.bottom = False
                for b in block.child_blocks:
                    recursive(b, depth + 1)
                block.child_blocks[-1].is_last_module_in_sequential = True
            else:
                block.bottom = True

        for root in roots:
            recursive(root, 0)
            root.is_root = True
