class Pooling:

    def __init__(self, stride_shape, pooling_shape):

        super().__init__()

        self.pooling_output_width = None
        self.pooling_output_height = None
        self.hi_m = None
        self.wi_m = None
        self.trainable = False
        self.is1dConv = 0

        self.input_tensor = None
        self.error_tensor = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass


