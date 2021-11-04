import numpy as np

# Relu Class
class Relu:

    # Activation function forward
    def forward(self, r):
        return np.maximum(r, 0)

    # Derivative of Relu (backward)
    def backward(self, r):
        return 1. * (r > 0)


# Convolution Layer Class
class ConvLayer:
    def __init__(self, input_channel, output_channel, kernel_size):

        # (width, length): filter dimensions(3,3)
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size

        # kernels
        self.w = np.zeros((self.output_channel, self.kernel_size, self.kernel_size, self.input_channel))

        # kernels' initilization according to He initialization
        for i in range(self.output_channel):
            self.w[i] = (np.random.randn(self.kernel_size,
                                         self.kernel_size, self.input_channel) / np.sqrt(
                self.input_channel * self.kernel_size * self.kernel_size / 2.))

        # bias initilization according to He initialization
        self.b = np.zeros(self.output_channel)
        for i in range(self.output_channel):
            self.b[i] = (np.random.randn(1)) / np.sqrt(self.input_channel*self.kernel_size*self.kernel_size / 2.)

    def padding(self, X):  # pad image with zeros before corrolating the image
        X = np.pad(X, ((1, 1), (1, 1), (0, 0)), mode='constant')
        return X

    def forward(self, X):
        self.Input = X
        pad_img = self.padding(X)

        # updating the output size for image after convolution (it should remain the same)
        self.img_conv_width = pad_img.shape[0] - len(self.w[0]) + 1

        conv_X = np.zeros(
            (self.img_conv_width, self.img_conv_width, self.output_channel))  # self.output_channel = len(self.w)


        # convolution
        for i in range(self.output_channel):
            for j in range(self.img_conv_width):
                for k in range(self.img_conv_width):
                    conv_X[j, k, i] = np.sum(pad_img[j:j + self.kernel_size, k:k + self.kernel_size, :] * self.w[i]) + \
                                      self.b[i]

        return conv_X

    def backward(self, error):  # back propagation
        X = self.Input
        # derivative of the kernels
        self.dw = np.zeros((self.output_channel, self.kernel_size, self.kernel_size, self.input_channel))
        self.db = np.zeros(self.output_channel)
        self.d_input = np.zeros((X.shape[0] + 2, X.shape[1] + 2, self.input_channel))  # added 2s are for padding size
        pad_img = self.padding(X)

        for i in range(0, self.output_channel):
            self.dw[i] = (np.zeros((self.kernel_size, self.kernel_size, self.input_channel)))
            self.db[i] = (np.zeros(1))

        for j in range(0, self.img_conv_width):
            for k in range(0, self.img_conv_width):
                for i in range(0, self.output_channel):
                    self.dw[i] += error[j, k, i] * pad_img[j:j + self.kernel_size, k:k + self.kernel_size, :]
                    self.d_input[j:j + self.kernel_size, k:k + self.kernel_size, :] += error[j, k, i] * self.w[i]

        for i in range(0, self.output_channel):
            self.db = np.sum(error[i])

        return self.d_input
