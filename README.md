# Implementation of CNN from scratch for the task of Super Resolution

In this task, an image is gradually transformed into a higher resolution version of itself. First, we prepare training data downscaling the image using bicubic interpolation. Then CNN uses this data for upscaling the image up to a desired resolution.

The CNN consists of 8 hidden convolutional layers each with 64 channels. The kernels are all 3x3 with padding of 1 to preserve the size of the original input. In other words, the input and output images of the CNN have the same size. To account for the upscaling that the network is supposed to learn, the input image is first upsampled by bicubic interpolation to get to the desired size. The network is then trained to adjust the difference between the input and output images.
