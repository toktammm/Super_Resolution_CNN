import numpy as np
import cv2
from modules import ConvLayer, Relu

# upscaling or downscaling by bicubic interpolation
def imgresize(input_img, UP=False):  # Downsampling if UP=False, otherwise Upsamble
    if (UP == True):
        resized_img = cv2.resize(input_img, (input_img.shape[0] + 1, input_img.shape[1] + 1),
                                 interpolation=cv2.INTER_CUBIC)
    else:
        resized_img = cv2.resize(input_img, (input_img.shape[0] - 1, input_img.shape[1] - 1),
                                 interpolation=cv2.INTER_CUBIC)
    return resized_img


def main():
    image_name = "image_0729.jpg"
    image_gt = cv2.imread(image_name, cv2.IMREAD_COLOR)

    n_epoch = 3
    lr = 0.001  # initial learning rate
    kernel_size = 3
    n_output = 7  # number of image pairs for training pairs (to prepare downsampled images)

    # the desired dimension to have for upsampling and later add pixels in training
    desired_dim = input("\nEnter the width of the desired (square) image ")
    if int(desired_dim) < image_gt.shape[0]:
        print("ERROR: The desired dimension should be larger than {} for SUPER RESOLUTION").format(image_gt.shape[0])
        desired_dim = input("Enter the width of the desired (square) image ")

    print("Dataset preparation")
    # Downsampling with Inter Cubic Interpolation:
    down_data = []
    input_img = image_gt
    image_gt = image_gt.astype(np.float64) / 1.  
    for i in range(n_output):
        input_img = imgresize(input_img)
        down_data.append(input_img.astype(np.float64) / 1.)

    # Upsampling with Inter Cubic Interpolation:
    up_data = []
    for i in range(n_output):
        up_data.append(imgresize(down_data[i], UP=True))

    ### Convolutional Layers ###
    n_channels = 64
    Conv_layer1 = ConvLayer(3, n_channels, kernel_size)
    Conv_layer2 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer3 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer4 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer5 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer6 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer7 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer8 = ConvLayer(n_channels, n_channels, kernel_size)
    Conv_layer9 = ConvLayer(n_channels, 3, kernel_size)
    relu = Relu()

    print("\n############## Training ##############")
    ############## Training ##############
    # train over 7 pairs of images

    # training: looping over epochs and the 7 pair of images
    for n in range(n_epoch):
        for i in range(n_output):

            ### Convolution forward ###
            Input = up_data[i]
            if i == 0:
                output_gt = image_gt
            else:
                output_gt = down_data[i - 1]

            ### forward ###
            Conv1_out = Conv_layer1.forward(Input)
            relu_out1 = relu.forward(Conv1_out)
            Conv2_out = Conv_layer2.forward(relu_out1)
            relu_out2 = relu.forward(Conv2_out)
            Conv3_out = Conv_layer3.forward(relu_out2)
            relu_out3 = relu.forward(Conv3_out)
            Conv4_out = Conv_layer4.forward(relu_out3)
            relu_out4 = relu.forward(Conv4_out)
            Conv5_out = Conv_layer5.forward(relu_out4)
            relu_out5 = relu.forward(Conv5_out)
            Conv6_out = Conv_layer6.forward(relu_out5)
            relu_out6 = relu.forward(Conv6_out)
            Conv7_out = Conv_layer7.forward(relu_out6)
            relu_out7 = relu.forward(Conv7_out)
            Conv8_out = Conv_layer8.forward(relu_out7)
            relu_out8 = relu.forward(Conv8_out)
            Conv9_out = Conv_layer9.forward(relu_out8)

            ### calculating cost ###
            G_H = output_gt - Input
            J_H = Conv9_out  # J and H refer to input and output images

            # MSE
            mse_loss = (np.square(J_H - G_H)).mean(axis=None)
            print("epoch ", n + 1, "iteration ", i + 1, "loss ", mse_loss)
            filename = "out_{}_{}{}".format(n + 1, i + 1, image_name[
                                      -8:])  # the last 4 character of the image_name refer to the format (jpg or png)
            cv2.imwrite(filename, ((Conv9_out + up_data[i]) * 1.))

            ### MSE derivative ###
            w, h, c = J_H.shape
            error_mse = 2 * (J_H - G_H) / (w * h * c)

            ### Back propagation ###
            conv_back9 = Conv_layer9.backward(error_mse)
            relu_back8 = relu.backward(conv_back9)
            conv_back8 = Conv_layer8.backward(relu_back8)
            relu_back7 = relu.backward(conv_back8)
            conv_back7 = Conv_layer7.backward(relu_back7)
            relu_back6 = relu.backward(conv_back7)
            conv_back6 = Conv_layer6.backward(relu_back6)
            relu_back5 = relu.backward(conv_back6)
            conv_back5 = Conv_layer5.backward(relu_back5)
            relu_back4 = relu.backward(conv_back5)
            conv_back4 = Conv_layer4.backward(relu_back4)
            relu_back3 = relu.backward(conv_back4)
            conv_back3 = Conv_layer3.backward(relu_back3)
            relu_back2 = relu.backward(conv_back3)
            conv_back2 = Conv_layer2.backward(relu_back2)
            relu_back1 = relu.backward(conv_back2)
            conv_back1 = Conv_layer1.backward(relu_back1)

            ### Gradient Descent ###
            Conv_layer9.w -= lr * Conv_layer9.dw
            Conv_layer9.b -= lr * Conv_layer9.db
            Conv_layer8.w -= lr * Conv_layer8.dw
            Conv_layer8.b -= lr * Conv_layer8.db
            Conv_layer7.w -= lr * Conv_layer7.dw
            Conv_layer7.b -= lr * Conv_layer7.db
            Conv_layer6.w -= lr * Conv_layer6.dw
            Conv_layer6.b -= lr * Conv_layer6.db
            Conv_layer5.w -= lr * Conv_layer5.dw
            Conv_layer5.b -= lr * Conv_layer5.db
            Conv_layer4.w -= lr * Conv_layer4.dw
            Conv_layer4.b -= lr * Conv_layer4.db
            Conv_layer3.w -= lr * Conv_layer3.dw
            Conv_layer3.b -= lr * Conv_layer3.db
            Conv_layer2.w -= lr * Conv_layer2.dw
            Conv_layer2.b -= lr * Conv_layer2.db
            Conv_layer1.w -= lr * Conv_layer1.dw
            Conv_layer1.b -= lr * Conv_layer1.db

            # changing step size after each iteration
            lr *= 0.9

        # changing step size after each epoch
        lr *= 0.1

    print("\n############## Test ##############")
    ############## Test ##############
    for i in range(int(desired_dim) - image_gt.shape[0]):
        print("test iteration ", i+1)
        if i == 0:
            up_img = image_gt

        up_img = imgresize(up_img, UP=True)

        ### forward ###
        Conv1_out = Conv_layer1.forward(up_img)
        relu_out1 = relu.forward(Conv1_out)
        Conv2_out = Conv_layer2.forward(relu_out1)
        relu_out2 = relu.forward(Conv2_out)
        Conv3_out = Conv_layer3.forward(relu_out2)
        relu_out3 = relu.forward(Conv3_out)
        Conv4_out = Conv_layer4.forward(relu_out3)
        relu_out4 = relu.forward(Conv4_out)
        Conv5_out = Conv_layer5.forward(relu_out4)
        relu_out5 = relu.forward(Conv5_out)
        Conv6_out = Conv_layer6.forward(relu_out5)
        relu_out6 = relu.forward(Conv6_out)
        Conv7_out = Conv_layer7.forward(relu_out6)
        relu_out7 = relu.forward(Conv7_out)
        Conv8_out = Conv_layer8.forward(relu_out7)
        relu_out8 = relu.forward(Conv8_out)
        Conv9_out = Conv_layer9.forward(relu_out8)

        up_img = (Conv9_out + up_img) * 1.

        # output with the desired dimensions
        filename = "super_resolution_{}_{}".format(i+1,
            image_name[-8:])  # last 4 character of the image_name are for format
        cv2.imwrite(filename, up_img)


if __name__ == '__main__':
    main()
