
# -*- coding:utf-8 -*-
import tensorflow as tf

l1_l2 = tf.keras.regularizers.L1L2(0.00001, 0.000001)
l1 = tf.keras.regularizers.l1(0.00001)

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def attention_residual_block(input, dilation=1, filters=256):
    # Depth wise는 GPU 메로리가 일반 conv보다 많이 필요하기때문에, 쓰는것을 보류
    # 1x1 conv로 해결점을 보자
    h = input
    x, y = tf.image.image_gradients(input)
    h_attenion_layer = tf.add(tf.abs(x), tf.abs(y))
    h_attenion_layer = tf.reduce_mean(h_attenion_layer, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[dilation,dilation],[0,0],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[0,0],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid",
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)

    return (h*h_attenion_layer) + input

def decode_residual_block(input, dilation=1, filters=256):

    h = input

    x, y = tf.image.image_gradients(input)
    h_attenion_layer = tf.add(tf.abs(x), tf.abs(y))
    h_attenion_layer = tf.reduce_mean(h_attenion_layer, axis=-1, keepdims=True)
    h_attenion_layer = tf.nn.sigmoid(h_attenion_layer)    # attenion map !

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="valid",
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(h, [[0,0],[dilation,dilation],[0,0],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.pad(h, [[0,0],[0,0],[dilation,dilation],[0,0]], mode='REFLECT', constant_values=0)
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="valid",
                                kernel_regularizer=l1_l2, activity_regularizer=l1, dilation_rate=dilation)(h)
    h = InstanceNormalization()(h)
    
    return (h*h_attenion_layer) + input

def sigmoid_4x(x):
    return 1/(1+tf.math.exp(-4*x))

def F2M_generator(input_shape=(256, 256, 3)):   # Need to fix parameters!! now!!!!

    h = inputs = tf.keras.Input(input_shape)
    first_grad_x, first_grad_y = tf.image.image_gradients(h)
    first_grad = tf.add(tf.abs(first_grad_x), tf.abs(first_grad_y))
    first_grad = tf.expand_dims(sigmoid_4x(tf.reduce_mean(first_grad, -1)), 3)

    #second_grad_x, second_grad_y = tf.image.image_gradients(tf.image.resize(h, [128, 128]))
    #second_grad = tf.add(tf.abs(second_grad_x), tf.abs(second_grad_y))
    #second_grad = tf.expand_dims(tf.nn.sigmoid(tf.reduce_mean(second_grad, -1)), 3)
    second_grad = tf.image.resize(first_grad, [128, 128])

    #third_grad_x, thrid_grad_y = tf.image.image_gradients(tf.image.resize(h, [64, 64]))
    #third_grad = tf.add(tf.abs(third_grad_x), tf.abs(thrid_grad_y))
    #third_grad = tf.expand_dims(tf.nn.sigmoid(tf.reduce_mean(third_grad, -1)), 3)
    third_grad = tf.image.resize(first_grad, [64, 64])

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=7,
                               kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_1")(h)
    h = InstanceNormalization()(h)  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1, 
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_2")(h)
    h = InstanceNormalization()(tf.multiply(h, first_grad))  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)
    
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_3")(h)
    h = InstanceNormalization()(h)  # [128, 128, 128]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_4")(h)
    h = InstanceNormalization()(tf.multiply(h, second_grad))  # [128, 128, 128]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_5")(h)
    h = InstanceNormalization()(h)  # [64, 64, 256]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.DepthwiseConv2D(kernel_size=1,
                                        depthwise_regularizer=l1_l2, activity_regularizer=l1, name="conv_en_6")(h)
    h = InstanceNormalization()(tf.multiply(h, third_grad))  # [64, 64, 256]

    for i in range(1, 7):
        h = attention_residual_block(h, dilation=i * 3, filters=256)

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same",
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)  # [128, 128, 128]

    for i in range(1, 2):
        h = decode_residual_block(h, dilation=4, filters=128)

    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",
                                        kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)  # [256, 256, 64]
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding="valid",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3*16, kernel_size=7, strides=1, padding="valid")(h)
    #h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def F2M_discriminator(input_shape=(256, 256, 3)):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=1, padding="same",
                               kernel_regularizer=l1_l2, activity_regularizer=l1)(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

#img = tf.io.read_file("C:/Users/Yuhwan/Pictures/Lenna.png")
#img = tf.image.decode_png(img, 4) 
#img = tf.image.resize(img, [512, 512])
#img = tf.unstack(img, axis=-1)
#r, g, b = img[0], img[1], img[2]

#a = tf.ones_like(r) * 1.0

#img = tf.stack([r / 127.5 - 1., g / 127.5 - 1., b / 127.5 - 1., a], axis=-1) 

#import matplotlib.pyplot as plt
#plt.imsave("C:/Users/Yuhwan/Pictures/36528_f.jpg", img * 0.5 + 0.5)
