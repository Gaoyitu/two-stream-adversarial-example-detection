
import tensorflow as tf
import utils

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(Conv2D, self).__init__()
        self.conv_op = tf.keras.layers.Conv2D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     use_bias=False,
                                     kernel_initializer='he_normal')
    def get_config(self):
        base_config = super(Conv2D,self).get_config()

        return dict(list(base_config.items()))


    def call(self, inputs, **kwargs):
        return self.conv_op(inputs)


class UpConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=4, strides=2, padding='same'):
        super(UpConv2D, self).__init__()
        self.up_conv_op = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                 kernel_size=kernel_size,
                                                 strides=strides,
                                                 padding=padding,
                                                 use_bias=False,
                                                 kernel_initializer='he_normal')

    def get_config(self):
        base_config = super(UpConv2D, self).get_config()
        return dict(list(base_config.items()))

    def call(self, inputs, **kwargs):
        return self.up_conv_op(inputs)


class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1, momentum=0.99):
        super(BatchNorm, self).__init__()
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=epsilon,
                                                    axis=axis,
                                                    momentum=momentum)

    def get_config(self):
        base_config = super(BatchNorm, self).get_config()

        return dict(list(base_config.items()))
    def call(self, inputs, **kwargs):
        return self.batch_norm(inputs)


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, axis=-1):
        super(LayerNorm, self).__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon, axis=axis)

    def get_config(self):
        base_config = super(LayerNorm, self).get_config()
        return dict(list(base_config.items()))

    def call(self, inputs, **kwargs):
        return self.layer_norm(inputs)


class LeakyRelu(tf.keras.layers.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=alpha)

    def get_config(self):
        base_config = super(LeakyRelu, self).get_config()
        return dict(list(base_config.items()))

    def call(self, inputs, **kwargs):
        return self.leaky_relu(inputs)


class AdamOptWrapper(tf.keras.optimizers.Adam):
    def __init__(self,
                 learning_rate=1e-4,
                 beta_1=0.5,
                 beta_2=0.999,
                 epsilon=1e-4,
                 amsgrad=False,
                 **kwargs):
        super(AdamOptWrapper, self).__init__(learning_rate, beta_1, beta_2, epsilon,
                                             amsgrad, **kwargs)


def d_loss_fn(f_logit, r_logit):
    f_loss = tf.reduce_mean(f_logit)
    r_loss = tf.reduce_mean(r_logit)
    return f_loss - r_loss


def g_loss_fn(f_logit):
    f_loss = -tf.reduce_mean(f_logit)
    return f_loss

def MSE(real_image, fake_image):
    loss = tf.reduce_mean(tf.square(real_image - fake_image))
    return loss

def get_pre_output(Gs, initial_noise, images, fix=True):
    generator_num = len(Gs)
    pre_output = initial_noise

    if fix == True:
        for i in range(generator_num):
            shape = images[i].shape
            generator = Gs[i]
            if i == 0:
                output = generator(pre_output, training=False)
                pre_output = output
            else:
                upsampling_out = utils.image_resize(pre_output, shape[1], shape[2])
                output = generator(upsampling_out, training=False)
                pre_output = output + upsampling_out
        return pre_output
    if fix == False:
        for i in range(generator_num):
            shape = images[i].shape
            generator = Gs[i]
            if i == 0:
                output = generator(pre_output, training=False)
                pre_output = output
            else:
                noise = tf.random.uniform(shape)
                upsampling_out = utils.image_resize(pre_output, shape[1], shape[2])
                output = generator(upsampling_out + 0.1 * noise, training=False)
                pre_output = output + upsampling_out
        return pre_output

def gradient_penalty(batch_size,f, real, fake):
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    diff = fake - real
    inter = real + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = f(inter)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp
