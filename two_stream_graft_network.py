
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from functools import partial
import tensorflow as tf
import numpy as np
from basics import d_loss_fn,AdamOptWrapper
from compact_bilinear_pooling import compact_bilinear_pooling_layer

class DetectNoise:
    def __init__(self,inputH,inputW,channel,epochs,batch_size,feature_extraction_net):
        self.inputH = inputH
        self.inputW = inputW
        self.channel = channel
        self.epochs = epochs
        self.batch_size = batch_size
        self.opt = AdamOptWrapper(learning_rate=0.0001, beta_1=0.5)
        self.SRM = self.SRM_network()
        self.Feature_extraction_net = feature_extraction_net
        self.Prediction_net = self.Prediction_network()

        self.SRM.summary()
        self.Feature_extraction_net.summary()
        self.Prediction_net.summary()


    def train(self,x_original,x_adv):

        x_original = tf.data.Dataset.from_tensor_slices(x_original)
        x_original = x_original.batch(self.batch_size)

        x_adv = tf.data.Dataset.from_tensor_slices(x_adv)
        x_adv = x_adv.batch(self.batch_size)

        train_loss = tf.keras.metrics.Mean()

        for epoch in range(self.epochs):
            for x_o_batch, x_n_batch in zip(x_original,x_adv):

                # self.train_step(x_o_batch,x_n_batch)
                cost = self.train_step(x_o_batch,x_n_batch)
                train_loss(cost)

                print(epoch)
                print('loss:',cost.numpy())
                train_loss.reset_states()

    @tf.function
    def train_step(self, x_o,x_a):
        y = tf.concat([np.ones((x_o.shape[0],1),dtype=int),np.zeros((x_a.shape[0],1),dtype=int)],axis=0)
        with tf.GradientTape() as t:
            x_input = tf.concat([x_o, x_a], 0)
            x_srm = self.SRM(x_input,training = False)
            outputs = self.Prediction_net([x_input,x_srm],training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y,outputs))

            loss_regularization = []
            for p in self.Prediction_net.trainable_variables:
                loss_regularization.append(tf.nn.l2_loss(p))
            loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
            cost = loss + 0.0005* loss_regularization

        grad = t.gradient(cost, self.Prediction_net.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.Prediction_net.trainable_variables))
        return cost


    def SRM_network(self):
        def truncate_2(x):
            neg = ((x + 2) + abs(x + 2)) / 2 - 2
            return -(2 - neg + abs(2 - neg)) / 2 + 2

        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
        filters = np.einsum('klij->ijlk', filters)
        filters = filters.flatten()
        initializer_srm = tf.constant_initializer(filters)

        inputs = tf.keras.Input(shape=(self.inputH, self.inputW, self.channel))

        conv = tf.keras.layers.Conv2D(filters=3, kernel_size=5, strides=1, padding='same',
                                      kernel_initializer=initializer_srm, use_bias=False)(inputs)

        outputs = truncate_2(conv)

        model = tf.keras.Model(inputs=inputs, outputs=outputs,name = 'SRM_net')
        return model


    def Prediction_network(self):
        RGB_inputs = tf.keras.Input(shape=(self.inputH, self.inputW, self.channel))
        Noise_inputs = tf.keras.Input(shape=(self.inputH, self.inputW, self.channel))

        RGB_outputs = self.Feature_extraction_net(RGB_inputs,training=False)
        Noise_outputs = self.Feature_extraction_net(Noise_inputs,training=False)
        self.Feature_extraction_net.trainable = False

        add_RGBandNoise = RGB_outputs+Noise_outputs

        outputs = tf.keras.layers.Dense(2,activation="softmax")(add_RGBandNoise)

        model = tf.keras.Model(inputs=[RGB_inputs,Noise_inputs],outputs=outputs,name = 'prediction_net')
        return model








