import tensorflow as tf


def get_mnist_local():

    inputs = tf.keras.Input(shape=(28, 28, 1))
    conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                                    padding='same', activation="relu", name="conv_1")(inputs)
    max_pooling_1 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu", name="conv_2")(max_pooling_1)
    max_pooling_2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding="same")(conv_2)

    max_pooling_2_flat = tf.keras.layers.Flatten()(max_pooling_2)

    fc_1 = tf.keras.layers.Dense(200, activation="relu",name='feature_layer')(max_pooling_2_flat)

    outputs = tf.keras.layers.Dense(10, activation=None)(fc_1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights("model/mnist/classifier/cnn_local_weights.h5")
    model.summary()

    return model

def get_mnist_black():

    model = tf.keras.models.load_model('model/mnist/classifier/cnn.h5')
    model.summary()

    return model


def get_cifar10_local():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    conv_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    activation="relu",
                                    name="conv_1",
                                    kernel_initializer='glorot_uniform')(inputs)
    conv_2 = tf.keras.layers.Conv2D(64, (3, 3),
                                    padding='same',
                                    activation="relu",
                                    name="conv_2",
                                    kernel_initializer='glorot_uniform')(conv_1)
    max_pooling_1 = tf.keras.layers.MaxPool2D((2, 2), (2, 2),
                                              padding="same",name="pool1")(conv_2)
    conv_3 = tf.keras.layers.Conv2D(128, (3, 3),
                                    padding='same',
                                    activation="relu",
                                    name="conv_3",
                                    kernel_initializer='glorot_uniform')(max_pooling_1)
    conv_4 = tf.keras.layers.Conv2D(128, (3, 3),
                                    padding='same',
                                    activation="relu",
                                    name="conv_4",
                                    kernel_initializer='glorot_uniform')(conv_3)
    max_pooling_2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2),
                                              padding="same",name="pool2")(conv_4)

    max_pooling_2_flat = tf.keras.layers.Flatten()(max_pooling_2)

    fc_1 = tf.keras.layers.Dense(256,
                                 activation="relu",
                                 kernel_initializer='he_normal')(max_pooling_2_flat)

    fc_2 = tf.keras.layers.Dense(256,
                                 activation="relu",
                                 kernel_initializer='he_normal',name = 'fc_2')(fc_1)

    outputs = tf.keras.layers.Dense(10,
                                    activation='softmax')(fc_2)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.load_weights('model/cifar10/classifier/cifar10_local_weights.h5')
    return model

def get_cifar10_vgg16():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    # conv_base.trainable = False


    model.summary()
    model.load_weights("model/cifar10/classifier/vgg16_weights.h5")
    return model
def get_cifar10_mobilenet():
    conv_base = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    # conv_base.trainable = False

    model.summary()

    model.load_weights("model/cifar10/classifier/mobileNet_weights.h5")
    return model
def get_cifar10_resnet50():
    conv_base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    # conv_base.trainable = False

    model.summary()

    model.load_weights("model/cifar10/classifier/resNet50_weights.h5")
    return model


def get_cifar100_densenet201():
    conv_base = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation="softmax"))

    model.summary()
    model.load_weights("model/cifar100/classifier/densenet201_weights.h5")

    return model

def get_cifar100_densenet169():

    conv_base = tf.keras.applications.DenseNet169(weights='imagenet',include_top = False, input_shape=(32,32,3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation="softmax"))
    # conv_base.trainable = False

    model.summary()
    model.load_weights("model/cifar100/classifier/densenet169_weights.h5")

    return model
def get_cifar100_resnet152():
    conv_base = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation="softmax"))
    # conv_base.trainable = False

    model.summary()
    model.load_weights("model/cifar100/classifier/resnet152_weights.h5")
    return model
def get_cifar100_resnet101v2():
    conv_base = tf.keras.applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.models.Sequential()

    model.add(conv_base)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation="softmax"))
    # conv_base.trainable = False

    model.summary()
    model.load_weights("model/cifar100/classifier/resnet101V2_weights.h5")
    return model













