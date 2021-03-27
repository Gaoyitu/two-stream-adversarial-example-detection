
import tensorflow as tf
import graft_network
import two_stream_graft_network
import two_stream_detection_network
import single_stream_detection_network

from models import get_cifar100_resnet101v2
import numpy as np


def get_training_data(x,x_adv,y,model):
    preds = model.predict(x)
    y1 = np.reshape(y,(y.shape[0],))
    inds_correct = np.where(preds.argmax(axis=1) == y1)[0]
    print(inds_correct.shape)
    x_adv = x_adv[inds_correct]
    y = y[inds_correct]
    print(x_adv.shape)
    y1= np.reshape(y,(y.shape[0],))
    x = x[inds_correct]
    preds_adv = model.predict(x_adv)
    inds_correct = np.where(preds_adv.argmax(axis=1) == y1)[0]

    print(inds_correct.shape)
    x_adv = np.delete(x_adv,inds_correct,axis=0)
    x = np.delete(x,inds_correct,axis=0)
    y = np.delete(y,inds_correct,axis=0)
    print(x_adv.shape)
    print(x.shape)
    print(y.shape)

    return x,x_adv,y

epochs = 100
batch_size = 64

classifier = get_cifar100_resnet101v2()
model = tf.keras.Model(inputs=classifier.input, outputs=classifier.get_layer('flatten').output,name='feature_extraction')


(x_train,y_train),(_,_) = tf.keras.datasets.cifar100.load_data()
x_train = x_train/255.0

x_adv_MIM = np.load('data/cifar100/resnet101v2/x_train_adv_MIM.npy')


x_train,x_adv_MIM,y_train = get_training_data(x_train,x_adv_MIM,y_train,classifier)
x_adv_MIM = tf.cast(x_adv_MIM,tf.float64)

g_rgb = graft_network.DetectNoise(32,32,3,epochs=epochs,batch_size=batch_size,feature_extraction_net=model)

g_rgb_n = two_stream_graft_network.DetectNoise(32,32,3,epochs=epochs,batch_size=batch_size,feature_extraction_net=model)

rgb = single_stream_detection_network.DetectNoise(32,32,3,epochs=epochs,batch_size=batch_size)

rgb_n= two_stream_detection_network.DetectNoise(32,32,3,epochs=epochs,batch_size=batch_size)

g_rgb.train(x_train, x_adv_MIM)
g_rgb.Prediction_net.save_weights('model/cifar100/discriminator/graftnet_RGB_w.h5')

g_rgb_n.train(x_train, x_adv_MIM)
g_rgb_n.Prediction_net.save_weights('model/cifar100/discriminator/graftnet_RGBandSRM_w.h5')

rgb.train(x_train, x_adv_BIM)
rgb.RGB_net.save_weights('model/cifar100/discriminator/RGB_w.h5')

rgb_n.train(x_train, x_adv_BIM)
rgb_n.Prediction_net.save_weights('model/cifar100/discriminator/RGBandSRM_w.h5')
