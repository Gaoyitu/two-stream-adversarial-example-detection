import tensorflow as tf
import numpy as np
import graft_network
import two_stream_graft_network
import single_stream_detection_network
import two_stream_detection_network
from utils import compute_roc
import models


def get_testing_data(x_test,x_adv_test,y_test,classifier):
    preds = classifier.predict(x_test)
    y1 = np.reshape(y_test,(y_test.shape[0],))
    inds_correct = np.where(preds.argmax(axis=1) == y1)[0]
    x_adv_test = x_adv_test[inds_correct]
    y_test = y_test[inds_correct]
    y1= np.reshape(y_test,(y_test.shape[0],))
    x_test = x_test[inds_correct]
    preds_adv = classifier.predict(x_adv_test)
    inds_correct = np.where(preds_adv.argmax(axis=1) == y1)[0]
    x_adv = np.delete(x_adv_test,inds_correct,axis=0)
    x = np.delete(x_test,inds_correct,axis=0)
    y = np.delete(y_test,inds_correct,axis=0)

    return x,x_adv,y

def RGBandSRM_test(x_test,x_adv,y_test,classifier,SRM_net, pred_net):


    x_test, x_adv, y_test = get_testing_data(x_test, x_adv, y_test, classifier)

    x_adv_shape = x_adv.shape[0]


    x_adv_srm = SRM_net(x_adv)
    x_srm = SRM_net(x_test)

    outputs_adv = pred_net([x_adv, x_adv_srm])
    outputs = pred_net([x_test, x_srm])


    adv_pre = outputs_adv[:, 1]
    x_pre = outputs[:, 1]

    outputs_adv = tf.argmax(outputs_adv, 1)
    outputs = tf.argmax(outputs, 1)

    adv_score = tf.reduce_sum(outputs_adv)
    x_score = tf.reduce_sum(outputs)
    print('total_number:', x_adv_shape)
    print('x_score:', x_score.numpy())
    print('x_adv_score:', x_adv_shape-adv_score.numpy())
    print('P:',x_score.numpy()/(x_score.numpy()+adv_score.numpy()))
    print('R:',x_score.numpy()/x_adv_shape)

    fpr, tpr, auc_score = compute_roc(adv_pre, x_pre)

    print('auc:',auc_score)

    concat = np.vstack((fpr, tpr))
    return concat

def RGB_test(x_test,x_adv,y_test,classifier, pred_net):
    x_test, x_adv, y_test = get_testing_data(x_test, x_adv, y_test, classifier)
    x_adv_shape = x_adv.shape[0]

    outputs_adv = pred_net(x_adv)
    outputs = pred_net(x_test)

    adv_pre = outputs_adv[:, 1]
    x_pre = outputs[:, 1]

    outputs_adv = tf.argmax(outputs_adv, 1)
    outputs = tf.argmax(outputs, 1)

    adv_score = tf.reduce_sum(outputs_adv)
    x_score = tf.reduce_sum(outputs)
    print('total_number:', x_adv_shape)
    print('x_score:', x_score.numpy())
    print('x_adv_score:', x_adv_shape - adv_score.numpy())
    print('P:', x_score.numpy() / (x_score.numpy() + adv_score.numpy()))
    print('R:', x_score.numpy() / x_adv_shape)

    fpr, tpr, auc_score = compute_roc(adv_pre, x_pre)

    print('auc:', auc_score)

    concat = np.vstack((fpr, tpr))
    return concat

if __name__ =='__main__':


    (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_test = x_test / 255.0

    model_names = ['resnet101v2','resnet152','densenet169','densenet201']
    adv_methods = ['FGSM','MIM','PGD','SPSA','BIM']
    classifier = models.get_cifar100_resnet101v2()
    feature_extraction = tf.keras.Model(inputs=classifier.input, outputs=classifier.get_layer('flatten').output,
                           name='feature_extraction')

    model = 'model/cifar100/discriminator/RGBandSRM_w.h5'

    detector = two_stream_detection_network.DetectNoise(32,32,3,100,64,feature_extraction)

    net = detector.Prediction_net


    SRM = detector.SRM
    net.load_weights(model)




    for i in range(1,len(model_names),1):
        for j in range(len(adv_methods)):
            print('model_name:',model_names[i])
            print('adv_method:',adv_methods[j])
            x_adv = np.load('data/cifar100/'+model_names[i]+'/x_test_adv_'+adv_methods[j]+'.npy')
            plot_data = RGB_test(x_test,x_adv,y_test,classifier,net)










