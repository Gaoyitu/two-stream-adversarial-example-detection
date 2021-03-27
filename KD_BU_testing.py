from __future__ import division, absolute_import, print_function


import os
import argparse
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from keras.models import load_model
import tensorflow as tf

from KD_BU_utils import (get_data_mnist,get_data_cifar10,get_data_cifar100,
                        get_mc_predictions,get_deep_representations, score_samples, normalize,
                         train_lr, compute_roc)

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar': 0.26}

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

def main(classifier,model, X_train, y_train, Y_train,X_test, y_test,Y_test, X_test_adv,Bandwidth):

    batch_size = 256


    X_test,X_test_adv,Y_test = get_testing_data(X_test,X_test_adv,y_test,classifier)



    uncerts_normal = np.zeros((X_test.shape[0],),dtype=float)
    uncerts_adv = np.zeros((X_test.shape[0],),dtype=float)


    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(model, X_train,
                                                batch_size=batch_size)
    X_test_normal_features = get_deep_representations(model, X_test,
                                                      batch_size=batch_size)
    X_test_adv_features = get_deep_representations(model, X_test_adv,
                                                   batch_size=batch_size)
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]

    # print('class_inds:', class_inds)
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=Bandwidth) \
            .fit(X_train_features[class_inds[i]])

    preds_test_normal = classifier.predict(X_test)
    preds_test_adv = classifier.predict(X_test_adv)
    preds_test_normal = preds_test_normal.argmax(axis=1)
    preds_test_adv = preds_test_adv.argmax(axis=1)

    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )
    # print(densities_adv)
    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z = normalize(
        uncerts_normal,
        uncerts_adv
    )
    densities_normal_z, densities_adv_z = normalize(
        densities_normal,
        densities_adv
    )

    values, labels, lr = train_lr(
        densities_pos=densities_adv_z,
        densities_neg=densities_normal_z,
        uncerts_pos=uncerts_adv_z,
        uncerts_neg=uncerts_normal_z
    )

    ## Evaluate detector
    # Compute logistic regression model predictions
    probs = lr.predict_proba(values)[:, 1]

    # Compute AUC
    n_samples = len(X_test)

    FPR, TPR, auc_score = compute_roc(
        probs_neg=probs[:n_samples],
        probs_pos=probs[n_samples:]
    )
    print('FPR:', FPR)
    print('TPR:', TPR)
    print('auc:', auc_score)
    print('Detector ROC-AUC score: %0.4f' % auc_score)

    print('Total:',n_samples)
    print('Clean:',np.sum((probs[n_samples:])>0.5))
    print('Adv:',np.sum(probs[:n_samples]<0.5))

    print('P:',np.sum((probs[n_samples:])>0.5)/np.sum(probs>0.5))
    print('R:', np.sum(probs[n_samples:] > 0.5) / probs[:n_samples].shape[0])
    print('Detector ROC-AUC score: %0.4f' % auc_score)

    concat = np.vstack((FPR, TPR))

    return concat


if __name__ == "__main__":
    from models import get_mnist_local,get_cifar100_resnet101v2,get_cifar10_local


    x_train,y_train,Y_train,x_test,y_test,Y_test = get_data_cifar100()

    classifier = get_cifar100_resnet101v2()
    model = tf.keras.Model(inputs=classifier.input, outputs=classifier.get_layer('flatten').output,
                           name='feature_extraction')

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model_names = ['resnet101v2', 'resnet152', 'densenet169', 'densenet201']
    adv_methods = ['FGSM', 'MIM', 'PGD', 'SPSA','BIM']



    for i in range(len(model_names)):
        for j in range(len(adv_methods)):
            print('model_name:',model_names[i])
            print('adv_method:',adv_methods[j])
            x_test_adv = np.load('data/cifar100/'+model_names[i]+'/x_test_adv_'+adv_methods[j]+'.npy')

            plot_data = main(classifier,model,x_train,y_train,Y_train,x_test,y_test,Y_test,x_test_adv,BANDWIDTHS['cifar'])




