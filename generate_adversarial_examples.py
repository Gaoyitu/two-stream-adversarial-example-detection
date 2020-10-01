from cleverhans_l.future.tf2.attacks import projected_gradient_descent,\
    momentum_iterative_method,fast_gradient_method,spsa
import numpy as np
import models
import tensorflow as tf

def generate_adv_examples_PGD(model,x,y,batch_size,eps,eps_iter):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = projected_gradient_descent(model_fn=model,
                                     x=x_batch, eps=eps,
                                     eps_iter=eps_iter,
                                     nb_iter=50,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     rand_init=True,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = projected_gradient_descent(model_fn=model,
                                         x=x_batch, eps=eps,
                                         eps_iter=eps_iter,
                                         nb_iter=50,
                                         norm=np.inf,
                                         clip_min=0.0,
                                         clip_max=1.0,
                                         y=y_batch,
                                         targeted=False,
                                         rand_init=True,
                                         sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output
def generate_adv_examples_BIM(model,x,y,batch_size,eps,eps_iter):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = projected_gradient_descent(model_fn=model,
                                     x=x_batch, eps=eps,
                                     eps_iter=eps_iter,
                                     nb_iter=50,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     rand_init=False,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = projected_gradient_descent(model_fn=model,
                                         x=x_batch, eps=eps,
                                         eps_iter=eps_iter,
                                         nb_iter=50,
                                         norm=np.inf,
                                         clip_min=0.0,
                                         clip_max=1.0,
                                         y=y_batch,
                                         targeted=False,
                                         rand_init=False,
                                         sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output
def generate_adv_examples_SPSA(model,x,y,batch_size,eps):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = spsa(model_fn=model,
                          x=x_batch,
                          y=y_batch,
                          eps=eps,
                          nb_iter=20,
                          clip_min=0.0,
                          clip_max=1.0,
                          targeted=False)
    output = adv
    for i in range(1, iters, 1):
       print(i)
       x_batch = x[index:index + batch_size]
       y_batch = y[index:index + batch_size]
       index = index + batch_size
       adv = spsa(model_fn=model,
                  x=x_batch,
                  y=y_batch,
                  eps=eps,
                  nb_iter=20,
                  clip_min=0.0,
                  clip_max=1.0,
                  targeted=False)
       print(adv.numpy().shape)
       output = tf.concat([output, adv], 0)

    print(output.numpy().shape)

    return output

def generate_adv_examples_MIM(model,x,y,batch_size,eps,eps_iter):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = momentum_iterative_method(model_fn=model,
                                     x=x_batch, eps=eps,
                                     eps_iter=eps_iter,
                                     nb_iter=50,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     decay_factor=1.0,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = momentum_iterative_method(model_fn=model,
                                        x=x_batch, eps=eps,
                                        eps_iter=eps_iter,
                                        nb_iter=50,
                                        norm=np.inf,
                                        clip_min=0.0,
                                        clip_max=1.0,
                                        y=y_batch,
                                        targeted=False,
                                        decay_factor=1.0,
                                        sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output

def generate_adv_examples_FGSM(model,x,y,batch_size,eps):

    index = 0

    iters = x.shape[0]//batch_size
    print(iters)

    x_batch = x[index:index + batch_size]
    y_batch = y[index:index + batch_size]
    index = index + batch_size
    print(x_batch.shape)

    adv = fast_gradient_method(model_fn=model,
                                     x=x_batch, eps=eps,
                                     norm=np.inf,
                                     clip_min=0.0,
                                     clip_max=1.0,
                                     y=y_batch,
                                     targeted=False,
                                     sanity_checks=False)
    output = adv

    for i in range(1,iters,1):
        x_batch = x[index:index+batch_size]
        y_batch = y[index:index+batch_size]
        index = index + batch_size
        adv = fast_gradient_method(model_fn=model,
                                   x=x_batch, eps=eps,
                                   norm=np.inf,
                                   clip_min=0.0,
                                   clip_max=1.0,
                                   y=y_batch,
                                   targeted=False,
                                   sanity_checks=False)
        print(adv.numpy().shape)
        output = tf.concat([output,adv],0)
    print(output.numpy().shape)

    return output

if __name__ == "__main__":


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test / 255.0
    x_train = x_train / 255.0

    y_train = np.reshape(y_train, (y_train.shape[0],)).astype(np.int64)
    y_test = np.reshape(y_test, (y_test.shape[0],)).astype(np.int64)


    model_mobilenet = models.get_cifar10_mobilenet()
    model_resnet50 = models.get_cifar10_resnet50()

    model_mobilenet.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    model_resnet50.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    batch_size = 1000
    eps = 0.05


    adv = generate_adv_examples_MIM(model=model_mobilenet, x=x_train, y=y_train, batch_size=batch_size, eps=eps,
                                    eps_iter=0.001)
















