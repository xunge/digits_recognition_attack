from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import json
import digits_recognition.model_attack as model
from attacks import fgm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import mpld3
from mpld3 import plugins, utils

# from digits_recognition import model_attack
# from digits_recognition import fgsm_mnist

x = tf.placeholder("float", [None, 784])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


class Dummy:
    pass


env = Dummy()


img_size = 28
img_chan = 1
n_classes = 10

with tf.variable_scope('model'):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model.model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver()

with tf.variable_scope('model', reuse=True):
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
    env.x_fgsm = fgm(model, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

# # restore trained model
# with tf.variable_scope("regression"):
#     y1, variables = model.regression(x)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "digits_recognition/model/regression.ckpt")
#
#
# with tf.variable_scope("convolutional"):
#     keep_prob = tf.placeholder("float")
#     y2, variables = model.convolutional(x, keep_prob)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "digits_recognition/model/convolutional.ckpt")


# def regression(input):
#     return sess.run(y1, feed_dict={x: input}).flatten().tolist()
#
#
# def convolutional(input):
#     return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

def train(sess, env, load=False, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, 'model/{}'.format(name))


def predict(sess, env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    过运行env.ybar进行推理。
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_fgsm(sess, env, X_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    print('\nMaking adversarials via FGSM')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)


    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv = sess.run(env.x_fgsm, feed_dict={
            env.x: X_data[start:end],
            env.fgsm_eps: eps,
            env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
    print()

    return X_adv


def index(request):
    return render(request, 'index.html')


train(sess, env, load=True, name='mnist')


@csrf_exempt
def process(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    X_adv = make_fgsm(sess, env, input, eps=0.02, epochs=12)
    # print(X_adv)

    # 一维数组，输出10个预测概率
    output1 = predict(sess, env, input).flatten().tolist()
    output2 = predict(sess, env, X_adv).flatten().tolist()
    # print(output3)
    # [2.4435046725557186e-05, 0.001754806493408978, 0.002589056035503745, 0.26431846618652344, 0.04292065650224686,
    # 0.1688787341117859, 0.0003956337459385395, 0.06867480278015137, 0.4420638382434845, 0.008379514329135418]
    return HttpResponse(json.dumps([output1, output2]))


@csrf_exempt
def drawInput(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    X_tmp1 = np.empty((10, 28, 28))
    X_tmp = 1 - input
    X_tmp1[0] = np.squeeze(X_tmp)
    print(X_tmp1[0])
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(X_tmp1[0], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(2, 2)

    gs.tight_layout(fig)
    js = mpld3.fig_to_html(fig)
    # plt.show()
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/fgsm_mnist.png')
    return HttpResponse(js)


@csrf_exempt
def attack(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    X_adv = make_fgsm(sess, env, input, eps=0.02, epochs=12)
    X_tmp = 1 - X_adv
    X_tmp1 = np.empty((10, 28, 28))
    X_tmp1[0] = np.squeeze(X_tmp[0])
    fig = plt.figure(figsize=(1, 1))
    gs = gridspec.GridSpec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(X_tmp1[0], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.set_size_inches(2, 2)

    gs.tight_layout(fig)
    js = mpld3.fig_to_html(fig)
    plt.show()
    # os.makedirs('img', exist_ok=True)
    # plt.savefig('img/fgsm_mnist.png')
    return HttpResponse(js)
