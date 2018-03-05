from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import json
import digits_recognition.model_attack as model

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


def index(request):
    return render(request, 'index.html')


train(sess, env, load=True, name='mnist')


@csrf_exempt
def process(request):
    #标准化数据
    input = ((255 - np.array(eval(request.POST.get('inputs')), dtype=np.float32)) / 255.0).reshape(1, 28, 28, 1)
    print(input)
    #一维数组，输出10个预测概率
    output3 = predict(sess, env, input).tolist()
    print(output3[0])
    # output1 = regression(input)
    # output2 = convolutional(input)
    '''
    {"results":[
        [0.0005708700628019869,0.010075394995510578,0.8699323534965515,0.0013963828096166253,0.028609132394194603,0.006814470514655113,0.06850877404212952,0.006337625440210104,0.004338784143328667,0.003416265593841672],
        [5.7194258261006325e-05,0.0006196154863573611,0.9920960664749146,0.000495785498060286,1.5396590242744423e-05,0.002464226447045803,0.00023624727327842265,0.0021845928858965635,0.0004759470175486058,0.001354911015368998]
    ]}
    '''
    return HttpResponse(json.dumps([output3[0], output3[0]]))

