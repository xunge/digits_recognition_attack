import tensorflow as tf
import numpy as np


__all__ = [
    'pgd',                      # Projected gradient descent
]


def pgd(model, x, eps=0.01, epsilon=0.3, epochs=1, sign=True, clip_min=0., clip_max=1., random_start=False):
    """
    Projected gradient descent.

    See https://arxiv.org/abs/1706.06083（Towards Deep Learning Models Resistant to Adversarial Attacks）
    for details.

    :param random_start: Give a nitialization disturbance
    :param epsilon: The scale factor for disturbance
    :param model: A wrapper that returns the output as well as logits.一个包装器，返回输出以及logits。
    :param x: The input placeholder.
    :param eps: The scale factor for noise.噪音的比例因子。
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.如果为真，则使用梯度符号，否则使用梯度值。
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)   # 返回了一个一模一样新的tensor

    ybar = model.model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    if random_start:
        # x_reshape = tf.reshape(x, [1, 28, 28, 1])
        # print(x_reshape.shape)
        # print(xadv.shape)
        xadv = x + tf.random_uniform((1, 28, 28, 1), -epsilon, epsilon)
        # print(np.random.uniform(-epsilon, epsilon, x_reshape.shape))
        # xadv = x + np.random.uniform(-epsilon, epsilon, (1, 28, 28, 1))

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model.model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, x - epsilon, x + epsilon)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)   #将张量值剪切到指定的最小值和最大值。
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient')
    return xadv
