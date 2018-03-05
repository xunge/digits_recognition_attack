import tensorflow as tf


__all__ = [
    'fgm',                      # fast gradient method
    'fgmt'                      # fast gradient method with target
]


def fgm(model, x, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.):
    """
    Fast gradient method.

    See https://arxiv.org/abs/1412.6572（Explaining and Harnessing Adversarial Examples）
    and https://arxiv.org/abs/1607.02533（Adversarial examples in the physical world）
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236)（Adversarial Machine Learning at Scale）.

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

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model.model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)   #将张量值剪切到指定的最小值和最大值。
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient')
    return xadv


def fgmt(model, x, y=None, eps=0.01, epochs=1, sign=True, clip_min=0.,
         clip_max=1.):
    """
    Fast gradient method with target

    See https://arxiv.org/pdf/1607.02533.pdf.(Adversarial examples in the physical world)
    This method is different from
    FGM that instead of decreasing the probability for the correct label, it
    increases the probability for the desired label.

    :param model: A model that returns the output as well as logits.
    :param x: The input placeholder.
    :param y: The desired target label, set to the least-likely class if None.
    :param eps: The noise scale factor.
    :param epochs: Maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise gradient values.
    :param clip_min: Minimum value in output.
    :param clip_max: Maximum value in output.
    """
    xadv = tf.identity(x)

    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    n, ydim = yshape[0], yshape[1]

    if y is None:
        indices = tf.argmin(ybar, axis=1)
    else:
        indices = tf.cond(tf.equal(0, tf.rank(y)),
                          lambda: tf.zeros([n], dtype=tf.int32) + y,
                          lambda: tf.zeros([n], dtype=tf.int32))

    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: 1 - ybar,
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = -tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*noise_fn(dy_dx))
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False,
                            name='fast_gradient_target')
    return xadv
