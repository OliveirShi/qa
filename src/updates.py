import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict


def sgd(params, gparams, learning_rate=0.01):
    return [(p, p-learning_rate*gp)for p, gp in zip(params, gparams)]


def adam(params, gparams, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    '''
    Adaptive Moment Estimation
    Reference: [ADAM: A Method for Stochastic Optimization.]
    '''
    updates = []
    i = theano.shared(np.dtype(theano.config.floatX).type(0))
    i_t = i + 1.
    fix1 = T.sqrt(1. - b2**i_t)
    fix2 = 1 - b1**i_t
    lr_t = lr * fix1 / fix2
    for p, g in zip(params, gparams):

        # g = T.clip(g, -grad_clip, grad_clip)
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = b1 * m + (1. - b1) * g
        v_t = b2 * v + (1. - b2) * T.sqr(g)
        M_t = m_t / (1 - b1**i_t)
        V_t = v_t / (1 - b2**i_t)
        p_t = p - lr_t * M_t / (T.sqrt(V_t) + e)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


def rmsprop(params, gparams, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """RMSProp updates
    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [1]_ for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    rho : float or symbolic scalar
        Gradient moving average decay factor
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    `rho` should be between 0 and 1. A value of `rho` close to 1 will decay the
    moving average slowly and a value close to 0 will decay the moving average
    fast.
    Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
    learning rate :math:`\\eta_t` is calculated as:
    .. math::
       r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
       \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}
    References
    ----------
    .. [1] Tieleman, T. and Hinton, G. (2012):
           Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
           Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU (formula @5:20)
    """

    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = T.constant(1)

    for param, grad in zip(params, gparams):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates