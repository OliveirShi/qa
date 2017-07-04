import numpy as np
import theano
import theano.tensor as T


class Logistic(object):
    def __init__(self, n_features, logits, n_class):
        self.x = logits

        init_W = np.asarray(np.random.uniform(low=-np.sqrt(1./n_features),
                                              high=np.sqrt(1./n_features),
                                              size=(n_features, n_class)),
                            dtype=theano.config.floatX)
        init_b = np.zeros((n_class,), dtype=theano.config.floatX)

        self.W = theano.shared(value=init_W, name='logistic_w', borrow=True)
        self.b = theano.shared(value=init_b, name='logistic_b', borrow=True)

        self.params = [self.W, self.b]

        self.output = T.nnet.sigmoid(T.dot(self.x, self.W) + self.b)
        self.predict = T.argmax(self.output, axis=1)
