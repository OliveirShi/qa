import numpy as np
import theano
import theano.tensor as T
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lstm import LSTM
from softmax import Softmax
from logistic import Logistic
from updates import *

class QAnet(object):
    def __init__(self, embedding_size, n_hidden1, n_hidden2,
                 vocab_size, embeddings=None, dropout=0.5, alpha=0.1, optimizer=sgd):
        self.x1 = T.imatrix('batched_inputs1')
        self.mask1 = T.matrix('mask1')
        self.x2 = T.imatrix('batched_inputs2')
        self.mask2 = T.matrix('mask2')
        self.y = T.ivector('labels')
        self.n_input1 = embedding_size
        self.n_input2 = embedding_size
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        if embeddings is not None:
            self.Embeddings = theano.shared(value=embeddings.astype(theano.config.floatX),
                                            name='embeddings', borrow=True)
        else:
            init_Embd = np.asarray(np.random.uniform(low=-np.sqrt(1. / vocab_size),
                                                     high=np.sqrt(1. / vocab_size),
                                                     size=(vocab_size, embedding_size)),
                                   dtype=theano.config.floatX)
            self.Embeddings = theano.shared(value=init_Embd, name='embeddings', borrow=True)
        # params for project from branch
        init_proj_M = np.asarray(np.random.uniform(low=-np.sqrt(1. / self.n_hidden2),
                                                   high=np.sqrt(1. / self.n_hidden2),
                                                   size=(self.n_hidden2, self.n_hidden1)),
                                 dtype=theano.config.floatX)
        self.proj_M = theano.shared(value=init_proj_M, name='proj_W', borrow=True)
        self.proj_b = theano.shared(value=np.zeros((self.n_hidden1,), dtype=theano.config.floatX),
                                    name='proj_b', borrow=True)
        self.dropout = dropout
        self.is_train = T.iscalar('is_train')
        self.lr = T.scalar('lr')
        self.optimizer = optimizer
        self.rng = RandomStreams(1234)
        self.alpha = alpha
        # parameters for softmax

        self.build()

    def build(self):
        print "building 2 channels lstm..."
        branch1 = LSTM(self.rng, self.n_input1, self.n_hidden1,
                       self.x1, self.mask1, self.Embeddings,
                       self.is_train, self.dropout)

        branch2 = LSTM(self.rng, self.n_input2, self.n_hidden2,
                       self.x2, self.mask2, self.Embeddings,
                       self.is_train, self.dropout)
        # output layer
        # p = f(Q * (A X proj_W))  # (n_batch, n_hidden1) * ((n_batch, n_hidden2) X (n_hidden2, n_hidden1))
        logits = branch1.activation * T.dot(branch2.activation, self.proj_M) + self.proj_b
        logi_len = self.n_hidden1
        # softmax layer
        # print "building softmax layer..."
        # softmax_layer = Softmax(logi_len, logits, 2)
        # logi_out = softmax_layer.output
        # prediction = softmax_layer.predict
        # logistic regression
        logistic_regression = Logistic(logi_len, logits, 2)
        logi_out = logistic_regression.output
        prediction = logistic_regression.predict
        loss = - T.sum(self.y * T.log(logi_out[:, 1]) + (1-self.y) * T.log(logi_out[:, 0]))
        # loss = T.mean(T.nnet.crossentropy_categorical_1hot(logi_out, self.y))
        self.params = [self.Embeddings, self.proj_M, self.proj_b]
        self.params += branch1.params
        self.params += branch2.params
        # self.params += softmax_layer.params
        self.params += logistic_regression.params
        # gradients
        gparams = [T.clip(T.grad(loss, p), -10, 10) for p in self.params]
        # update parameters
        updates = self.optimizer(params=self.params, gparams=gparams, learning_rate=self.lr)

        self.train = theano.function(inputs=[self.x1, self.x2, self.mask1, self.mask2, self.y, self.lr],
                                     outputs=[loss, logits, logi_out],
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})
        self.test = theano.function(inputs=[self.x1, self.x2, self.mask1, self.mask2, self.y],
                                    outputs=[loss, prediction],
                                    givens={self.is_train: np.cast['int32'](0)})
        self.predict = theano.function(inputs=[self.x1, self.x2, self.mask1, self.mask2],
                                       outputs=prediction,
                                       givens={self.is_train: np.cast['int32'](0)})