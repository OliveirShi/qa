import numpy as np
import theano
import theano.tensor as T
from config.conf import batch_size_train
from utils import init_lstm_params


class LSTM:
    def __init__(self, rng, n_input, n_hidden,
                 x, mask, Embeddings,
                 is_train=1, p=0.5):
        self.rng = rng
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.x = x
        self.mask = mask
        self.Embeddings = Embeddings
        self.is_train = is_train
        self.p = p
        self.f = T.nnet.sigmoid
        # initial parameters
        lstm_W = np.concatenate([init_lstm_params(self.n_input, self.n_hidden),
                                 init_lstm_params(self.n_input, self.n_hidden),
                                 init_lstm_params(self.n_input, self.n_hidden),
                                 init_lstm_params(self.n_input, self.n_hidden)], axis=1)
        self.W = theano.shared(value=lstm_W, name='lstm_W')
        lstm_U = np.concatenate([init_lstm_params(self.n_hidden, self.n_hidden),
                                 init_lstm_params(self.n_hidden, self.n_hidden),
                                 init_lstm_params(self.n_hidden, self.n_hidden),
                                 init_lstm_params(self.n_hidden, self.n_hidden)], axis=1)
        self.U = theano.shared(value=lstm_U, name='lstm_U')
        lstm_b = np.zeros((4 * self.n_hidden, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=lstm_b, name='lstm_b')
        # Params
        self.params = [self.W, self.U, self.b]
        self.build()

    def build(self):
        '''
            Compute the hidden state in an LSTM.
            params:
                x_t : Input Vector
                h_tm1: hidden varibles from previous time step.
                c_tm1: cell state from previous time step.
            return [h_t, c_t]
        '''
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim: (n + 1) * dim]
            return _x[:, n * dim: (n + 1) * dim]


        def _recurrence(x_t, m, h_tm1, c_tm1):
            preact = T.dot(h_tm1, self.U)
            preact += x_t
            # Input gate
            i_t = self.f(_slice(preact, 0, self.n_hidden))
            # Output gate
            f_t = self.f(_slice(preact, 1, self.n_hidden))
            # Output gate
            o_t = self.f(_slice(preact, 2, self.n_hidden))
            # Cell update
            c_tilde_t = T.tanh(_slice(preact, 3, self.n_hidden))
            c_t = f_t * c_tm1 + i_t * c_tilde_t

            # hidden state
            h_t = o_t * T.tanh(c_t)
            c_t = c_t * m[:, None] + c_tm1 * (1. - m)[:, None]
            h_t = h_t * m[:, None] + h_tm1 * (1. - m)[:, None]
            return [h_t, c_t]

        # embedding layer
        emb_layer = self.Embeddings[self.x.flatten()].reshape([self.x.shape[0],
                                                               self.x.shape[1],
                                                               self.n_input])
        state_below = T.dot(emb_layer, self.W) + self.b
        # lstm layer
        [h, c], _ = theano.scan(fn=_recurrence,
                                sequences=[state_below, self.mask],
                                truncate_gradient=-1,
                                outputs_info=[dict(initial=T.zeros((batch_size_train, self.n_hidden))),
                                              dict(initial=T.zeros((batch_size_train, self.n_hidden)))])
        # sampling layer with average sampling
        h = (h * self.mask[:, :, None]).sum(axis=0)
        h = h / self.mask.sum(axis=0)[:, None]
        # Dropout layer
        if self.p > 0:
            drop_mask = self.rng.binomial(n=1, p=1.-self.p, size=h.shape, dtype=theano.config.floatX)
            self.activation = T.switch(T.eq(self.is_train, 1), h*drop_mask, h*(1-self.p))
        else:
            self.activation = T.switch(T.eq(self.is_train, 1), h, h)