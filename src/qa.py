import theano
from theano import tensor as T
import cPickle as pkl
import numpy as np
import random
import time
from utils import PrepareData, load_model, save_model, reformat_data
from config.conf import *
from qa_net import QAnet



class AnswerSelect(object):
    def __init__(self, reload_model=False):
        # load data
        self.data_loader = PrepareData(dataTxt, test_size)
        self.data_loader.load_data()
        self.vocab = self.data_loader.vocab
        self.vocab_size = len(self.vocab)
        # initial embeddings by word2vec

        self.lr = learning_rate
        self.decay = decay
        # create model
        print "creating model..."
        self.qa_net = QAnet(embedding_size, hidden_size1, hidden_size2, self.vocab_size, dropout=dropout)

        if reload_model is True:
            self.qa_net = load_model(savePath, self.qa_net)

    def get_set(self, is_train=True):
        # return positive samples + negative samples
        if is_train:
            print "getting training set..."
            _sources, _targets = self.data_loader.get_train()
        else:
            _sources, _targets = self.data_loader.get_test()
        length = len(_targets)
        labels = [1] * length
        # generate negative samples
        shuffled_index = []
        for i in range(length):
            n = int(random.random() * (length - 1))
            if n < i:
                shuffled_index.append(n)
            else:
                shuffled_index.append(n + 1)
        neg_sources = [_sources[idx] for idx in shuffled_index]
        neg_targets = [_targets[idx] for idx in shuffled_index]
        neg_labels = [0] * length
        sources = _sources + neg_sources
        targets = _targets + neg_targets
        labels = labels + neg_labels
        return sources, targets, labels


    def train(self):
        print "start training..."
        old_costs = 1.
        for epoch in range(1):
            beg_time = time.time()
            batch_idx = 0
            costs = 0.
            # get training set
            train_sources, train_targets, train_labels = self.get_set(is_train=True)
            n_samples = len(train_sources)
            # shuffle training data
            new_index = range(n_samples)
            random.shuffle(new_index)
            sources = [train_sources[idx] for idx in new_index]
            targets = [train_targets[idx] for idx in new_index]
            labels = [train_labels[idx] for idx in new_index]

            inputs1, mask1 = reformat_data(sources)
            inputs2, mask2 = reformat_data(targets)
            output = np.array(labels).astype('float32')

            while (batch_idx + 1) * batch_size_train < n_samples:
                # batch input
                in1 = inputs1[:, batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]
                in2 = inputs2[:, batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]
                out = output[batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]
                m1 = mask1[:, batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]
                m2 = mask2[:, batch_idx * batch_size_train: (batch_idx + 1) * batch_size_train]
                cost = self.qa_net.train(in1, in2, m1, m2, out, self.lr)

                costs += cost
                batch_idx += 1
                if batch_idx % display_step == 0:
                    print "epoch: %d, batch: %d, cost: %f, lr: %f" % (epoch, batch_idx, costs/batch_idx, self.lr)
                    test_sources, test_targets, test_labels = self.get_set(is_train=False)
                    test_inputs1, test_mask1 = reformat_data(test_sources)
                    test_inputs2, test_mask2 = reformat_data(test_targets)
                    test_output = np.array(test_labels).astype('float32')
                    cost, prediction = self.qa_net.test(test_inputs1, test_inputs2, test_mask1, test_mask2, test_output)
                    # compute accuracy
                    auc = 0.
                    for i in range(test_size):
                        if prediction[i] == 1:
                            auc += 1
                    for i in range(test_size, 2*test_size):
                        if prediction[i] == 0:
                            auc += 1
                    print "test cost: %f, accuracy: %f" % (cost, auc/test_size/2)

            # decay learning rate
            if old_costs < costs/batch_idx:
                self.lr *= decay
                old_costs = costs/batch_idx
            # stop early
            if self.lr <= stop_early_lr:
                save_model(savePath, self.qa_net)
                print "Stop early, model saved in file: ", savePath
            # save model
            if epoch % save_freq == 0:
                save_model(save_freq, self.qa_net)
                print "model saved in file: ", savePath

            print "epoch: %d, time consuming: %f" % (epoch, time.time() - beg_time)


if __name__ == "__main__":
    answerselection = AnswerSelect()
    answerselection.train()
