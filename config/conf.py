# -*- coding: utf-8 -*-
# configuration for project
# train model
hidden_size1 = 128  # hidden_size for branch query
hidden_size2 = 128  # hidden_size for branch answer
learning_rate = 0.01  # initial lr
init_scale = 0.1  # for W and b
batch_size_train = 256  # batch
embedding_size = 100
decay =0.95  # float
dropout = 0.8  # tuple of float
keep = 0.1  # float in (0.,1.)
n_layer = 1  # int
stop_early_lr = 0.001  # float,must < learning_rate,if lr<stop_early_lr,stop training.
save_freq = 10  # int,save every n epochs
display_step = 10  # int,display every n steps
n_epoch = 20
# path
savePath = '../model/webqa/model_bestNg.pkl'
vocabPath = '../data/webqa/vocab.pkl'
embPath = '../data/webqa/qa_pairs_128.bin'
# embPath = 'data/webqa/big_webqa.bin'
dataTxt = '../data/webqa/qa_pairs.txt'
greatCorpus = '../data/webqa/great_corpus.txt'
dataPkl = '../data/webqa/data.pkl'
# data process
test_size = 128
s_size = 30
t_size = 10
use_jieba = True
min_count = 1
