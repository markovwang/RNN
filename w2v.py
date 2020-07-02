# word2vec function
import os
import utils
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec


def train_word2vec(x):
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=35, workers=12, iter=10, sg=1)
    return model


if __name__ == '__main__':
    print("loading training data")
    train_x, y = utils.load_training_data()
    train_x_no_label = utils.load_training_data(path='training_nolabel.txt')
    print("loading testing data")
    test_x = utils.load_testing_data()
    
    model = train_word2vec(train_x+train_x_no_label+test_x)

    print("saving model")
    model.save('w2v_all.model')
