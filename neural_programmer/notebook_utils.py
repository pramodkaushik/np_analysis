""" Functions to load data, build Neural Programmer graph and restore checkpoints. 
Adapted from code in neural_programmer.py 
"""

import copy
import numpy as np
import tensorflow as tf
import time
import re
import itertools
import sys
import pickle
import os
import string
import autoreload
import wiki_data, data_utils, parameters, model

from random import shuffle
from neural_programmer import Utility

def init_data(data_dir):
    """ Load WikiTableQuestions data """
    utility = Utility()
    train_name = "random-split-1-train.examples"
    dev_name = "random-split-1-dev.examples"
    test_name = "pristine-unseen-tables.examples"
    # load data
    dat = wiki_data.WikiQuestionGenerator(train_name, dev_name, test_name, data_dir)
    train_data, dev_data, test_data = dat.load()
    utility.words = []
    utility.word_ids = {}
    utility.reverse_word_ids = {}
    # construct vocabulary
    data_utils.construct_vocab(train_data, utility)
    data_utils.construct_vocab(dev_data, utility, True)
    data_utils.construct_vocab(test_data, utility, True)
    data_utils.add_special_words(utility)
    data_utils.perform_word_cutoff(utility)
    # convert data to int format and pad the inputs
    train_data = data_utils.complete_wiki_processing(train_data, utility, True)
    dev_data = data_utils.complete_wiki_processing(dev_data, utility, False)
    test_data = data_utils.complete_wiki_processing(test_data, utility, False)
    print("# train examples ", len(train_data))
    print("# dev examples ", len(dev_data))
    print("# test examples ", len(test_data))
    return train_data, dev_data, test_data, utility

def build_graph(utility):
    """ Build Neural Programmer graph """
    # creates TF graph and calls evaluator
    batch_size = utility.FLAGS.batch_size 
    model_dir = utility.FLAGS.output_dir + "/model" + utility.FLAGS.job_id + "/"
    # create all paramters of the model
    param_class = parameters.Parameters(utility)
    params, global_step, init = param_class.parameters(utility)
    key = "test" #if (FLAGS.evaluator_job) else "train"
    graph = model.Graph(utility, batch_size, utility.FLAGS.max_passes, mode = "test")
    graph.create_graph(params, global_step)
    sess = tf.InteractiveSession()
    sess.run(init.name)
    sess.run(graph.init_op.name)
    return sess, graph, params
  
def restore_model(sess, graph, params, model_file):
    """ Restore checkpoint """
    to_save = params.copy()
    saver = tf.train.Saver(to_save, max_to_keep=500)
    saver.restore(sess, model_file)
    return sess, graph


def softmax_to_names(softmax, names):
    """Returns the names of the highest probability operator per stage in 'softmax'"""
    ids = [np.asscalar(np.argmax(softmax[stage,:])) for stage in range(softmax.shape[0])]
    #return [names[idx] + '(' + str(round(softmax[stage,idx],3)) + ')' for stage, idx in enumerate(ids)]
    return [names[idx] for idx in ids]


def get_column_names(wiki_example):
    """Returns the column names as a list"""
    ans = wiki_example.column_names + wiki_example.word_column_names
    ans = [' '.join([str(w) for w in s]) for s in ans]
    return ans

rename_dict = {
    'entry_match': 'tm_token',
    'column_match': 'cm_token',
    'first_rs': 'first',
    'last_rs': 'last',
    'group_by_max': 'mfe',
    'word-match': 'select',
    'reset_select': 'reset'
}

def rename(word):
    if word in rename_dict:
        return rename_dict[word]
    return word