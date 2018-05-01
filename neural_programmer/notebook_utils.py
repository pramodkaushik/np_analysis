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
from neural_programmer import Utility, evaluate

def init_data(data_dir, preserve_vocab=False, 
        split_filenames = {
            'train': 'random-split-1-train.examples', 
            'dev': 'random-split-1-dev.examples', 
            'test': 'pristine-unseen-tables.examples'
            }, 
        annotated_filenames = {
            'train': 'training.annotated',
            'test': 'pristine-unseen-tables.annotated'
            }):
    """ Load WikiTableQuestions data. 
    preserve_vocab is used when perturbed data is loaded, 
    in which case special words are given hard-coded ids
    to match that of the unperturbed data case
    """
    utility = Utility()
    train_name = split_filenames['train']
    dev_name = split_filenames['dev']
    test_name = split_filenames['test']
    # load data
    dat = wiki_data.WikiQuestionGenerator(train_name, dev_name, test_name, data_dir)
    train_data, dev_data, test_data = dat.load(annotated_filenames)
    utility.words = []
    utility.word_ids = {}
    utility.reverse_word_ids = {}
    # construct vocabulary
    data_utils.construct_vocab(train_data, utility)
    data_utils.construct_vocab(dev_data, utility, True)
    data_utils.construct_vocab(test_data, utility, True)
    data_utils.add_special_words(utility)
# set absolute word_ids for special words
    if preserve_vocab:
        print("hardcoded ids for special words")
        word_to_swap = utility.reverse_word_ids[9133]
        word_id_to_swap = utility.word_ids[utility.entry_match_token]
        utility.word_ids[word_to_swap] = utility.word_ids[utility.entry_match_token]
        utility.word_ids[utility.entry_match_token] = 9133
        utility.entry_match_token_id = utility.word_ids[utility.entry_match_token]
        utility.reverse_word_ids[word_id_to_swap] = word_to_swap
        utility.reverse_word_ids[9133] = utility.entry_match_token

        word_to_swap = utility.reverse_word_ids[9134]
        word_id_to_swap = utility.word_ids[utility.column_match_token]
        utility.word_ids[word_to_swap] = utility.word_ids[utility.column_match_token]
        utility.word_ids[utility.column_match_token] = 9134
        utility.column_match_token_id = utility.word_ids[utility.column_match_token]
        utility.reverse_word_ids[word_id_to_swap] = word_to_swap
        utility.reverse_word_ids[9134] = utility.column_match_token

        word_to_swap = utility.reverse_word_ids[9135]
        word_id_to_swap = utility.word_ids[utility.dummy_token]
        utility.word_ids[word_to_swap] = utility.word_ids[utility.dummy_token]
        utility.word_ids[utility.dummy_token] = 9135
        utility.dummy_token_id = utility.word_ids[utility.dummy_token]
        utility.reverse_word_ids[word_id_to_swap] = word_to_swap
        utility.reverse_word_ids[9135] = utility.dummy_token

        word_to_swap = utility.reverse_word_ids[9136]
        word_id_to_swap = utility.word_ids[utility.unk_token]
        utility.word_ids[word_to_swap] = utility.word_ids[utility.unk_token]
        utility.word_ids[utility.unk_token] = 9136
        utility.unk_token_id = utility.word_ids[utility.unk_token]
        utility.reverse_word_ids[word_id_to_swap] = word_to_swap
        utility.reverse_word_ids[9136] = utility.unk_token

        print(utility.entry_match_token_id, utility.column_match_token_id, utility.dummy_token_id, utility.unk_token_id)

    data_utils.perform_word_cutoff(utility)
    # convert data to int format and pad the inputs
    train_data = data_utils.complete_wiki_processing(train_data, utility, True)
    dev_data = data_utils.complete_wiki_processing(dev_data, utility, False)
    test_data = data_utils.complete_wiki_processing(test_data, utility, False)
    print(("# train examples ", len(train_data)))
    print(("# dev examples ", len(dev_data)))
    print(("# test examples ", len(test_data)))
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

def process_table_key(key):
    return re.sub(r'csv/([0-9]*)-csv/([0-9]*).csv', r'\1_\2', key)

def evaluate_concatenation_attack(sess, data, batch_size, graph, model_step, utility, phrase, suffix=False):
    ids = [utility.word_ids[w] if w in utility.word_ids else utility.word_ids[utility.unk_token] for w in phrase.split()]
    print(ids)
    new_data = copy.deepcopy(data)
    for i, wiki_example in enumerate(new_data):
        question_begin = np.nonzero(
            wiki_example.question_attention_mask)[0].shape[0]
        if suffix:
            word_ids = [w for w in new_data[i].question if w not in [utility.entry_match_token_id, utility.column_match_token_id]]
            addnl_token_ids = [w for w in new_data[i].question if w in [utility.entry_match_token_id, utility.column_match_token_id]]
            new_data[i].question = word_ids + ids + addnl_token_ids
            new_data[i].question = new_data[i].question[len(ids):]

            new_data[i].question_attention_mask.extend([0] * len(ids))
            new_data[i].question_attention_mask = new_data[i].question_attention_mask[len(ids):]
        else:
            new_data[i].question[question_begin-len(ids):question_begin] = ids
            new_data[i].question_attention_mask[question_begin-len(ids):question_begin] = [0] * len(ids)
    return evaluate(sess, new_data, batch_size, graph, model_step)
