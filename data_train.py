import os
import re
import cPickle
import copy

import numpy
import torch
import nltk
from nltk.corpus import ptb

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']

test_file_ids = []
for id in ptb.fileids():
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
        test_file_ids.append(id)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if self.word2idx.has_key(item):
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.iteritems():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print 'Number of words:', len(self.idx2word)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, maxlen, dictname):
        train_file_ids = []
        for filename in os.listdir(path):
            if filename.endswith('.txt'):
                train_file_ids.append(os.path.join(path, filename))
        dict_file_name = os.path.join(path, dictname)
        if os.path.exists(dict_file_name):
            print 'Using the dict ' + dict_file_name + ' !'
            self.dictionary = cPickle.load(open(dict_file_name, 'rb'))
        else:
            self.dictionary = Dictionary()
            self.add_words_train(train_file_ids, maxlen)
            self.add_words_test(test_file_ids, maxlen)
            self.dictionary.rebuild_by_freq()
            cPickle.dump(self.dictionary, open(dict_file_name, 'wb'))


        self.train, self.train_sens = self.tokenize_train(train_file_ids, maxlen)
        print "Train: " + str(len(self.train_sens)) + " sentences"
        self.test, self.test_sens, self.test_trees = self.tokenize_test(test_file_ids, maxlen)
        print "Test: " + str(len(self.test_sens)) + " sentences"
        self.train = self.train + self.test # train with test set

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            if tag in word_tags:
                w = w.lower()
                w = re.sub('[0-9]+', 'N', w)
                # if tag == 'CD':
                #     w = 'N'
                words.append(w)
        return words

    def add_words_train(self, file_ids, maxlen):
        print 'add_words'
        # Add words to the dictionary
        for id in file_ids:
            # print "Processing " + id
            with open(id, 'r') as f:
                for line in f:
                    words = line.strip().split()
                    words = ['<s>'] + words + ['</s>']
                    if maxlen != -1:
                        if len(words) > maxlen + 2:	# <s> and </s>
                            continue
                    for word in words:
                        self.dictionary.add_word(word)

    def add_words_test(self, file_ids):
        print 'add_words'
        # Add words to the dictionary
        for id in file_ids:
            # print "Processing " + id
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = ['<s>'] + words + ['</s>']
                if maxlen != -1:
                    if len(words) > maxlen + 2:	# <s> and </s>
                        continue
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize_train(self, file_ids, maxlen):

        sens_idx = []
        sens = []
        for id in file_ids:
            with open(id, 'r') as f:
                for line in f:
                    words = line.strip().split()
                    words = ['<s>'] + words + ['</s>']
                    if maxlen != -1:
                        if len(words) > maxlen + 2:	# <s> and </s>
                            continue
                    sens.append(words)
                    idx = []
                    for word in words:
                        idx.append(self.dictionary[word])
                    sens_idx.append(torch.LongTensor(idx))

        return sens_idx, sens

    def tokenize_test(self, file_ids, maxlen):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    return tree.leaves()[0]
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        sens_idx = []
        sens = []
        trees = []
        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = ['<s>'] + words + ['</s>']
                if maxlen != -1:
                    if len(words) > maxlen + 2:	# <s> and </s>
                        continue
                sens.append(words)
                idx = []
                for word in words:
                    idx.append(self.dictionary[word])
                sens_idx.append(torch.LongTensor(idx))
                trees.append(tree2list(sen_tree))

        return sens_idx, sens, trees
