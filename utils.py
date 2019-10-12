# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:14:59 2019

@author: Stuart
"""
import torch
import json 
import queries

# Pre-processing Data

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, make everything lowercase, and trim most
# punctuation.
# 
# 
# 


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
"""
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
"""
# Lowercase, trim, and remove non-letter characters

"""
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
"""
 
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
# 
# 
# 

     
def loadData(fp):
    #e.g. fp = "../data/qald9/qald-9-train-multilingual.json"
    data = json.load(open( fp, "r", encoding="UTF-8"))
    pairs = [ (str(qa["question"][3]["string"]).lower().strip()[:-1], queries.encode(queries.select(qa["query"]["sparql"])), queries.preprocess(qa["answers"][0]) )  for qa in data["questions"] ] 
    return pairs

 
def readLangs(fp, reverse=False):
    print(" Reading lines...")
    # Read the file and split into lines
    #lines = open('../input/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    
    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = loadData(fp)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang("sparql")
        output_lang = Lang("en")
    else:
        input_lang = Lang("en")
        output_lang = Lang("sparql")

    return input_lang, output_lang, pairs

 
# Since there are a *lot* of example sentences and we want to train
# something quickly, we'll trim the data set to only relatively short and
# simple sentences. Here the maximum length is 10 words (that includes
# ending punctuation) and we're filtering to sentences that translate to
# the form "I am" or "He is" etc. (accounting for apostrophes replaced
# earlier).
# 
# 
# 

 
MAX_LENGTH = 30

def filterPair(p):
    return ( len(p[0].split(' ')) < MAX_LENGTH )and( len(p[1].split(' ')) < MAX_LENGTH )

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

 
# The full process for preparing the data is:
# 
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
#  

 
def prepareData(fp, reverse=False):
    input_lang, output_lang, pairs = readLangs(fp, reverse)
    print("Read %s sentence pairs"%len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs"%len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# e.g.
#input_lang, output_lang, pairs = prepareData(fp, True)
#print(random.choice(pairs))




    
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# Preparing Training Data
# -----------------------
# 
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
# 
# 
# 
 
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, ):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)



import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

