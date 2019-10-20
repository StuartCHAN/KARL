# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 17:14:59 2019

@author: Stuart
"""
import torch
import torch.nn.utils.rnn as rnn_utils
import json 
import math 
import os
import ast
import kg_utils.queries as queries 
import kg_utils.kgutils as kgutils

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
    if fp.endswith(".json"):
        #e.g. fp = "./data/qald9/qald-9-test-multilingual.json"
        data = json.load(open( fp, "r", encoding="UTF-8"))
        pairs = [ (kgutils.get_en(qa["question"])[:-1], kgutils.encode(kgutils.select(qa["query"]["sparql"])) )  for qa in data["questions"] ] 
        #pairs = [ ( str(kgutils.get_en(qa["question"]))[:-1], kgutils.encode(kgutils.select(qa["query"]["sparql"])), str(queries.preprocess(qa["answers"][0])) )  for qa in data["questions"] ] 
    elif fp.endswith(".txt") or fp.endswith(".bin") or fp.endswith(".tsv"):
        pairs = [line.split("\t") for line in open(fp, "r", encoding="UTF-8").readlines()]
    else:
        print(" The dataset file format is not supported... ")
        pairs = None
    return pairs 
 
    
def loadEvalData(fp):
    data = [ line.split("\t") for line in open( fp, "r", encoding="UTF-8").readlines() ]
    pairs = [ (pair[0], pair[1], ast.literal_eval(pair[-1])) for pair in data ] 
    return filterPairs(pairs)
    
 
def readLangs(fp, reverse=False):
    print("\n Reading lines...")
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

 
def prepareData(training_fp, eval_fp, reverse=False):
    input_lang, output_lang, training_pairs = readLangs(training_fp, reverse)
    print("\n Read %s sentence pairs"%len(training_pairs))
    eval_pairs = loadEvalData(eval_fp)
    #all_pairs = training_pairs + eval_pairs
    pairs = filterPairs(training_pairs) 
    print(" Trimmed to %s sentence pairs"%len(pairs))
    print("\n Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print(" Counted words:")
    print("\t", input_lang.name, input_lang.n_words)
    VOCAB_SIZE = int(input_lang.n_words)
    print("\t", output_lang.name, output_lang.n_words)
    print("")
    return input_lang, output_lang, pairs, eval_pairs, VOCAB_SIZE ;

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
 
def indexesFromSentence(lang, sentence,  ):
    idxs = [lang.word2index[word] for word in sentence.split(' ')]
    return idxs ;

"""l = len(idxs)
    assert(l <= size)
    gap = size - l
    if gap > 0:
        _ = [idxs.append(EOS_token)  for _ in range(gap)]"""


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence,  )
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, device):
    input_tensor = tensorFromSentence(input_lang, pair[0],  device)
    target_tensor = tensorFromSentence(output_lang, pair[1],  device)
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


def teacher_force(loss):
    if loss >= 1.0 :
        ratio = loss/math.pow(10, len(str(int(loss)))) 
        force = ratio if ratio >= 0.5 else 0.5
    elif 1.0 > loss > 0.5:
        force = 0.5
    else:
        force = loss if loss <= 0.5 else 0.5 
    return force 


MODELPATH = "./model"

def prepare_dir( model_name, stamp):    
    files= os.listdir(MODELPATH)
    models_pool = []
    for file in files: #iterate to get the folders
         if os.path.isdir(MODELPATH+"/"+file): # whether a folder 
              models_pool.append(file)
    savepath = MODELPATH+"/"+model_name+"/"+stamp
    if (model_name not in models_pool) or not( os.path.exists(savepath)) :
        try:
            os.makedirs(savepath)
        except:
            os.makedir(savepath)
    return savepath


class TxtDataset(torch.utils.data.Dataset):#????Dataset??
    def __init__(self, input_tensors, target_tensors ):
        self.input_tensors = input_tensors
        self.target_tensors = target_tensors
        assert(len(input_tensors) == len(target_tensors) )
        self.length = len(input_tensors)
 
    def __getitem__(self, index):
        input_tensor = torch.LongTensor(self.input_tensors[index].long()) #torch.FloatTensor(self.input_tensors[index]) #!!! 
        target_tensor = torch.LongTensor(self.target_tensors[index].long()) #!!! 
        #pair = torch.LongTensor(self.train_pairs[index])
        return input_tensor, target_tensor ;  #????
 
    def __len__(self):
        return self.length ;

'''
def padding(src, tgt ):
    src_len, tgt_len = len(src), len(tgt )

    src_maxsize = [int(s.size(0))  for s in src ]
    tgt_maxsize = [int(s.size(0))  for s in tgt ]

    src_pad = [torch.ones(src_maxsize, ).long() for _ in range(src_len) ] #[torch.ones(src_maxsize, ).float() for _ in range(src_len) ] #!!! 
    tgt_pad = [torch.ones(tgt_maxsize, ).long() for _ in range(tgt_len) ] #!!! 

    for i in src_len:
        end = src[i].size(0)
        src_pad[i][:end, 1] = src[i] 

    for i in tgt_len:
        end = tgt[i].size(0)
        tgt_pad[i][:end, 1] = tgt[i]

    return torch.LongTensor(src_pad), torch.LongTensor(tgt_pad) ; #torch.FloatTensor(src_pad), torch.FloatTensor(tgt_pad) #!!! 


def max_size(src_sents, tgt_sents):
    src_size = max([ len(sent.split(" "))  for sent in src_sents ])
    tgt_size = max([ len(sent.split(" "))  for sent in tgt_sents ])
    return src_size, tgt_size ; 
'''

'''
def collate_fn(src):
    src.sort(key=lambda x: len(x), reverse=True)
    src_length = [len(sq) for sq in src]
    src = rnn_utils.pad_sequence(src, batch_first=True, padding_value=0) 
    tgt.sort(key=lambda x: len(x), reverse=True)
    tgt_length = [len(sq) for sq in tgt]
    tgt = rnn_utils.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src.unsqueeze(-1), tgt.unsqueeze(-1), src_length, tgt_length 
'''


"""
def myfunction(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt
           """
