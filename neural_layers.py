# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:23:17 2019

@author: Stuart Chen

#(env)> pip3 install pytorch-pretrained-bert 
"""
from __future__ import unicode_literals, print_function, division
from io import open
#import unicodedata
#import string
#import re
import random
import time
import numpy as np 
import os
import math 

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

import kg_utils.reward as reward 

import utils
from utils import *

import transf_decoder 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n device: ", device)

print("\n torch version ", torch.__version__ )




# Define the Encoder
def swish(x):
    return x * torch.sigmoid(x)

class BERTEncoder(BertForSequenceClassification):
#    def __init__(self, config, num_labels=2):
#        super(BERTEncoder, self).__init__(config, num_labels)
#        self.num_labels = num_labels
#        self.bert = BertModel(config)
#        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#        self.hidden_size = config.hidden_size #!
#        self.classifier = nn.Linear(config.hidden_size, num_labels)
#        self.apply(self.init_bert_weights)
        
    def __init__(self, config, num_labels=2):
        super(BERTEncoder, self).__init__(config, num_labels)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size #!
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        
    def get_embedding(self, VOCAB_SIZE):
        self.bert.embeddings.word_embeddings = nn.Embedding(VOCAB_SIZE, self.hidden_size, padding_idx=0)
        print(self.bert.embeddings)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = swish(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits ;
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Define the Decoder 
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size ) #!!! , num_layers=3 ) 
        #self.gru2 = nn.GRU(hidden_size, hidden_size ) #!!! , num_layers=3 ) 
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        #output, hidden = self.gru2(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device) ;
 
    
 
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.birnn = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=True ) #!!! added
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #input, hidden = input_and_hidden
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            F.relu(self.attn(torch.cat((embedded[0], hidden[0]), 1))), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        bihidden = hidden.repeat(2,1,1).view(2,1,-1)
        bioutput, bihidden = self.birnn(output, bihidden ) #!!! added
        #print("\n birnn--> output, hidden : ", output.size(), hidden.size() )
        bioutput, bihidden = bioutput.view([-1,1,256]), torch.mean(bihidden,dim=0,keepdim=True)
        output, hidden = self.gru(bioutput, bihidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

 
   
# Evaluation for the Model 
"""
def evaluate(encoder, decoder, sentence, training_ans, input_lang, output_lang, max_length=utils.MAX_LENGTH, rl=False):
    with torch.no_grad():
        input_tensor = utils.tensorFromSentence(input_lang, sentence, device )
        input_length = input_tensor.size()[0]
        
        encoder_hidden = encoder(input_tensor)
    
        encoder_hidden = encoder_hidden.unsqueeze(0)
  
        decoder_input = torch.tensor([[utils.SOS_token]], device=device)

        decoder_hidden = encoder_hidden 

        decoded_words = [] #!

        for di in range(input_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #!!!
            if topi.item() == utils.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            #!!!
            decoder_input = topi.squeeze().detach()  # detach from history as input

            #if decoder_input.item() == utils.EOS_token:
                #break;

        decoded_sentence = str(" ").join(decoded_words) 
        
        if rl and (training_ans is not None):
            rewrd = reward.get_reward(decoded_sentence, training_ans )
            print("\n   -reward -> ", rewrd)
            return rewrd
        else:
            print("\n   -query -> ", decoded_sentence, "\n ")
            return decoded_sentence ;
"""
#
#def evaluate(encoder, decoder, sentence, training_ans, input_lang, output_lang, max_length=utils.MAX_LENGTH, rl=False):
#    with torch.no_grad():
#        
#        input_tensor = utils.tensorFromSentence(input_lang, sentence, device )
#        
#        input_length = input_tensor.size(0)
#        print(" evaluation input_length: ", input_length)
#        
#        encoder_hidden = encoder(input_tensor)
#        
#        encoder_hidden = encoder_hidden.unsqueeze(0)
#    
#        decoder_input = torch.tensor([[utils.SOS_token]], device=device)
#    
#        decoder_hidden = encoder_hidden
#        
#        #decoder_hidden_input = decoder_hidden #!!!
#
#        # Without teacher forcing: use its own predictions as the next input
#        decoded_words = [] #!
#
#        for di in range(input_length):
#            print(di, " decoder_hidden shape: ", decoder_hidden.size(), " \n ",  decoder_hidden )
#            decoder_hidden = decoder_hidden[:, 0, :]
#            #decoder_hidden = decoder_hidden_input[:, di, :]  #!!!
#            decoder_hidden = decoder_hidden.view(1,1,256)
#            
#            decoder_output, decoder_hidden = decoder(
#                decoder_input, decoder_hidden)
#            topv, topi = decoder_output.topk(1)
#            #!!!
#            if topi.item() == utils.EOS_token:
#                decoded_words.append('<EOS>')
#                break
#            else:
#                decoded_words.append(output_lang.index2word[topi.item()])
#            #!!!
#            decoder_input = topi.squeeze().detach()  # detach from history as input
#
#            if decoder_input.item() == utils.EOS_token:
#                break;
#                
#        decoded_sentence = str(" ").join(decoded_words) 
#        print("\n  --query--> ", decoded_sentence, "\n ") 
#        
#        if not rl or (training_ans is None):
#            return decoded_sentence 
#        else:
#            rewrd = reward.get_reward(decoded_sentence, training_ans )
#            print("\n  --reward--> ", rewrd)
#            return rewrd 
#            


def evaluate(encoder, decoder, sentence, training_ans, input_lang, output_lang, max_length=utils.MAX_LENGTH, rl=True):
    with torch.no_grad():
        
        input_tensor = utils.tensorFromSentence(input_lang, sentence, device )
        
        input_length = input_tensor.size(0)
        print(" evaluation input_length: ", input_length) 

        """#!!! 
        encoder_hidden = encoder.initHidden() 
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei] )
            encoder_outputs[ei] = encoder_output[0, 0] ; """ 
        
        encoder_hidden = encoder(input_tensor)
        
        encoder_hidden = encoder_hidden.unsqueeze(0)
    
        decoder_input = torch.tensor([[utils.SOS_token]], device=device)
    
        decoder_hidden = encoder_hidden
        
        #decoder_hidden_input = decoder_hidden #!!!

        src = encoder_hidden.reshape([-1, 256])
        #print("\n src: ", src.size(), "\t", src.size(1) )
        #print(max_length, src.size(0))
        encoder_outputs = F.pad(src, (0,0,0,int(max_length-src.size(0))), "constant", 0)
        #print(encoder_outputs.size())

        # Without teacher forcing: use its own predictions as the next input
        decoded_words = [] #!

        for di in range(max_length):
            #print(di, " decoder_hidden shape: ", decoder_hidden.size(), " \n ",  decoder_hidden )
            decoder_hidden = decoder_hidden[:, 0, :]  
            decoder_hidden = decoder_hidden.view(1,1,256)
            
            decoder_output, decoder_hidden, decoder_attention = decoder( #decoder_input, decoder_hidden)
                decoder_input, decoder_hidden, encoder_outputs )
            topv, topi = decoder_output.topk(1)
            #!!!
            if topi.item() == utils.EOS_token or (topi.item() == utils.SOS_token and di>1 ):
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            #!!!
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if decoder_input.item() == utils.EOS_token or (decoder_input.item() == utils.SOS_token and di>1 ):
                break;
                
        decoded_sentence = str(" ").join(decoded_words) 
        print("\n  --query--> ", decoded_sentence, "\n ") 
        
        if (not rl) or (training_ans is None):
            return decoded_sentence 
        else:
            rewrd = reward.get_reward(decoded_sentence, training_ans )
            print("  --reward--> ", rewrd, "\n")
            return rewrd 
'''                 
def evaluate(encoder, decoder, sentence, training_ans, input_lang, output_lang, max_length=utils.MAX_LENGTH, rl=True):
    with torch.no_grad():
        input_tensor = utils.tensorFromSentence(input_lang, sentence, device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        decoded_sentence = str(" ").join(decoded_words)
        print("\n  --query--> ", decoded_sentence, "\n ") 
        
        if (not rl) or (training_ans is None):
            return decoded_sentence 
        else:
            rewrd = reward.get_reward(decoded_sentence, training_ans )
            print("\n  --reward--> ", rewrd)
            return rewrd 
'''         
       
def evaluateRandomly(encoder, decoder, eval_pairs, input_lang, output_lang, n=10, rl=True):
    #output_sentences = []
    rewrds = 0.0
                                                       
    training_pairs = []
    training_answers = []#!

    for _ in range(n):
        pair = random.choice(eval_pairs)
        #training_pairs.append( utils.tensorsFromPair(pair[:-1], input_lang, output_lang, device) )
        training_pairs.append( pair[:-1] )
        training_answers.append(pair[-1])#!

    for pair, ans in zip(training_pairs, training_answers):
        #pair = random.choice(pairs)
        #print('>', pair[0])
        #print('=', pair[1])
        rewrd = evaluate(encoder, decoder, pair[0], ans, input_lang, output_lang, rl=rl)
        if rl:
            rewrds += rewrd ;
        #output_sentence = ' '.join(output_words)
        #print('<', output_sentence)
        #output_sentences.append(output_sentence)
        #print('')
        #rew += reward.get_reward(output_sentence, pair[-1])
    if rl:
        reward_value = rewrds/n
        return reward_value 
    else:
        return 1.0; 



# Training the Model 
"""
teacher_forcing_ratio = 0.5

def trainBert(input_tensor, target_tensor, encoder, decoder, training_ans, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, max_length=utils.MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0.0
    
    encoder_hidden = encoder(input_tensor)
    
    encoder_hidden = encoder_hidden.unsqueeze(0)

    #for ei in range(input_length):
        #encoder_output, encoder_hidden = encoder(
            #input_tensor[ei], encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[utils.SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    rewards = [] #!

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
            v = loss.detach().numpy()
            print("\t * %s step xentrop: "%str(di), float(v/(di+1.0)), " \n" ) #!
    else:
        # Without teacher forcing: use its own predictions as the next input
        decoded_words = [] #!

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #!!!
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            #!!!
            decoder_input = topi.squeeze().detach()  # detach from history as input

            decoded_sentence = str(" ").join(decoded_words) 
            print("\n   -query -> ", decoded_sentence, "\n ")
            rew = reward.get_reward(decoded_sentence, training_ans )
            rewards.append(rew)
            print("\n   -reward -> ", rew)
            loss += criterion(decoder_output, target_tensor[di])
            v = loss.detach().numpy()
            print("\n\t * %s step xentrop: "%str(di), float(v/(di+1.0)) )#!
            print("\n ")#!
            if decoder_input.item() == utils.EOS_token:
                break;
               
    #_, rewrd = evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=target_length)
    if np.mean(rewards) >= 1.0:
        loss = torch.mul(loss, torch.FloatTensor([np.mean(rewards)]))
    else:
        pass;
    
    var = loss.detach().numpy()/target_length
    print("\n Loss:",  var)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length 
"""


"""def trainBert(input_tensor, target_tensor, encoder, decoder, eval_pairs, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5, rl=True ):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, 256, device=device) #!!! (max_length, encoder.hidden_size, device=device)
    
    loss = 0
    
    encoder_hidden = encoder(input_tensor)
    
    encoder_hidden = encoder_hidden.unsqueeze(0)
    #print("\n real encoder_hidden: ", encoder_hidden.size(), " \n")

    src = encoder_hidden.reshape([-1, 256])
    #print("\n src: ", src.size(), "\t", src.size(1) )
    #print(MAX_LENGTH, src.size(0))
    encoder_outputs = F.pad(src, (0,0,0,int(MAX_LENGTH-src.size(0))), "constant", 0)
    #print(encoder_outputs.size())

    decoder_input = torch.tensor([[utils.SOS_token]], device=device)

    decoder_hidden = encoder_hidden[0, 0].view(1,1,-1) #!!! v02 
    #encoder_outputs = decoder_hidden

    use_teacher_forcing = False #True if random.random() < teacher_forcing_ratio else False
    #print("  * T-forcing ratio: ", teacher_forcing_ratio , str(use_teacher_forcing) )
    #!!!
    decoded_words = []
    
    #!!!
    losses = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        #print("  * Teacher forcing: ")
        #print("\n decoder_input, decoder_hidden, encoder_outputs : ",
        #    decoder_input.size(), decoder_hidden.size(), encoder_outputs.size() )
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(input=decoder_input, \
                hidden=decoder_hidden, encoder_outputs=encoder_outputs )#!!!
            #    decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])

            loss_ = loss
            v = float(loss_.detach().numpy()/(di+1.0)) 
            losses.append(v)
             
            decoder_input = target_tensor[di]  # Teacher forcing

            if decoder_input.item() == utils.EOS_token or (decoder_input.item() == utils.SOS_token and di>1 ):
                break
    else:
        # Without teacher forcing: use its own predictions as the next input
        #print("  * Not teacher forcing: ")
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(input=decoder_input, \
                hidden=decoder_hidden, encoder_outputs=encoder_outputs )#!!!
            #    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #!!!
            if topi.item() == utils.EOS_token :#or (topi.item() == utils.SOS_token and di>1 ):
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            
            #!!!
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

            loss_ = loss
            v = float(loss_.detach().numpy()/(di+1.0))
            losses.append(v) 

            if decoder_input.item() == utils.EOS_token :#or (decoder_input.item() == utils.SOS_token and di>1 ):
                break

    #tgt_sent = str(" ").join([output_lang.index2word[i] for i in list(target_tensor.detach().numpy()) if i in output_lang.index2word.keys()])
    nmt_sent = str(" ").join(decoded_words)
    '''print("  tgt: ", target_tensor.detach().numpy())
    print("  output: ", output_tensor.detach().numpy())'''
    print("\n  nmt_sent: ", nmt_sent, " \n")

    if rl and (np.mean(losses) < 1.0) :
        reward_value = evaluateRandomly(encoder, decoder, eval_pairs, input_lang, output_lang, n=10, rl=rl) 
        if 2.0 > reward_value > 1.0:
            loss = torch.mul(loss, torch.FloatTensor([reward_value]) )
    else:
        pass;

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length """

def trainBert(input_tensor,  target_tensor,  model, eval_pairs, input_lang, output_lang, optimizer, criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5, rl=True ):

    #encoder_optimizer.zero_grad()
    #decoder_optimizer.zero_grad()
    optimizer.zero_grad()

    #input_length = input_tensor.size(0)
    #target_length = target_tensor.size(0)

    model = transf_decoder.Transformer(input_lang, output_lang)

    output = model(input_tensor, target_tensor )

    batch_size, seq_len = target_tensor.size(0), target_tensor.size(1)
    target_tensor = target_tensor.reshape([batch_size, seq_len])

    loss = criterion(output, target_tensor)
    
    loss.backward()

    #encoder_optimizer.step()
    #decoder_optimizer.step()
    optimizer.step()

    return loss.item()
       


def trainItersBert(model, n_iters, training_pairs, eval_pairs, input_lang, output_lang, batch_size, learning_rate=0.01, mom=0,  model_name="qald-test"):
    #start = time.time()
    plot_losses = []
    losses_trend = []
     
    #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=mom)
    #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=mom)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=mom)
    
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, amsgrad=True)
    #encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, n_iters)
    #decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, amsgrad=True)
    #decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, n_iters)                                                       

    teacher_forcing_ratio = 1.0

    criterion = nn.NLLLoss()
    '''
    input_tensors, target_tensors, train_pairs = [], [], []
    for pair in training_pairs:
        tensors = utils.tensorsFromPair(pair, input_lang, output_lang, device)
        train_pairs.append(tensors)
        #print("tensor shape--> ", tensors[0].size())
        #print(tensors[0])
        input_tensors.append(tensors[0].view(-1,1).long()) #float() #!!! 
        target_tensors.append(tensors[1].view(-1,1).long()) #!!!

    print("\n Dataset preparing... ")
    input_tensors  = rnn_utils.pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensors  = rnn_utils.pad_sequence(target_tensors, batch_first=True, padding_value=0)
    
    torch.save(input_tensors, "./model/input_tensors.pt")
    torch.save(target_tensors, "./model/target_tensors.pt")'''

    eval_tensors = [utils.tensorsFromPair(pair, input_lang, output_lang, device) for pair in eval_pairs ] 
    eval_inputs = [ tensors[0] for tensors in eval_tensors ]
    eval_targets = [ tensors[1] for tensors in eval_tensors ]
    
    eval_inputs  = rnn_utils.pad_sequence(eval_inputs, batch_first=True, padding_value=0)
    eval_targets = rnn_utils.pad_sequence(eval_targets, batch_first=True, padding_value=0)

    #input_tensors, target_tensors = utils.padding(input_tensors, target_tensors )
    '''torch_dataset = utils.TxtDataset(input_tensors, target_tensors  )'''
    torch_dataset = utils.TxtDataset(eval_inputs, eval_targets  )
    
    # put the dataset into DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,  # MINIBATCH_SIZE = 6
        shuffle=True,
        drop_last= False,
        num_workers= 2 if utils.getOSystPlateform() else 0  # set multi-work num read data based on OS plateform
        #collate_fn= utils.collate_fn  #!!! 
    ) 
    print(" Dataset loader ready, begin training. \n") 

    datset_len = len(loader)
    
    print("\n Dataset loader length is ", datset_len, ", save model every batch. " )

    for epoch in range(1, n_iters + 1):
    # an epoch goes the whole data
        for batch, (batch_input, batch_target) in enumerate(loader):
            # here to train your model
            print('\n\n  - Epoch ', epoch, ' | batch ', batch, '\n | input lenght:   ', batch_input.size(), '\n | target length:   ', batch_target.size() ," \n")  
            
            #input_tensor, target_tensor = batch_input, batch_target  #!!! 
            #print("  * T-forcing ratio: ", teacher_forcing_ratio  )
            '''try:
                input_seq_len, target_seq_len = batch_input.size(1), batch_target.size(1)
                batch_input = batch_input.reshape( [input_seq_len, batch_size] ) #!!!  [6, 1, -1] 
                batch_target = batch_target.reshape( [target_seq_len, batch_size] )
                print("\n input_seq_len, target_seq_len : ", input_seq_len, target_seq_len )
            except:
                pass ; '''

            """input_lens = [utils.getNzeroSize(tensor) for tensor in batch_input ]
            target_lens = [utils.getNzeroSize(tensor) for tensor in batch_target ]"""

            rl = False #True if (epoch > 1) and ( np.mean(losses_trend)<1.0 and len(losses_trend)>1 ) else False  #!!! and / or 

            loss = trainBert(batch_input,  batch_target,  model, eval_pairs, \
                                    input_lang, output_lang, optimizer, criterion, \
                                      teacher_forcing_ratio = teacher_forcing_ratio, rl=rl )
            plot_losses.append( loss )

            print("\t- the %s batch xentropy loss: "%str(str(epoch)+"."+str(batch)), loss, " " )

            '''if 0 == batch%savepoint and batch > 1:
                print("\n Batch %d savepoint, save the trained model...\n"%batch )
                save_model(encoder, decoder, plot_losses, model_name ) ;'''
        
        losses_trend.append(np.mean(plot_losses))
        plot_losses.clear()

        if epoch > 1 :#and 0 == epoch%5 :
            save_model(model, losses_trend, model_name ) 
            '''if epoch > 5 and 0 == epoch%5 :
                utils.showPlot(losses_trend, model_name, "epoch"+str(epoch) )'''
            print("\n Finish Epoch %d -- model saved. \n "%epoch ); #!!!


def save_model(model, plot_losses, model_name ):
    stamp = str(time.time())
    savepath = utils.prepare_dir( model_name, stamp)
    torch.save(model.state_dict(), savepath+"/%s.model"%stamp )
    try:
        utils.showPlot(plot_losses, model_name, stamp ) 
    except:
        pass ;
    print(" * model save with time stamp: ", stamp )




        