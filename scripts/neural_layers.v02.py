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

torch.manual_seed(1) 

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
        '''print("\n ", input_ids.size())
        print("\n ", token_type_ids.size())
        print("\n ", attention_mask.size())'''
        print("\n ------------ " )
        print("\n input_ids: ", input_ids)
        print("\n token_type_ids: ", token_type_ids)
        print("\n attention_mask: ", attention_mask)
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
        #self.gru = nn.GRU(hidden_size, hidden_size,  ) #batch_first=True #!!! 
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True )
        
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        print("\n input--> ", input.size()) 
        #output = self.embedding(input).view(1, 1, -1)
        output = self.embedding(input).view( 6, -1)
        print(output.size())
        output = F.relu(output) 
        output = output.reshape([1, 6, 256])
        hidden = hidden.reshape([1, 6, 256])
        hidden = hidden[0,0].reshape([1, 1, -1])
        print("\n output--> ", output.size()) 
        print(" hidden--> ", hidden.size(), " \n ")

        output, hidden = self.gru(output, hidden)
        print("\t *output--> ", output.size())
        print("\t *output[0]--> ", output[0].size())
        print("\t *hidden--> ", hidden.size(), " \n ")
        output = self.softmax(self.out(output[0]))
        print("\t *output after softmax--> ", output.size())
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 6, self.hidden_size, device=device) ;#!!!
 
    

   
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
        
        encoder_hidden = encoder(input_tensor)
        
        encoder_hidden = encoder_hidden.unsqueeze(0)
    
        decoder_input = torch.tensor([[utils.SOS_token]], device=device)
    
        decoder_hidden = encoder_hidden
        
        #decoder_hidden_input = decoder_hidden #!!!

        # Without teacher forcing: use its own predictions as the next input
        decoded_words = [] #!

        for di in range(max_length):
            #print(di, " decoder_hidden shape: ", decoder_hidden.size(), " \n ",  decoder_hidden )
            decoder_hidden = decoder_hidden[:, 0, :]
            decoder_hidden = decoder_hidden.view(1,1,256)
            
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #!!!
            if topi.item() == utils.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            #!!!
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if decoder_input.item() == utils.EOS_token:
                break;
                
        decoded_sentence = str(" ").join(decoded_words) 
        print("\n  --query--> ", decoded_sentence, "\n ") 
        
        if (not rl) or (training_ans is None):
            return decoded_sentence 
        else:
            rewrd = reward.get_reward(decoded_sentence, training_ans )
            print("\n  --reward--> ", rewrd)
            return rewrd 
                 
          
    
    
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


def trainBert(input_tensor, target_tensor, encoder, decoder, eval_pairs, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5, rl=True ):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    print("\n\t input_tensor.size(): ", input_tensor.size())
    input_length = input_tensor.size(0)
    print("\t input_length: ", input_length)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    print("\n encoder(input_tensor) : ", input_tensor.size())
    encoder_hidden = encoder(input_tensor)
    print("\n encoder_hidden : ", encoder_hidden.size())

    encoder_hidden = encoder_hidden.unsqueeze(0)

    #for ei in range(input_length):
        #encoder_output, encoder_hidden = encoder(
            #input_tensor[ei], encoder_hidden)
        #encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[utils.SOS_token]], device=device) 
    decoder_input = torch.cat((decoder_input, decoder_input, decoder_input,decoder_input, decoder_input, decoder_input), 0)
    print("\n decoder_input: ", decoder_input.size() )
    #decoder_input = decoder_input.reshape([1,-1,256]) 

    #decoder_hidden = encoder_hidden
    decoder_hidden = encoder_hidden[0].reshape([1, 1, -1]) #!!! 

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    losses = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        print("  * Teacher forcing: ")
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            print("\n\t --- decoder_output: ", decoder_output.size() )
            print("\t --- decoder_output[0]: ", decoder_output[0].size(), "\n\t ", decoder_output[0] )
            print("\t --- target_tensor[%d]: "%di, target_tensor[di].size(), "\n\t ", target_tensor[di] )
            loss += criterion(decoder_output, target_tensor[di])

            loss_ = loss
            v = float(loss_.detach().numpy()/(di+1.0)) 
            losses.append(v)
             
            decoder_input = target_tensor[di]  # Teacher forcing

            if decoder_input.item() == utils.EOS_token: #!!!added
                break

    else:
        # Without teacher forcing: use its own predictions as the next input
        print("  * Not teacher forcing: ")
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

            loss_ = loss
            v = float(loss_.detach().numpy()/(di+1.0))
            losses.append(v) 

            if decoder_input.item() == utils.EOS_token:
                break
    
    if rl and (np.mean(losses) < 1.0) :
        reward_value = evaluateRandomly(encoder, decoder, eval_pairs, input_lang, output_lang, n=10, rl=rl) 
        if 2.0 > reward_value > 1.0:
            loss = torch.mul(loss, torch.LongTensor([reward_value]) ) #torch.mul(loss, torch.FloatTensor([reward_value]) ) #!!! 
    else:
        pass;

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length
    


def trainItersBert(encoder, decoder, n_iters, training_pairs, eval_pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01, mom=0,  model_name="QALD-dev"):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    plot_loss_avg = 1.0 #!!!

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=mom)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=mom)
    
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, amsgrad=True)
    #encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, n_iters)
    #decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, amsgrad=True)
    #decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, n_iters)                                                       

    teacher_forcing_ratio = 1.0

    criterion = nn.NLLLoss()

    '''src_sents, tgt_sents = [], []
    for pair in training_pairs:
        src_sents.append(pair[0])
        tgt_sents.append(pair[1])
    src_size, tgt_size = utils.max_size(src_sents, tgt_sents )'''
    
    #!!!
    input_tensors, target_tensors, train_pairs = [], [], []
    for pair in training_pairs:
        tensors = utils.tensorsFromPair(pair, input_lang, output_lang, device)
        train_pairs.append(tensors)
        '''print("tensor shape--> ", tensors[0].size())
        print(tensors[0])'''
        input_tensors.append(tensors[0].view(-1,1).long()) #float() #!!! 
        target_tensors.append(tensors[1].view(-1,1).long()) #!!! 
    print("\n Dataset preparing... ")

    """input_tensors, target_tensors = torch.Tensor(len(training_pairs), 1, 1, 256), torch.Tensor(len(training_pairs), 1, 1, 256)
    torch.cat(en_tensors, out=input_tensors)
    torch.cat(sparql_tensors, out=target_tensors)"""

    '''print(" assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors) ")
    print(input_tensors[0].size(0), )'''

    input_tensors  = rnn_utils.pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensors  = rnn_utils.pad_sequence(target_tensors, batch_first=True, padding_value=0)
    #input_tensors, target_tensors = utils.padding(input_tensors, target_tensors )
    torch_dataset = utils.TxtDataset(input_tensors, target_tensors  )
    
    # put the dataset into DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=6,  # MINIBATCH_SIZE
        shuffle=True,
        #num_workers=1           # set multi-work num read data
        #collate_fn= utils.collate_fn  #!!! 
    ) 
    print(" Dataset loader ready, begin training. \n")

    for epoch in range(1, n_iters + 1):
    # 1 epoch go the whole data
        for step, (batch_input, batch_target) in enumerate(loader):
            # here to train your model
            print('\n\n  - epoch: ', epoch, ' | step: ', step, '\n | batch_input: \n', batch_input.size(), '\n | batch_target: \n', batch_target.size() ) 
            
            #input_tensor, target_tensor = batch_input, batch_target  #!!! 

            batch_input = batch_input.reshape( [6,  -1, 1] ) #!!!  [6, 1, -1] 
            batch_target = batch_target.reshape( [6,  -1, 1] )
            print("\n input_batch : ", batch_input.size())
            print("\n target_batch : ", batch_target.size())

            '''loss = trainBert(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion) '''
            rl = True if (epoch > 1) and (np.mean(plot_losses) < 1.0 ) else False 

            '''loss = 0.0
            for i in range(6):
                input_tensor, target_tensor = batch_input[i], batch_target[i]
                print("\n input_tensor : ", input_tensor.size() )
                print("\n target_tensor : ", target_tensor.size() )
                loss += trainBert(input_tensor, target_tensor, encoder, decoder, eval_pairs, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio = teacher_forcing_ratio, rl=rl )
            plot_losses.append( loss/6 )
            '''
            loss = 0
            for batch_input_, batch_target_ in zip(batch_input, batch_target):
                loss += trainBert(batch_input, batch_target, encoder, decoder, eval_pairs, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio = teacher_forcing_ratio, rl=rl )
            
            plot_losses.append( loss/6 )

            print("\t -  %s step xentropy loss: "%str(epoch), loss, " \n" )

            teacher_forcing_ratio = utils.teacher_force(float(loss) ) ;

            

"""
    train_pairs = [utils.tensorsFromPair(random.choice(training_pairs), input_lang, output_lang, device)  for i in range(n_iters) ]
    for iter in range(1, n_iters + 1):
        #encoder_scheduler.step()
        #decoder_scheduler.step()
        
        train_pair = train_pairs[iter - 1]
        input_tensor = train_pair[0]
        target_tensor = train_pair[1]
        input_tensor.transpose_(0,1)
        #print(input_tensor.size())

        '''loss = trainBert(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion) '''
        rl = True if (iter > 1000) and (plot_loss_avg < 1.0 ) else False 
        
        loss = trainBert(input_tensor, target_tensor, encoder, decoder, eval_pairs, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio = teacher_forcing_ratio, rl=rl )
        print_loss_total += loss
        plot_loss_total += loss

        print("\t -  %s step xentropy loss: "%str(iter), loss, " \n" )

        teacher_forcing_ratio = utils.teacher_force(float(loss) )

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if ( 0==iter%print_every or (n_iters + 1)==iter ):# and iter > 1 :
            save_model(encoder, decoder, plot_losses, model_name ) ;
"""

def save_model(encoder, decoder, plot_losses, model_name ):
    stamp = str(time.time())
    savepath = utils.prepare_dir( model_name, stamp)
    torch.save(encoder.state_dict(), savepath+"/%s.encoder"%stamp )
    torch.save(decoder.state_dict(), savepath+"/%s.decoder"%stamp )
    try:
        utils.showPlot(plot_losses) 
    except:
        pass ;




        