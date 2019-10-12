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

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

import kg_utils.reward as reward 

import utils
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n device: ", device)

print("\n torch version ", torch.__version__ )




# Define the Encoder
def swish(x):
    return x * torch.sigmoid(x)

class BERTEncoder(BertForSequenceClassification):
    def __init__(self, config, num_labels=2):
        super(BERTEncoder, self).__init__(config, num_labels)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden_size = config.hidden_size #!
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        
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
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device) ;
 
    

   
# Evaluation for the Model after Training 
def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=utils.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = utils.tensorFromSentence(input_lang, sentence, device )
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[utils.SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == utils.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1] 
    
    
def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=10):
    output_sentences = []
    rew = 0.0
    for i in range(n):
        pair = random.choice(pairs)
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        #print('<', output_sentence)
        #output_sentences.append(output_sentence)
        #print('')
        rew += reward.get_reward(output_sentence, pair[-1])
    rewrd = rew/n
    return output_sentences, rewrd ; 



# Training the Model 
teacher_forcing_ratio = 0.5

def trainBert(input_tensor, target_tensor, encoder, decoder, training_ans, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion, max_length=utils.MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    
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
            print(" * %s step: "%str(di), loss.detach() ) #!
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
            print("\n --query--> ", decoded_sentence, "\n ")
            rew = reward.get_reward(decoded_sentence, training_ans )
            rewards.append(rew)
            print("\n --reward--> ", rew)
            loss += criterion(decoder_output, target_tensor[di])
            print(" * %s step: "%str(di), loss.detach() )#!
            print("\n ")#!
            if decoder_input.item() == utils.EOS_token:
                break;
               
    #_, rewrd = evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=target_length)
    loss = loss*torch.FloatTenso([np.mean(rewards)])
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length 



def trainItersBert(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100, learning_rate=0.01, mom=0):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=mom)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate, momentum=mom)
    
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, amsgrad=True)
    #encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, n_iters)
    #decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, amsgrad=True)
    #decoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, n_iters)                                                       
                                                       
    training_pairs = []
    training_answers = []#!
    for _ in range(n_iters):
        pair = random.choice(pairs)
        training_pairs.append( utils.tensorsFromPair(pair[:-1], input_lang, output_lang, device) )
        training_answers.append(pair[-1])#!

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        #encoder_scheduler.step()
        #decoder_scheduler.step()
        print("\n ----- %s Epoch ----- "%str(iter) )
        training_pair = training_pairs[iter - 1]
        training_ans = training_answers[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        input_tensor.transpose_(0,1)
        #print(input_tensor.size())

        loss = trainBert(input_tensor, target_tensor, encoder,
                     decoder, training_ans, input_lang, output_lang, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
    stamp = str(time.time())
    torch.save(encoder.state_dict(), "./model/%s.encoder"%stamp )
    torch.save(decoder.state_dict(), "./model/%s.decoder"%stamp )
    utils.showPlot(plot_losses) ;


        