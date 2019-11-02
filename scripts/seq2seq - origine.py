# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:36:35 2019

@author: Stuart
"""
import neural_layers as nl
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import torch 
import torch.nn.utils.rnn as rnn_utils
import argparse



class Seq2SeqModel:
    def __init__(self, output_lang, VOCAB_SIZE):
        hidden_size = 256
        #self.encoder = nl.BERTEncoder.from_pretrained('bert-base-multilingual-cased', num_labels=hidden_size ).to(nl.device) 
        #self.encoder = nl.BERTEncoder.from_pretrained('bert-base-uncased', num_labels=hidden_size ).to(nl.device) 
        self.encoder = nl.BERTEncoder.from_pretrained('bert-large-cased', num_labels=hidden_size ).to(nl.device) 
        self.encoder.get_embedding(VOCAB_SIZE)
        #self.decoder = nl.DecoderRNN(hidden_size = 256, output_size = output_lang.n_words).to(nl.device) 
        self.decoder = nl.AttnDecoderRNN(hidden_size = 256, output_size = output_lang.n_words).to(nl.device) 
        

    def trainItersBert(self, training_pairs, eval_pairs, input_lang, output_lang):
        nl.trainItersBert(self.encoder, self.decoder, 75000, training_pairs, eval_pairs, input_lang, output_lang, print_every=500, learning_rate=0.01, mom=0.001) 
        
    def translate(self, sentence, training_ans, eval_pairs, input_lang, output_lang):
        output_words, attentions = nl.evaluate(self.encoder, self.decoder, sentence, training_ans, input_lang, output_lang )
        plt.matshow(attentions.numpy())
        return output_words, attentions 
    
    def load_model(self, load_model_name ):
        enc_path, dec_path = "./model/qald-test/%(model_name)s/%(model_name)s.encoder"%{"model_name":load_model_name} , "./model/qald-test/%(model_name)s/%(model_name)s.decoder"%{"model_name":load_model_name} 
        self.encoder.load_state_dict(torch.load(enc_path))
        self.encoder.eval()
        self.decoder.load_state_dict(torch.load(dec_path)) 
        self.decoder.eval()
        print("\n ...model loaded.")
        
    
      
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description=" Neural Knowledge-graph QA Model Based On Pretrained BERT & Reinforcement Learning ")
    parser.add_argument('--train_dataset', type=str, default="./data/qald9/dataset.txt")
    parser.add_argument('--eval_dataset', type=str, default="./data/qald9/train.txt")
    parser.add_argument('--train', type=str, default = None)
    parser.add_argument('--pretrain', type=str, default = None)
    parser.add_argument('--translate', type=str, default = None)
    parser.add_argument('--model_name', type=str, default = None)
    args = parser.parse_args()
     
    training_fp = args.train_dataset
    eval_fp = args.eval_dataset 
    train = args.train
    load_model_name = args.pretrain
    sentence = args.translate
    model_name = args.model_name if args.model_name is not None else str().join(str(training_fp.split("/")[-1]).split(".")[:-1] )
    
    input_lang, output_lang, training_pairs, eval_pairs, VOCAB_SIZE = utils.prepareData(training_fp, eval_fp, False)
    '''
    eval_tensors = [utils.tensorsFromPair(pair, input_lang, output_lang, nl.device) for pair in eval_pairs ] 
    eval_inputs = [ tensors[0] for tensors in eval_tensors ]
    eval_targets = [ tensors[1] for tensors in eval_tensors ]
    
    eval_inputs  = rnn_utils.pad_sequence(eval_inputs, batch_first=True, padding_value=0)
    eval_targets = rnn_utils.pad_sequence(eval_targets, batch_first=True, padding_value=0)
    
    torch.save(eval_inputs, "./model/eval_inputs.pt")
    torch.save(eval_targets, "./model/eval_targets.pt")'''

    seq2seq = Seq2SeqModel(output_lang, VOCAB_SIZE)

    if load_model_name is not None:
        seq2seq.load_model(load_model_name) ; 
    
    if train is not None:
        seq2seq.trainItersBert(training_pairs, eval_pairs, input_lang, output_lang)
    
        
    if train or load is not None:
        seq2seq.translate(sentence, input_lang, output_lang)
    
    
    
    