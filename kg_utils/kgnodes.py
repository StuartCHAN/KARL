# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:56:03 2019

@author: Stuart
"""
import kgutils 

replacements = [ str(pair[-1]).strip() for pair in kgutils.REPLACEMENTS if (len(str(pair[-1]).strip()) > 1) and ("_" in str(pair[-1]).strip().strip("_")) ]

criteria = [ str(pair[-1]).strip() for pair in kgutils.REPLACEMENTS if (str(pair[-1]).strip() not in replacements) and (len(str(pair[-1]).strip()) > 1) ]
criteria.append("onto_")
criteria.append('"')
criteria.append('dbpedia2_')
criteria.append('http')


with open("F:/portfolio/GSoC/DBpedia/neural-qa/data/DBNQA/DBNQA/bertnmt/criteria.en", "w", encoding="UTF-8") as criti:
    for prefix in criteria :
        criti.write(str(prefix)+"\n")
        
################################ 
        
enfp = "F:/portfolio/GSoC/DBpedia/neural-qa/data/DBNQA/DBNQA/bertnmt/valid.en" 

sfp = "F:/portfolio/GSoC/DBpedia/neural-qa/data/DBNQA/DBNQA/bertnmt/valid.sparql" 

en = open(enfp,"r",encoding="UTF-8").readlines() 

sp = open(sfp,"r",encoding="UTF-8").readlines() 

criteria =[ line.strip("\n").strip() for line in open("F:/portfolio/GSoC/DBpedia/neural-qa/data/DBNQA/DBNQA/bertnmt/criteria.en", "r", encoding="UTF-8").readlines()]


import re

#tokenset = set()
#_ = [ [ tokenset.add(token) for token in sent.split() if not(token.startswith("db") or token.startswith("var") or token.startswith("onto")) ] for sent in sp ]
#tokens = list(tokenset)

def capture(query):
    conds = re.findall(r"WHERE brack_open(.+)brack_close", query)
    conditions = [replacement(cond) for cond in conds ]
    phrases = [[ filtering(triple) for triple in cond.split("@@") if len(filtering(triple)) >1 or (triple.strip().startswith("db") or triple.strip().startswith("var"))]  for cond in conditions]
    triples = padtriples(phrases[0]) 
    return triples  

def replacement(sentence):
    for token in sentence.split():
        if (not any([token.startswith(prefix) for prefix in criteria ])) and (token is not "a"):
            sentence = sentence.replace(token, " @@ ")
    return sentence 

def filtering(triple):
    collected = []
    for enti in triple.split():
        if any([enti.startswith(token) for token in criteria]) or enti == "a":
            collected.append(enti) 
    return collected 
            

       
'''def pad_triple(triplelist):
    padded = []
    for i, triple in enumerate(triplelist):
        if 2==len(triple) and i>0 :
#            if str(triple[-1]).startswith("var") and str(triple[0]).startswith("db"):
#                triple.insert(0, triplelist[i-1][0])
#            elif (str(triple[-1]).startswith("var") and not str(triple[0]).startswith("db")) or \
#                (str(triple[0]).startswith("var") and not str(triple[-1]).startswith("db")):
#                triple.insert(1, "a")
#            else:
            triple.insert(0, triplelist[i-1][0]) 
        elif 3<len(triple):
            triple = [element.strip() for element in triple if element.startswith("'") or element.startswith('"')]
        else:
            pass;
        #if len(triple) == 3 :
        padded.append(triple)
    print(padded)
    return padded ;'''

def padtriples(triplelist):
    padded = []
    for i, triple in enumerate(triplelist):
        if 2==len(triple) and i>0 :
            triple.insert(0, triplelist[i-1][0]) 
        if len(triple) == 3: 
            padded.append(triple)
    #print(padded)
    return padded ;


           
entities = [capture(query) for query in sp ]

outliers = []
for i,sent in enumerate(entities):
    for triple in sent:
        if 3 != len(triple):
            outliers.append((i, triple))


conds = re.findall(r"WHERE brack_open(.+)brack_close", sp[36])#92
conditions = [replacement(cond) for cond in conds ]
phrases_ = [[ triple for triple in cond.split("@@") if len(filtering(triple)) >1 and len(triple.strip())>1 ] for cond in conditions]
phrases = [[ filtering(triple) for triple in cond.split("@@") if len(filtering(triple)) >1 or (triple.strip().startswith("db") or triple.strip().startswith("var")) ] for cond in conditions]

pad_triple(phrases[0])

phrases_ = [[ triple for triple in cond.split("@@") if len(filtering(triple)) >1 or (triple.strip().startswith("db") or triple.strip().startswith("var")) ] for cond in conditions]


outliers=[]
for triples in entities:
    for triple in triples:
        if 3 != len(triple):
           outliers.append(triple) 
           
lengthiers=[]
maxl = max([len(triples) for triples in entities])
for triples in entities:
    if maxl == len(triples):
        lengthiers.append(triples) 


class Node:
    def __init__(self, entity):
        self.entity = entity
        self.indegrees = []
        self.outdegrees = []
        self.outedges = []
        self.var = True if str(entity).startswith("var") else False 
    def add_in(self,innode,inedge):
        self.indegrees.append((innode,inedge))
    def add_out(self, outnode, outedge):
        self.outdegrees.append((outnode, outedge)) 
 

import spacy
#nlp = spacy.load("en_core_web_sm")
glove = spacy.load('en_vectors_web_lg')

import numpy as np 

def glove_embedding(string):
    return glove(preprocess(string)).vector 

def preprocess_entity(string):
    return string
            
#def build_graph(triplelist, core_var):
#    core_node = Node(core_var)
triplelist = lengthiers[0]
#triplelist = list(set(triplelist))
varents = list()
allents = list()
varsdict = {}
allsdict = {}
for triple in triplelist:   
    if triple[0].startswith("var"):
        ent = triple[0]
        varents.append(ent)
        varsdict[str(ent)] = list()
    if triple[-1].startswith("var"):
        ent = triple[-1]
        varents.append(ent)
        varsdict[str(ent)] = list()
    for ent in triple:
        allents.append(ent)
        allsdict[str(ent)] = list()

varents = set(varents)
allents = set(allents) 
        
_ = [allsdict[ent].append(glove(ent)) for ent in allents-varents ]

flag = any([len(allsdict[ent])<1 for ent in varents]) 

steps = set() 
    
while flag :       
    for i, triple in enumerate(triplelist):
        print("-- ",i)
        if i not in steps:
            if "a" in triple or "rdf_type" in triple:
                if triple[0].startswith("var") and len(allsdict[triple[-1]]) >0:
                    allsdict[triple[0]].append( allsdict[triple[-1]][0])
                    steps.add(i)
                    print(1.1)
                elif triple[-1].startswith("var") and len(allsdict[triple[0]]) >0:
                   allsdict[triple[-1]].append( allsdict[triple[0]][0])
                   steps.add(i)
                   print(1.2)
            else:
                if triple[0].startswith("var") and ( len(allsdict[triple[1]])>0 and len(allsdict[triple[2]]) >0 ) :
                    allsdict[triple[0]].append( glove(preprocess(triple[2])) - glove(preprocess(triple[1])) )
                    steps.add(i)
                    print(2.1)
                elif triple[2].startswith("var") and ( len(allsdict[triple[1]])>0 and len(allsdict[triple[0]]) >0 ) :
                    allsdict[triple[2]].append( glove(preprocess(triple[0])) + glove(preprocess(triple[1])) ) 
                    steps.add(i)
                    print(2.2)
        flag = any([len(allsdict[ent])<1 for ent in varents]) 
   
           
value = sum(allsdict["var_uri"])                

heads = list(set([query.split()[0] for query in sp]))              

            
_ = [print(len(allsdict[var])) for var in allsdict.keys() ]        
        
    
_ = [[varents.add(ent) for ent in triple if ent.startswith("var")] for triple in triplelist ]
varents = list(varents)
nodesdict = {}
for var in varents :
    nodesdict[str(var)] = set()
while not all([ len(nodesdict[str(var)]) >=1 for var in varents]):
    for triple in triplelist:


################################### 
        
import spacy
import numpy as np 
import re 
import string 
import torch 

#nlp = spacy.load("en_core_web_sm")
glove = spacy.load('en_vectors_web_lg') 
criteria =[ line.strip("\n").strip() for line in open("F:/portfolio/GSoC/DBpedia/neural-qa/data/DBNQA/DBNQA/bertnmt/criteria.en", "r", encoding="UTF-8").readlines()]



def get_reward(cand_sents_tensor, ref_sents_tensor, dictionary ):
     cand_sents = get_sentences(cand_sents_tensor, dictionary)
     ref_sents = get_sentences(ref_sents_tensor, dictionary)
     reward_smoothed = 2.0 - np.mean([ calculate_reward(cand_sent, ref_sent) for cand_sent, ref_sent in zip(cand_sents, ref_sents) ])
     return reward_smoothed

def calculate_reward(cand_sent, ref_sent ):
	cand_simile = get_simile(cand_sent)
    ref_simile = get_simile(ref_sent)
    cos_sim = np.inner(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
	return cos_sim ; 

def get_simile(query):
    sentence = preprocess_sentence(query)
    head_var = get_head_var(sentence)
    triplelist = capture(sentence)
    allsdict = build_sem_graph(triplelist)
    simile = allsdict[head_var]
    assert isinstance(simile[0] , np.ndarray )
    return simile[0] 
    
    
def build_sem_graph(triplelist):
    varents = list()
    allents = list()
    varsdict = {}
    allsdict = {}
    for triple in triplelist:   
        if triple[0].startswith("var"):
            ent = triple[0]
            varents.append(ent)
            varsdict[str(ent)] = list()
        if triple[-1].startswith("var"):
            ent = triple[-1]
            varents.append(ent)
            varsdict[str(ent)] = list()
        for ent in triple:
            allents.append(ent)
            allsdict[str(ent)] = list();
    varents = set(varents)
    allents = set(allents)         
    _ = [allsdict[ent].append(glove_embedding(ent)) for ent in allents-varents ]
    flag = any([len(allsdict[ent])<1 for ent in varents]) 
    steps = set()     
    while flag :       
        for i, triple in enumerate(triplelist):
            #print("-- ",i)
            if i not in steps:
                if "a" in triple or "rdf_type" in triple or "type" in triple:
                    if triple[0].startswith("var") and len(allsdict[triple[-1]]) >0:
                        allsdict[triple[0]].append( allsdict[triple[-1]][0])
                        steps.add(i)
                        #print(1.1)
                    elif triple[-1].startswith("var") and len(allsdict[triple[0]]) >0:
                       allsdict[triple[-1]].append( allsdict[triple[0]][0])
                       steps.add(i)
                       #print(1.2)
                else:
                    if triple[0].startswith("var") and ( len(allsdict[triple[1]])>0 and len(allsdict[triple[2]]) >0 ) :
                        allsdict[triple[0]].append( glove_embedding(preprocess(triple[2])) - glove_embedding(preprocess(triple[1])) )
                        steps.add(i)
                        #print(2.1)
                    elif triple[2].startswith("var") and ( len(allsdict[triple[1]])>0 and len(allsdict[triple[0]]) >0 ) :
                        allsdict[triple[2]].append( glove_embedding(preprocess(triple[0])) + glove_embedding(preprocess(triple[1])) ) 
                        steps.add(i)
                        #print(2.2)
        flag = any([len(allsdict[ent])<1 for ent in varents])             
    return allsdict ; 


def get_head_var(query):
    head_var = [ent for ent in query.lower().split(" WHERE ")[0].split() if ent.startswith("var")][0]
    return head_var 


def capture(query):
    if " WHERE " in query: 
        conds = re.findall(r" WHERE brack_open(.+)brack_close", query)
    else:
        conds = re.findall(r" where brack_open(.+)brack_close", query)
    #print("conds = ", conds )
    conditions = [replacement(cond) for cond in conds ]
    phrases = [[ filtering(triple) for triple in cond.split("@@") if len(filtering(triple)) >1 or (triple.strip().startswith("db") or triple.strip().startswith("var"))]  for cond in conditions ]  
    triples = padtriples(phrases[0]) 
    return triples 
    
     
def replacement(sentence):
    #print(sentence)
    for token in sentence.split():
        if (not any([token.startswith(prefix) for prefix in criteria ])) and (token is not "a"):
            sentence = sentence.replace(token, " @@ ")
    return sentence 


def filtering(triple):
    collected = []
    for enti in triple.split():
        if any([enti.startswith(token) for token in criteria]) or enti == "a":
            collected.append(enti) 
    return collected 
            

def padtriples(triplelist):
    padded = []
    for i, triple in enumerate(triplelist):
        if 2==len(triple) and i>0 :
            triple.insert(0, triplelist[i-1][0]) 
        if len(triple) == 3: 
            padded.append(triple)
    #print(padded)
    return padded ;                    


def glove_embedding(strn):
    entity = preprocess_entity(strn)
    return np.ndarray(glove(preprocess_entity(entity)).vector)


def preprocess_entity(strn):
    if "http_//www.w3.org/1999/02/22-rdf-syntax-ns#" in strn:
        strn = strn.replace("http_//www.w3.org/1999/02/22-rdf-syntax-ns#", " ")
    if "_" in strn:
        strn = strn.split("_")[-1]
    for punc in string.punctuation :
        strn = strn.replace(punc," ")
    entity = str().join([ char if char.islower() else " "+char.lower() for char in strn ]).strip().lower()
    return entity ; 

    
def preprocess_sentence(strn):
    if "http_//www.w3.org/1999/02/22-rdf-syntax-ns#" in strn:
        strn = strn.replace("http_//www.w3.org/1999/02/22-rdf-syntax-ns#", " ")
    if " where " in strn :
        strn = strn.replace(" where ", " WHERE ")
    return strn ; 
    

def get_sentences(tokens_tensor, dictionary):
    sentences = [dictionary.string(tokens.long(), bpe_symbol="@@ ") for tokens in torch.LongTensor(tokens_tensor) ]  
    return sentences 


######################################## 
import string
    
def preprocess(strn):
    if "http_//www.w3.org/1999/02/22-rdf-syntax-ns#" in strn:
        strn = strn.replace("http_//www.w3.org/1999/02/22-rdf-syntax-ns#", " ")
    if "_" in strn:
        strn = strn.split("_")[-1]
    for punc in string.punctuation :
        strn = strn.replace(punc," ")
    strn = str().join([ char if char.islower() else " "+char for char in strn ]).strip().lower()
    return strn ; 


import pickle 
import numpy as np 
import torch 

dfp = "./translations/seethetensors/src_dict.pkl"

src_dict = pickle.load(open("./translations/seethetensors/src_dict.pkl", "br"))   

tgt_dict = pickle.load(open("./translations/seethetensors/tgt_dict.pkl", "br"))
            
prev_output = np.load("./translations/seethetensors/1574693463.0723605.samp_prev_output_tokens.np.npy") 

tgts = np.load("./translations/seethetensors/1574693463.0723605.samp_target.np.npy") 
    
prev_output_tokens = torch.Tensor(prev_output, )                            
        
output_tokens = [tgt_dict.string(tokens.to(dtype=torch.long, device="cpu"), bpe_symbol="@@ ") for tokens in prev_output_tokens ]   
#to(dtype=torch.long, device="cpu")
targets = [tgt_dict.string(tokens.long(), bpe_symbol="@@ ") for tokens in torch.Tensor(tgts) ]   

bpe_symbol = pickle.load(open("./translations/seethetensors/bpe_symbol.pkl", "br"))

for char in bpe_symbol:
    print("-"+char+"-")

def get_sentences(tokens_tensor, dictionary):
    sentences = [dictionary.string(tokens.long(), bpe_symbol="@@ ") for tokens in torch.LongTensor(tokens_tensor) ]  
    return sentences 

sentences = get_sentences(tgts, tgt_dict ) 

sentences_cap = [capture(sentence) for sentence in output_tokens ]

    