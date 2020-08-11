# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:27:14 2019

@author: Stuart C 
"""
import spacy
import numpy as np 
import re 
import string 
import pickle 
import ast
import torch 
import generator_utils

#nlp = spacy.load("en_core_web_sm")
glove = spacy.load('en_vectors_web_lg') 
criteria = [ line.strip("\n").strip()  for line in open("../rl_utils/bin/criteria.en", "r", encoding="UTF-8").readlines()]
tgt_dict = pickle.load(open("../rl_utils/bin/tgt_dict.pkl", "br"))


def get_sem_reward(cand_sents_tensor, ref_sents_tensor, dictionary=tgt_dict ):
     cand_sents = get_sentences(cand_sents_tensor, dictionary)
     ref_sents = get_sentences(ref_sents_tensor, dictionary)
     sents_pairs = list(zip(cand_sents, ref_sents))
     scores = []
     for cand_sent, ref_sent in sents_pairs :
         score = calculate_reward(cand_sent, ref_sent)
         if score is not None:
             scores.append(score)
     reward_smoothed = np.mean(scores)
     return reward_smoothed, sents_pairs ;


def calculate_reward(cand_sent, ref_sent ):
    cand_simile = get_simile(cand_sent)
    ref_simile = get_simile(ref_sent) 
    num = np.inner(cand_simile, ref_simile)
    #print("\n cand_simile : ", np.linalg.norm(cand_simile))
    #print("\n num = ", num )
    #print("\n ref_simile :",np.linalg.norm(ref_simile))
    #print("\n s = ", ref_simile, cand_simile)
    s = (np.linalg.norm(cand_simile) * np.linalg.norm(ref_simile))
    if 0 == s:
        #print("\n got zeros.")
        return None  
    else:
        cos_sim = 2.0 - (1+ num/s)/2.0
        return cos_sim ; 

def cosine(vec1, vec2):
    num = np.inner(vec1, vec2)
    s = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if 0 == s:
        #print("\n got zeros.")
        return None  
    else:
        cos_sim = 1+ (num/s)
        return cos_sim ;
#def cos_sim(a, b):
#    cos_sim = np.inner(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))+1.0
#    print(cos_sim)
#    return cos_sim;
#
#cs = cos_sim([0,1], [1,0])

def get_simile(query):
    sentence = preprocess_sentence(query)
    head_var = get_head_var(sentence)
    triplelist = capture(sentence)
    allsdict = build_sem_graph(triplelist)
    if head_var is not None:
        try:
            """
            if len(allsdict[head_var]) >1:
                simile = allsdict[head_var]
            else:
                simile = allsdict[head_var][0]"""
            simile = sum_embedding(allsdict[head_var])
        except:
            print("\n zeros from here ")
            return np.zeros([300,])
    else:
        print("\n or from here ")
        simile = np.zeros([300,]) 
        for triple in triplelist:
            trpl = np.ones([300,])
            for ent in triple :
                if len(allsdict[ent]) > 0 :
                    embedding = sum_embedding(allsdict[ent])
                    if len(np.nonzero(embedding)) > 0 :
                        trpl = np.multiply(trpl, embedding)
            simile = np.add(trpl, simile) 
    assert isinstance(simile , np.ndarray )
    assert len(np.nonzero(simile)) > 0  
    #print("* simile: ", simile)
    return simile


def sum_embedding(entity):
    if len(entity) >1:
        simile = np.zeros([300,])
        for embedding in entity:
            if len(np.nonzero(embedding)) > 0 :
                simile += embedding
    else:
        simile = entity[0] 
    return simile 


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
    count = 0    
    while flag and count <= len(allents)*10:       
        for i, triple in enumerate(triplelist):
            print("-- ",i)
            print(triple)
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
                        allsdict[triple[0]].append( glove_embedding(triple[2]) - glove_embedding(triple[1]) )
                        steps.add(i)
                        #print(2.1)
                    elif triple[2].startswith("var") and ( len(allsdict[triple[1]])>0 and len(allsdict[triple[0]]) >0 ) :
                        allsdict[triple[2]].append( glove_embedding(triple[0]) + glove_embedding(triple[1]) ) 
                        steps.add(i)
                        #print(2.2)
        flag = any([len(allsdict[ent])<1 for ent in varents])
        count += 1             
    return allsdict ; 


def get_head_var(query):
    header = list(query.lower().split(" where "))[0]
    if header.startswith("ask "):
        head_var = [ent for ent in header.split() if ent.startswith("var")][0]
        return head_var
    else:
        return None


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
    return glove(entity).vector


def preprocess_entity(strn):
    if "http_//www.w3.org/1999/02/22-rdf-syntax-ns#" in strn:
        strn = strn.replace("http_//www.w3.org/1999/02/22-rdf-syntax-ns#", " ")
    if "_" in strn:
        strn = str(" ").join(strn.split("_")[1:])
    for punc in string.punctuation :
        strn = strn.replace(punc," ")
    entity = str().join([ char if char.islower() else " "+char.lower() for char in strn ]).strip().lower()
    #print("entity : ", entity)
    return entity ; 

    
def preprocess_sentence(strn):
    if "http_//www.w3.org/1999/02/22-rdf-syntax-ns#" in strn:
        strn = strn.replace("http_//www.w3.org/1999/02/22-rdf-syntax-ns#", " ")
    if " where " in strn :
        strn = strn.replace(" where ", " WHERE ")
    return strn.strip() ; 


def get_sentences(tokens_tensor, dictionary):
    sentences = [dictionary.string(tokens.long(), bpe_symbol="@@ ") for tokens in tokens_tensor ] 
    #sentences = [dictionary.string(tokens.long(), bpe_symbol="@@ ") for tokens in torch.Tensor(tokens_tensor) ] #TypeError: expected Float (got Long)
    return sentences 


if __name__ == "__main__":

    fp = "F:/portfolio/GSoC/DBpedia/gsoc2019/bert_rl_qa/bert_rl_qa/data/qald9/train.txt"
    dataset = []
    queriset = []
    with open(fp, "r", encoding="UTF-8") as text:
        for i, line in enumerate(text.readlines()):
            three = line.split("\t")
            query = three[1]
            if not (query.startswith("ASK") or query.startswith("ask")):
                simile = get_simile(query)
                if simile is not None:
                    lis = ast.literal_eval(three[-1])
                    flag = any([ent.startswith("http:") for ent in lis])
                    if flag:
                        queriset.append((simile, i))
                        #lis = ast.literal_eval(three[-1])
                        for ent in lis:
                            if ent.startswith("http:"):
                                ent = preprocess_sentence(generator_utils.encode(ent))
                                vec = glove_embedding(ent)
                                dataset.append((vec, i))
                            else:
                                continue 
    #print("dataset length: ", len(dataset))
    #print("queriset length: ", len(queriset))
    def calcul_rank(queriset, dataset):
        mrrs = []
        hits10 = []
        hits100 = []
        for simile, i in queriset:
            #sorted_dataset = sorted(dataset, key=lambda pair:cosine(simile, pair[0]))
            sorted_dataset = sort_data(simile, dataset)
            def add(num):
                return num+1
            indexs = [ 1/add(n) for n, pair in enumerate(sorted_dataset) if pair[-1]==i]
            h10indexs = [add(n) for n, pair in enumerate(sorted_dataset) if n<10 and pair[-1]==i ]
            h100indexs = [add(n) for n, pair in enumerate(sorted_dataset) if n<100 and pair[-1]==i ] 
            #print("indexs: \n", indexs)
            #print("hindexs: \n", hindexs)
            #print(indexs)
            if len(indexs)>0:
                mrr = indexs[0] #np.mean(indexs)
                hit10 = len(h10indexs) #sum(indexs[:10])
                hit100 = len(h100indexs)
                if mrr is not None:
                    mrrs.append(mrr)
                #if hit is not None:
                hits10.append(hit10)
                hits100.append(hit100)
        print("mrrs: ", np.mean(mrrs))
        #print(len(mrrs))
        #print(mrrs)
        print("hits@10: ", np.mean(hits10))
        print("hits@100: ", np.mean(hits100))
        return mrrs, hits10, hits100 ;

    def sort_data(simile, dataset):
        collected = []
        for vec, i in dataset:
            sim = cosine(simile,vec)
            if sim is not None:
                collected.append((sim,i))
        sorted_dataset = sorted(collected, key=lambda s: s[0])
        return sorted_dataset

    _ = calcul_rank(queriset, dataset)
    


#hits:  0.03651180051424653

"""reciprocal
mrrs:  0.01629881518149452
hits:  0.037618218711647944 """ 
"""
*
TEST:
    mrrs:  0.01629881518149452
    hits@10:  0.08080808080808081
    hits@100:  0.7575757575757576
TRAIN:
    mrrs:  0.0064926602213051755
    hits@10:   0.02877697841726619
    hits@100:  0.28776978417266186
"""

'''not
mrrs:  1450.2222222222222
hits:  6518.717171717171 ''' 

'''
mrrs:  0.0010266127323468434
hits:  10.0 ''' #Separated










