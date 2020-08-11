# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:08:37 2019

@author: Stuart
"""
import numpy as np
import rl_utils
import rl_utils.kgutils as kgutils
import rl_utils.queries as queries


def get_ans_reward(cand_sent, ref_sent):
    try:
    	cand_query = kgutils.interprete(cand_sent)
    	cand_ents = queries.process(cand_query)
    	ref_query = kgutils.interprete(ref_sent)
    	ref_ents = queries.process(ref_query)
    	rewrd = criterion_func(cand_ents, ref_ents)
    	return rewrd
    except:
        return None ;


def criterion_func(cand_ents, ref_ents):
    notempty = "empty" not in (cand_ents or ref_ents)
    noterror = "error" not in (cand_ents or ref_ents)
    if notempty and noterror:
        if len(set(cand_ents)&set(ref_ents)) is not 0 :
            return -0.01 
        else:
            return 1
    elif (not notempty) and noterror :
        return -0.01 
    else:
        return 1 ;
    
    

"""
    pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
	neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
	pos_y = self.get_positive_labels(in_batch = True)
	neg_y = self.get_negative_labels(in_batch = True)
		
	p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
	p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
	p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
	n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
	n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
	n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
	_p_score = self._calc(p_h, p_t, p_r)
	_n_score = self._calc(n_h, n_t, n_r)
	print (_n_score.get_shape())
	loss_func = tf.reduce_mean(tf.nn.softplus(- pos_y * _p_score) + tf.nn.softplus(- neg_y * _n_score))
	regul_func = tf.reduce_mean(p_h ** 2 + p_t ** 2 + p_r ** 2 + n_h ** 2 + n_t ** 2 + n_r ** 2) 
	self.loss =  loss_func + config.lmbda * regul_func
"""


    
    

