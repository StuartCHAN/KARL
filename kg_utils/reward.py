# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:08:37 2019

@author: Stuart
"""
import numpy as np
import kg_utils
import kg_utils.kgutils as kgutils
import kg_utils.queries as queries
 

from sematch.semantic.similarity import EntitySimilarity
sim = EntitySimilarity()


def get_reward(cand_sent, ref_ent):
    try:
    	cand_query = kgutils.interprete(cand_sent)
    	entity = queries.process(cand_query)
    	if entity == ref_ent:
    		return 1.0
    	elif entity is "empty":
    		return 1.0
    	elif entity is "error":
    		return 2.0
    	else: 
    		reward = calculate_reward(entity, ref_ent ) 
    		return reward  
    except:
        return 2.0 ;


def calculate_reward(cand_ent, ref_ent ):
	rewrd = 2.0
	try:
		rewrd = 2.0-sim.similarity(cand_ent, ref_ent )
	except:
		rewrd = 2.0
	return rewrd ;

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


    
    

