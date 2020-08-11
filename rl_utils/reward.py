# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:28:12 2019

@author:  
""" 
import torch 
import numpy as np 
import random 
import rl_utils
#import rl_utils.kgutils as kgutils
#import rl_utils.queries as queries
#import rl_utils.sem_reward as sem_reward   

def get_reward(cand_sents_tensor, ref_sents_tensor):
    sem_reward, sents_pairs = rl_utils.sem_reward.get_sem_reward(cand_sents_tensor, ref_sents_tensor) 
    print("sem_reward : ", sem_reward)
    if sem_reward >= 2 :
        ans_rewards = []
        for _ in range(10):
            cand_sent, ref_sent = random.choice(sents_pairs)
            rewrd = rl_utils.ans_reward.get_ans_reward(cand_sent, ref_sent)
            if rewrd is not None :
                ans_rewards.append(rewrd) 
                
        if len(ans_rewards) > 0 :
            ans_reward = np.mean(ans_rewards)+1.0 
            reward_value = sem_reward if ans_reward == 2 else ans_reward 
            print("ans_reward : ", ans_reward )
            return reward_value
        else :
            return sem_reward
    else:
        return sem_reward ;  




"""
if __name__ == "__main__" : 

    prev_output = np.load("../translations/seethetensors/1574693463.0723605.samp_prev_output_tokens.np.npy") 
    prev_output_tokens = torch.Tensor(prev_output, )

    tgts = np.load("../translations/seethetensors/1574693463.0723605.samp_target.np.npy") 
    targets = torch.Tensor(tgts)   

    rewrd = get_reward(prev_output_tokens, targets )
    print(rewrd)
    print("\nThe reward is {}".format(rewrd))
"""

