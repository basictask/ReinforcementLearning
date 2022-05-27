# Homework1 - Daniel Kuknyo
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:31:55 2022

@author: Daniel Kuknyo
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


#%% Let's define a class to operate the bandit
class bandit:
    def __init__(self, k, eps, iters, rewards):
        ###### Running parameters init ######
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Reward for certain scenarios
        self.rewards = rewards
        
        ###### Inner parameters init ######
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
    def pull(self):
        p = np.random.rand() # Determine chance of exploitation
        
        # a -> action
        if self.eps == 0 and self.n == 0:   # First iteration
            a = np.random.choice(self.k)    
        elif p < self.eps:                  # Exploration
            a = np.random.choice(self.k)
        else:                               # Exploitation
            a = np.argmax(self.k_reward)
        
        # Determine bridge values
        Ajammed = np.random.rand() < rewards['APjam'] # Is bridge A jammed?
        Bjammed = np.random.rand() < rewards['BPjam'] # Is bridge B jammed?
        
        # Give a value to the reward
        reward = 0
        if(a==0): # Bridge A is chosen
            if(Ajammed): #Bridge A is jammed
                reward = rewards['ARewardJam']
            else: # Bridge A is normal
                reward = rewards['ARewardNormal']
        
        elif(a==1): # Bridge B is chosen 
            if(Bjammed): # Bridge B is jammed
                reward = rewards['BRewardJam']
            else: # Bridge B is normal
                reward = rewards['BRewardNormal']
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total reward
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
                
    def run(self):
        for i in range(self.iters):
            self.pull()
            self.reward[i] = self.mean_reward
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
        

#%% Function to run the learning
def run_learning(k, eps, iters, episodes, rewards, eps_rewards):
    for i in range(episodes):
        b = bandit(k, eps, iters, rewards)
        
        b.run()
        
        eps_rewards = eps_rewards + (b.reward - eps_rewards) / (i + 1)
    return eps_rewards


#%% Define the Hyperparameters
APjam = 0.18
ARewardJam = 49
ARewardNormal = 9

BPjam = 0.29
BRewardJam = 46
BRewardNormal = 12

# Create a Data structure for them 

rewards = {'APjam': APjam, 'ARewardJam': ARewardJam, 'ARewardNormal': ARewardNormal,
           'BPjam': BPjam, 'BRewardJam': BRewardJam, 'BRewardNormal': BRewardNormal}


#%% Run learning with multiple epsilon strategies
# Define hyperparams
k=2
iters = 1000
episodes = 1000
eps_rewards = np.zeros(iters)

# Define logging data structures 
eps_to_try = [x/10 for x in range(0,12,2)]
results = {x:[] for x in eps_to_try}

# Run the learning
for eps in eps_to_try:
    result = run_learning(k, eps, iters, episodes, rewards, eps_rewards)
    results[eps] = result


#%% Plot the run
plt.figure(figsize=(10,8))

for eps in results.keys():
    plt.plot(results[eps], label="$\epsilon="+str(eps))
    
plt.legend(bbox_to_anchor=(1.3, 0.5))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards after " + str(episodes) + " Episodes")
plt.show()
