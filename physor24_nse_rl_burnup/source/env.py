# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:48:34 2023

@author: Majdi Radaideh
"""

import gym
from gym.spaces import Box, MultiDiscrete, Discrete, Dict
import numpy as np
import csv
from tensorflow.keras.models import load_model
from pathlib import Path

#--------------------------------------
#RL environment
#--------------------------------------
class MicroReactorEnv(gym.Env):
  def __init__(self, surr_path, casename='mycase', action_type='cont'):
    """
    surr_path (str): The path to the surrogate model files of the Serpent environment 
    casename (str): a string used to identify files and paths
    action_type (str): Action space type. Either 'cont' (continuous) or 'disc' (discrete). For now only 'disc' is suppoerted
    """
    #Inputs to class:
    self.action_type=action_type
    self.episode_length=3
    self.epoch_length= 300
    self.state_mode='random'
    
    if self.action_type=='cont':
      self.action_space = Box(low=-1, high=1, shape=(6,))
    else:
      self.action_space = MultiDiscrete([181, 181, 181, 181, 181, 181])
    
    spaces = {'time': Discrete(3),
              'power': Box(low=0, high=1, shape=(6,)),
              'keff': Box(low=0.8, high=1.2, shape=(1,))
              }
    self.observation_space = Dict(spaces)
                                 
    self.model0=Surr(surr_path, 0)
    self.model1=Surr(surr_path, 2)
    self.model2=Surr(surr_path, 4)

    #initialize the environment
    self.reset()
    self.done=False
    self.counter = 0
    self.log_counter=0
    self.epoch_log=np.zeros((3))
    self.casename=casename
    self.log_init()

  def step(self, action):

    #---- Calculate Reward
    time_index=self.state['time']
    self.reward, self.real_action, Qs, k, qptr=self.calc_reward(action=action, time_index=time_index) 
    
    info={'theta': self.real_action, 'power': Qs, 'keff': k, 'qptr': qptr, 'time': time_index}
    self.rwd_list.append(self.reward)
    
    self.reward=np.mean(self.rwd_list) - np.std(self.rwd_list)
    #---- Update State Dictionary
    #print(self.state)
    
    self.state=self.observation_space.sample()
  
    if time_index==0 or time_index==1:
        time_index+=1   #move to the next step 
    else:
        time_index=0   #else reset back to first time step
    
    self.state['time'] = time_index        #move the time step
    
    self.log_data=np.array(list([k, qptr, self.reward]))
    
    #---- Check if the episode has ended       
    self.counter += 1
    self.log_counter += 1
    self.eps_log += self.log_data
    self.epoch_log += self.log_data
    
    if self.counter >= self.episode_length:
      self.done=True
      self.counter = 0
    
    if self.log_counter >= self.epoch_length:
      self.epoch_log=self.epoch_log/self.epoch_length
      #-- update the logger --
      with open (self.casename+'_log.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
        csvwriter.writerow(self.epoch_log)
      
      self.log_counter = 0
      self.epoch_log=np.zeros((3))
    
    return self.state, self.reward, self.done, info

  def reset(self):
    self.done=False
    self.state=self.observation_space.sample()
    self.state['time']
    self.eps_log=np.zeros((3))
    self.rwd_list=[]
    self.best_reward=-np.inf
    
    return self.state

  def render(self):
    print('Value of action: ', self.state) #print action
    print('Reward for this action: ', self.reward)

  def calc_reward(self, action, time_index):
    
    if self.action_type=='cont':
      real_action=self.descale_action(action)
    else:
      real_action=action.copy()
      
    
    #decide the time step and right NN model
    if time_index==0:
      k, Qs=self.model0.predict(action)
    elif time_index==1:
      k, Qs=self.model1.predict(action)
    elif time_index==2:
      k, Qs=self.model2.predict(action)
    else:
      raise ('The time index {} is out of bounds, it must be 0, 1, or 2'.format(time_index))
    
    #flatten data
    k=k.item()
    Qs=Qs.flatten()
    
    qptr= 6*np.max(Qs)/np.sum(Qs)
    
    f1=np.mean(np.abs(Qs-0.166667))
    f2=np.std(Qs)
    f3=np.abs(k-1.00000)
    
    #if qptr <= 1.01:
    #  rew=f1 + f2 + f3
    #else:
    #  rew=f1 + f2
    
    rew=f1 + f2 + f3
    
    rew = 1/rew   #since we are maximizing the reward

    return rew, real_action, Qs, k, qptr

  def log_init(self):

      with open (self.casename+'_log.csv', 'w') as csvfile:
          csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
          csvwriter.writerow(['keff', 'QPTR', 'Reward'])

  def descale_action(self, action):
    """
    action: np.array of the normalized action [-1, 1]

    return: real_action is the action in the original scale [0, 180]

          (b-a)(x - min)
    f(x) = --------------  + a
              max - min
    """

    action_max=180
    action_min=0
    a=-1
    b=1
    real_action=((action - a) * (action_max - action_min))/(b-a) + action_min

    return real_action

#---------------------------
#Surrogate class
#---------------------------
class Surr:
    def __init__(self, model_path, t):
        """
        This is the Neural Network surrogate model for the Serpent simulation
        
        model_path (str): is the path to the surroagte models
        t (int): 0, 2 or 4 depending on what year the surrogates are made for
        """

        self.m = load_model(Path(model_path).resolve().parent / Path("models/t%s/s"%str(t)))
        self.k_bnds = [0.88, 1.07]
        self.Q_bnds = [0.15, 0.19]

    def predict(self, X):
        """
        Input:
            X: numpy array of shape (# of input calculations, 6) which holds control drum angles
               in degrees for prediction
        Output:
            k: estimated core criticality
            Qs: numpy array of hexant powers, as fraction
        """
        shp = X.shape
        if len(shp) == 1:
            Xin = np.zeros((1, X.size))
        else:
            Xin = np.zeros_like(X)
        Xin[:, :] = X / 360
        y = self.m.predict(Xin, verbose=False)
        y[:,0] = y[:,0]*(self.k_bnds[1] - self.k_bnds[0]) + self.k_bnds[0]
        y[:,1:] = y[:,1:]*(self.Q_bnds[1] - self.Q_bnds[0]) + self.Q_bnds[0]
        k = y[:, 0]
        Qs = y[:,1:]/ y[:, 1:].sum(1).reshape(-1, 1)
        return k, Qs

if __name__ == '__main__':
    
    surr_path = '../NPIC_EMD/models/'
    
    #---------------------------
    #test the surrogate
    #---------------------------
    a = Surr(surr_path, 0)
    sample=np.random.uniform(0, 360, (1, 6))   #angles
    print(sample)
    k, Qs = a.predict(sample)
    k=k.item()
    Qs=Qs.flatten()
    print(k,Qs)