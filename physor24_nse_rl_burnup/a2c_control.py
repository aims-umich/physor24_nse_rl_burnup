# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:12:06 2023

@author: Majdi Radaideh
"""

# --- Common packages ---
import os, sys
import time as tme
import numpy as np
import pandas as pd
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import product
from typing import Callable
import torch

#--- Filter warnings ---
import warnings
warnings.filterwarnings("ignore")
os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#os.environ['CUDA_AVAILABLE_DEVICES']= ""
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#--- Stable Baselines ---
from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import MlpPolicy, CnnPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import get_device
import stable_baselines3
stable_baselines3.common.utils.get_device(device='cpu')    #if GPU is full uncomment this line to use cpu


#our source functions
from source.env import MicroReactorEnv, Surr
from source.tools import calc_cumavg

# Path to the surrogate model
surr_path = './NPIC_EMD/models/'
run_mode= 'test'                 # train or test
model_path= './a2c_nsteps_200/a2c_best_model.pkl'             #model path for the test mode

#-------------------------------------------------------------
# Wrapper function to construct a vectorized environment
#--------------------------------------------------------------

def make_env(env_id: str, surr_path: str, action_type: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = MicroReactorEnv(casename=env_id, surr_path=surr_path, action_type=action_type)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

#-------------------------------------------------------------
# Callback function for RL logging
#--------------------------------------------------------------
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN and others)
    This callback is used to access the algorathim on the fly to plot statistics and save models
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    
    true_n_steps=(n_steps+1)*num_cpu
    # Print stats every check_freq calls
    if (true_n_steps) % (check_freq) == 0:
      # Evaluate policy training performance
      data=pd.read_csv(os.path.join(dir_name, casename + '_log.csv'))
      mean_reward = np.mean(data["Reward"].iloc[-eps_per_epoch:])
      max_reward = np.max(data["Reward"].iloc[-eps_per_epoch:])

      #--------------
      #model saving
      #-------------
      _locals['self'].save(os.path.join(dir_name, casename +'_last_model.pkl')) 
      if mean_reward > best_mean_reward:
          best_mean_reward = mean_reward
          _locals['self'].save(os.path.join(dir_name, casename +'_best_model.pkl'))
          print(f"Reward at Step {true_n_steps}/{total_timesteps}: Mean Reward = {mean_reward:.2f}, Max Reward = {max_reward:.2f}, a new best model is saved")
      else:
        print(f"Reward at Step {true_n_steps}/{total_timesteps}: Mean Reward = {mean_reward:.2f}, Max Reward = {max_reward:.2f}")

      #------------------
      #Progress Plot
      #------------------
      color_list=['b', 'g', 'r', 'darkorange', 'm', 'c']
      plot_data=data.copy()
      #plot_data=plot_data.iloc[:,[0,2,4,6,8,10,12,13,15,14]]
      #ny=6
      plot_data=plot_data.iloc[:,[0,1,2]]
      ny=3

      xx= [(1,3,1), (1,3,2), (1,3,3)]
      plt.figure(figsize=(18, 7))
      color_index=0
      y_index=0
      for i in range (ny): #exclude caseid from plot, which is the first column
          plt.subplot(xx[i][0], xx[i][1], xx[i][2])
          if color_index >= len(color_list):
            color_index=0
          
          if 0:
          #if i==0 or i==1:
            if i==0:
              y_names='Drum Angle'
              y_labels=[r'$\theta_1$', r'$\theta_2$', r'$\theta_3$', r'$\theta_4$', r'$\theta_5$', r'$\theta_6$']
            elif i==1:
              y_names='Hexant Power Fraction'
              y_labels=['$Q_1$', '$Q_2$', '$Q_3$', '$Q_4$', '$Q_5$', '$Q_6$']

            for j in range(3):
              if color_index >= len(color_list):
                color_index=0
              ravg, rstd, rmax, rmin=calc_cumavg(list(plot_data.iloc[:,y_index]),eps_per_epoch)
              epochs=np.array(range(1,len(ravg)+1),dtype=int)

              plt.plot(epochs,ravg, '-o', c=color_list[color_index], label=y_labels[j])
              plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
              alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index])

              color_index+=1
              #plt.plot(epochs,rmax,'s', c='k', markersize=3)
              #plt.plot(epochs,rmin,'+', c='k', markersize=6)

              plt.xlabel('Epoch', fontsize=14)
              plt.ylabel(y_names, fontsize=14)
              plt.legend()
              y_index+=1
          else:

            ravg, rstd, rmax, rmin=calc_cumavg(list(plot_data.iloc[:,y_index]),eps_per_epoch)
            epochs=np.array(range(1,len(ravg)+1),dtype=int)

            plt.plot(epochs,ravg, '-o', c=color_list[color_index], label=r'$V_{k}$'.format(k=i))
            plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
            alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index])

            color_index+=1
            plt.plot(epochs,rmax,'s', c='k', markersize=3)
            plt.plot(epochs,rmin,'+', c='k', markersize=6)

            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel(plot_data.columns[y_index], fontsize=14)
            y_index+=1

      legend_elements = [Line2D([0], [0], color='k', marker='o', label='Mean ' + r'$\pm$ ' +r'$1\sigma$' + ' per epoch (color changes)'),
            Line2D([0], [0], color='k', marker='s', label='Max per epoch'),
            Line2D([0], [0], linestyle='-.', color='k', marker='+', label='Min per epoch')]
      plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4)
      plt.tight_layout()
      plt.savefig(os.path.join(dir_name, casename + '_plt.png'), format='png', dpi=200, bbox_inches="tight")
      plt.close()
    #------------------

    n_steps += 1
    return True

if __name__ == '__main__':

  surr_path = './NPIC_EMD/models/'             #path to the surrogate model    
  
  #User inputs
  total_timesteps = 1200000                    #total number of time steps
  check_freq = 20000                           #frequency of data logging in time steps
  eps_per_epoch = 100                          #number of episodes per epoch (for grouping and statistics plotting)
  num_cpu = 20                                 # Number of parallel processes (agents) to use
  action_type='disc'                           #don't change, only discrete is tested
  n_epochs= int(total_timesteps/check_freq)    # total number of epochs to run
  casename='a2c'                               #prefix for naming convention
  
                
  
  print('***************User Parameters**********************')
  print('n_epochs = ', n_epochs)
  print('num_cpu = ', num_cpu)
  print('eps_per_epoch = ', eps_per_epoch)
  print('check_freq = ', check_freq)
  print('total_timesteps = ', total_timesteps)
  print('****************************************************')

  if 1:
    #-----------------
    #A2C block
    #-----------------
    
    # Uncomment for grid search
    #n_a2c_steps=[200, 300, 400]           # the larger the value, the longer the time the policy gets updated. 
    #ent_coef=[0.001, 0.005, 0.01, 0.02]   #0.01 some exploration, 0.1 exploration term is high, 0.5 very high exploration 
    #vf_coef= [0.5,0.75,1]                #0.75 means VF optimization is more important than the policy. 0.5 is equal weights. 
    #clip_range= [0.1, 0.2, 0.3, 0.4]     #0.3 more exploration and less conservative policy update. 0.5 is extreme. 
    
    #use lines below if you decided the best values to run a single case
    n_a2c_steps = [200]
    ent_coef=[0.02]
    vf_coef= [0.5] 
    maxgradnorm= [5]
    
    grid=list(product(n_a2c_steps, ent_coef, vf_coef, maxgradnorm))   #this forms a multidimensional grid for search
    
    for k in grid: 
      
      #create a directory for the results
      dir_name=casename + '_nsteps_{}'.format(k[0])
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)
      
      if run_mode == 'train':
          #******************************
          # Policy train mode
          #******************************
          # Create a vectorized environment
          a2c_env = SubprocVecEnv([make_env(os.path.join(dir_name, casename), surr_path, action_type, i) for i in range(num_cpu)])
          best_mean_reward, n_steps = -np.inf, 0  #init the callback variables
          print('-----------------------------------')
          print('Running Case {}'.format(k))
          print('-----------------------------------')
          
          t0=tme.time()
          #create a a2c object
          a2c = A2C(MultiInputPolicy, a2c_env, n_steps=k[0], ent_coef=k[1], vf_coef=k[2], max_grad_norm=k[3], seed=42, verbose=0, device='cpu')
          #train a2c, this line takes time
          a2c.learn(total_timesteps=total_timesteps, log_interval=10, callback=callback)
          #training time
          tend=tme.time()
          print('Train Time = ', tend-t0)
      else:
          #******************************
          # Policy test mode
          #****************************** 
          # Create a vectorized environment
          a2c_env = SubprocVecEnv([make_env(casename, surr_path, action_type, i) for i in range(2)])
          # Load the pretrained policy
          model = A2C.load(model_path, device='cpu')
          
          
          # Enjoy trained agent
          n_samples=3
          test_data=np.zeros((n_samples,10))
          obs = a2c_env.reset()
          t0=tme.time()
          for s in range(n_samples):
              t0=tme.time()
              action, _states = model.predict(obs)
              obs, rewards, dones, info = a2c_env.step(action)
              tend=tme.time()
              data_vec=list([info[0]['time']]) + list(action[0]) + [info[0]['keff'], info[0]['qptr'], rewards[0]]
              test_data[s,:] = data_vec
          
          test_data=pd.DataFrame(test_data, columns=['Time (yr)', '$\\theta_1$', '$\\theta_2$', '$\\theta_3$', '$\\theta_4$', '$\\theta_5$', '$\\theta_6$', '$k_{eff}$', 'QPTR', 'Reward'])
          test_data['Time (yr)']=test_data['Time (yr)'].replace([1,2], [2,4])     #replace integer index with the real time step value in years
          test_data.index+=1
          test_data.to_csv(casename + '_test.csv', index=True, index_label=['Sample'])
          
          os.remove('a2c_log.csv')
          print(test_data)
          print('Test Time = ', tend-t0) #test time