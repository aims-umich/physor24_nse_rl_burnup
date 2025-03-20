# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:12:06 2023

@author: Majdi Radaideh
"""

# --- Common packages ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def calc_cumavg(data, N):
    """
    This function returns statistics every N data points of the vector data
    
    :param data: (list) vector of data points
    :param N: (scalar) size of a subgroup
    """
    cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
    cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
    cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
    cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]

    return np.array(cum_aves), np.array(cum_std), np.array(cum_max), np.array(cum_min)

# Path to the surrogate model
path_to_ppo = './ppo_nsteps_300_best/ppo_log.csv'
path_to_a2c = './a2c_nsteps_200_best/a2c_log.csv'
eps_per_epoch = 100

# Evaluate policy training performance
ppo_data=pd.read_csv(path_to_ppo)
a2c_dada=pd.read_csv(path_to_a2c)

plot_data=pd.concat([ppo_data, a2c_dada], axis=1)
#------------------
#Progress Plot
#------------------
color_list=['b', 'g', 'r', 'b', 'g', 'r']

subplot_title=['(a) PPO $k_{eff}$ convergence', 
               '(b) PPO QPTR convergence',
               '(c) PPO Reward convergence',
               '(d) A2C $k_{eff}$ convergence',
               '(e) A2C QPTR convergence',
               '(f) A2C Reward convergence']

ny=6

xx= [(2,3,1), (2,3,2), (2,3,3), (2,3,4), (2,3,5), (2,3,6)]
plt.figure(figsize=(18, 10))
color_index=0
y_index=0
for i in range (ny): #exclude caseid from plot, which is the first column
    plt.subplot(xx[i][0], xx[i][1], xx[i][2])
    if color_index >= len(color_list):
      color_index=0
    

    ravg, rstd, rmax, rmin=calc_cumavg(list(plot_data.iloc[:,y_index]),eps_per_epoch)
    epochs=np.array(range(1,len(ravg)+1),dtype=int)

    plt.plot(epochs,ravg, '-o', c=color_list[color_index], label=r'$V_{k}$'.format(k=i))
    plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
    alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index])

    color_index+=1
    plt.plot(epochs,rmax,'s', c='k', markersize=3)
    plt.plot(epochs,rmin,'+', c='k', markersize=6)

    plt.xlabel('Epoch', fontsize=15)
    if plot_data.columns[y_index] == 'keff':
        plt.ylabel('$k_{eff}$', fontsize=15)
    else:
        plt.ylabel(plot_data.columns[y_index], fontsize=15)
    plt.title(subplot_title[i], fontsize=15)
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    y_index+=1

legend_elements = [Line2D([0], [0], color='k', marker='o', label='Mean ' + r'$\pm$ ' +r'$1\sigma$' + ' per epoch (color changes)'),
      Line2D([0], [0], linestyle='None', color='k', marker='s', label='Max per epoch'),
      Line2D([0], [0], linestyle='None', color='k', marker='+', label='Min per epoch')]
plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=15)
plt.tight_layout()
plt.savefig('rwd_plt.png', format='png', dpi=300, bbox_inches="tight")
plt.close()
#------------------