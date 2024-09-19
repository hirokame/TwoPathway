#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:46:40 2023

@author: gunahn
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import subprocess
import scienceplots
import glob
import os
from matplotlib import rcParams
from tqdm import tqdm
from datetime import datetime
today = datetime.today().strftime("%y%m%d")
task = 'PNOC'
#task = "NTS"

# ignore the warning
import warnings
warnings.filterwarnings("ignore")

today = today + task + 'Value_visualization'
#%%
# create the directory  '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}/'.format(today)
try:
    os.makedirs('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}/'.format(today))
except FileExistsError:
    # directory already exists
    pass
#%%

def visualize_Action_value(path, today, animal_number, day_number):

    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0'])

    # plot the action value and mouse behavior
    plt.figure()
    plt.rcParams['pdf.use14corefonts'] = True

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #hist, bin_edges = np.histogram(df['Action_value'], bins=20)

    #plt.plot(df['Action_value'], , label = 'Action value')

    change = df[df.Correct_Port_Right.diff()!=0]

    # plot the action value

    plt.plot(df['Q Action Values'], label = 'Action value', color= 'green')
    for i in change.index[:-1]:
        plt.axvline(x=i, color='r', linestyle='--')
    plt.axvline(x=change.index[-1], color='r', linestyle='--', label = 'Change of port')
    plt.scatter(range(len(df)), df['Mouse_Right']*7-3.5, label = 'Mouse_Behavior', marker='x', s=5)

    df['reward_direction'] = df['Reward'] * (df['Mouse_Right']-0.5)
    #plot the reward direction
    plt.scatter(range(len(df)), df['reward_direction']*6, label = 'Reward_Direction', marker='d', s=5)

    #plot the airpuff
    df['airpuff_direction'] = df['Airpuff'] * (df['Mouse_Right']-0.5)
    air_puffed = df[df.airpuff_direction!=0]
    plt.scatter(air_puffed.Trial, air_puffed['airpuff_direction']*2, label = 'Airpuff', marker='o', s=5)

    plt.xlabel('Trials')
    plt.ylabel('Action value of turning right')
    plt.title('Action value of mouse {} on day {}'.format(animal_number, day_number))

    plt.legend(loc='upper right', borderaxespad=2.1, bbox_to_anchor=(1.2, 1))

    # save in svg format
    plt.savefig('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}/Action_value_animal_{}_day{}.pdf'.format(today, animal_number, day_number))

    plt.show()
    return df

#%%
all_mice = glob.glob('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/SessionInfo{}_Beh_Sig_new/*'.format(task))

for mouse_num, mouse_file in tqdm(enumerate(all_mice)):
    mouse_files = glob.glob(all_mice[mouse_num] + '/*')
    mouse_files.sort()

    for idx, path in enumerate(mouse_files):
        animal_number = path.split('/')[-2]
        day_number = path.split('/')[-1].split('_')[1]
        print(animal_number, day_number)


        df = visualize_Action_value(path, today, animal_number, day_number)


