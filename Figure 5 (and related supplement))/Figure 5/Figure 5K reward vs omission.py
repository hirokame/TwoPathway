#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  26 10:46:40 2024

@author: gunahn
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from statistics import mean
import os
import glob
from datetime import datetime
from tqdm import tqdm

animal_types = ["PNOC", "NTS"]
boolean_tasks = ['Entries(even if out of task)', 'left entry', 'right entry',
       'In Turn Area']
counter_tasks = ['Left correct+rewarded','Right correct+rewarded',
                 'Left correct+omission', 'Right correct+omission',
                 'Left error', 'Right error', 'Airpuff']


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
# save today
today = datetime.today().strftime('%y%m%d')
#%%
#for event_type in tasks:
def get_reaction_of_the_neuron_after_event(path, task, enter):
    df = pd.read_csv(path)
    event_type = task
    variable_names = ['Green-L-z(Ast)', 'Green-R-z(Ast)', 'Red-L-z(DA)', 'Red-R-z(DA)']
    matter_variable = tasks + ['Initiation(start) counter']
    # change event_tpye to double
    df[event_type] = df[event_type] +1 -1

    if enter:
        event = df[df[event_type].diff() == 1]
    else:
        event = df[df[event_type].diff() == -1]
    event = event[variable_names + matter_variable]
    all_df = pd.DataFrame()
    event_unique = event.drop_duplicates(subset = 'Initiation(start) counter', keep = 'first')
    for idx, event_happens in enumerate(event_unique.index.tolist()):
        event_df = df.iloc[event_happens - 30:event_happens + 30]
        event_df = event_df[variable_names]
        event_df = event_df.reset_index(drop=True)
        all_df = pd.concat([all_df, event_df], axis=1)
    return all_df

#%%
def plot_mean_STD(df, variable_name, color, plot_name, line_style, animal_type):

    mean = df[variable_name].mean(axis=1)
    std = df[variable_name].std(axis=1)
    n = df[variable_name].shape[1]
    stderr = std / np.sqrt(n)
    confidence_interval = 1.96 * stderr
    plt.figure(figsize=(6, 6))
    plt.plot(mean, color=color, label=plot_name, linestyle=line_style, alpha=0.9)
    plt.fill_between(mean.index, mean-confidence_interval, mean+confidence_interval, alpha = 0.1, color = color)
    plt.axvline(x=30, color='black', linestyle='--', alpha=0.5, label='Event')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-1.5, 1.5)
    plt.xticks([0, 30, 60], ['-1', '0', '1'])
    plt.xlabel('Time (s)')
    plt.ylabel('z-score')
    #plt.legend()
    #plt.title(f'{animal_type}_{plot_name} Activity after {event_type}')



#%%
def plot_all(input_df):
    colors = ['g', 'g', 'r', 'r']
    plot_names = ['Neuron-DMS', 'Neuron-DLS', 'Dopamine-DMS', 'Dopamine-DLS']
    for idx, variable_name in enumerate(variable_names):
        if idx % 2 == 0:
            line_style = ':'
        else:
            line_style = '--'
        plot_mean_STD(input_df, variable_name, color = colors[idx], plot_name = plot_names[idx], line_style = line_style , animal_type= animal_types[0])
        plt.show()
#%%
def save_event_all_df(animal_type, task, today = datetime.today().strftime('%y%m%d'), enter=True):
    list_all = glob.glob(('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/{}_240116_fixed_arduino/*/*.csv').format(animal_type))
    list_all.sort()
    event_all_df = pd.DataFrame()
    for idx in range(len(list_all)):
        try:
            os.makedirs('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}'.format(today, today, animal_type))
        except FileExistsError:
            # directory already exists
            pass
        all_df = get_reaction_of_the_neuron_after_event(list_all[idx], task, enter)
        all_df = all_df.iloc[::3, :]
        all_df = all_df.reset_index(drop=True)
        event_all_df = pd.concat([event_all_df, all_df], axis=1)
    if enter:
        event_all_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}.pkl'.format(today,today, animal_type, task))
    else:
        event_all_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_exit.pkl'.format(today, today, animal_type, task))
def plot_heatmap_2s(df, variable_name, plot_name, animal_type):
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.title(f'{animal_type}_{plot_name} Activity after event')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Trial Number')
    plt.xticks([0, 10, 19], ['-1', '0', '1'])
    for_drawing = df[variable_name]
    for_drawing = for_drawing.iloc[:20]
    for_drawing = for_drawing.T
    if variable_name == 'Green-L-z(Ast)' or variable_name == 'Green-R-z(Ast)':
        plt.imshow(for_drawing, interpolation='nearest', aspect='auto', vmin=-2, vmax=2)
    else:
        plt.imshow(for_drawing, interpolation='nearest', aspect='auto', vmin=-2, vmax=2, cmap="coolwarm")

    #for_drawing = for_drawing.dropna()
    #for i in array:
    #    plt.hlines(i, 0, for_drawing.shape[1]-1, colors='r', linestyles='dashed')

    plt.axvline(10,  color='r', linestyle='dashed')

    plt.colorbar()
    plt.savefig(
        '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_heatmap_{}.pdf'.format(today, today, animal_type,
                                                                                                           plot_name), format="pdf")
    plt.show()
#%%
tasks = boolean_tasks + counter_tasks
for animal_type in tqdm(animal_types):
    for task in tqdm(tasks):
        if task in boolean_tasks:
            save_event_all_df(animal_type, task, enter= True)
            save_event_all_df(animal_type, task, enter=False)
        elif task in counter_tasks:
            save_event_all_df(animal_type, task, enter=True)
#%%
#event_all_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_{}_{}.pkl'.format(today, animal_type, task))

#%%
#load_pkl to df
#today = datetime.today().strftime('%y%m%d')
def load_pkl_to_df(animal_type, task, enter=True):
    if enter:
        event_all_df = pd.read_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}.pkl'.format(today, today, animal_type, task))
    else:
        event_all_df = pd.read_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_exit.pkl'.format(today, today, animal_type, task))
    return event_all_df
#%%
def plot_mean_STD_basic(df, variable_name, plot_name, color, line_style, legend_name):
    mean = df[variable_name].mean(axis=1)
    std = df[variable_name].std(axis=1)
    n = df[variable_name].shape[1]
    stderr = std / np.sqrt(n)
    confidence_interval = 1.96 * stderr

    plt.plot(mean, color=color, label='{}_{}'.format(plot_name, legend_name), linestyle=line_style, alpha=0.9)
    plt.fill_between(mean.index, mean-confidence_interval, mean+confidence_interval, alpha = 0.1, color = color)


#%%
variable_names = ['Green-L-z(Ast)', 'Green-R-z(Ast)', 'Red-L-z(DA)', 'Red-R-z(DA)']
from scipy.stats import ttest_ind, ttest_rel
# plot the two dataframes with t-test

def Ttest_two_df(df1, df2, variable_name):
    t_test_results = []
    # Iterate over each row index
    for i in range(len(df1)):
        # Perform t-test on the i-th row of df1 and df2

        t_statistics, p_value = ttest_ind(df1[variable_name].iloc[i, :], df2[variable_name].iloc[i, :], nan_policy='omit')
        # Append the results
        t_test_results.append([t_statistics, p_value])
    return t_test_results

#%%
plt.rcParams['font.size']= 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
# enter = True means entry event, enter = False means exit event
# boolean_event = True means boolean event, boolean_event = False means counter event, which means that reward delivery happens)
def plot_two_and_ttest(animal_type1,animal_type2, task1, task2, enter= True, boolean_event = True):
    df1 = load_pkl_to_df(animal_type1, task1, enter=enter)
    df2 = load_pkl_to_df(animal_type2, task2, enter=enter)


    colors = ['g','b', 'g','b', 'r','orange', 'r', 'orange']
    plot_names = ['Neuron-DMS', 'Neuron-DLS', 'Dopamine-DMS', 'Dopamine-DLS']
    if task1 == 'Entries(even if out of task)':
        task1 = 'initiation entry'
    if task2 == 'Entries(even if out of task)':
        task2 = 'initiation entry'
    if enter == False:
        task1 = task1.split()[0] + ' exit'
        task2 = task2.split()[0] + ' exit'
    for idx, variable_name in enumerate(variable_names):
        if idx % 2 == 0:
            line_style = ':'
        else:
            line_style = '--'
        t_test_result = Ttest_two_df(df1, df2, variable_name)
        print(t_test_result)
        significance= [x[1] < 0.0001 for x in t_test_result]

        plt.figure(figsize=(6, 6))
        x_values = [i for i, value in enumerate(significance) if value]
        y_values = [1.5 for i in x_values]
        plt.scatter(x_values, y_values, marker='$*$', color='black', label='p<0.0001')
        if animal_type1 == animal_type2:
            plot_mean_STD_basic(df1, variable_name, plot_names[idx], colors[2*idx], line_style, task1)
            plot_mean_STD_basic(df2, variable_name, plot_names[idx], colors[2*idx+1], line_style, task2)
        else:
            plot_mean_STD_basic(df1, variable_name, plot_names[idx], colors[2*idx], line_style, animal_type1)
            plot_mean_STD_basic(df2, variable_name, plot_names[idx], colors[2*idx+1], line_style, animal_type2)
        if boolean_event:
            plt.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='Event')
        else:
            plt.axvline(x=10, color='black', linestyle='--', alpha=0.5, label='Choice port enter')
            plt.axvline(x=12, color='black', linestyle=':', alpha=0.5, label='Delivery')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim(-1.5, 1.6)
        plt.xticks([0, 10, 20], ['-1', '0', '1'])
        plt.xlabel('Time (s)')
        plt.ylabel('z-score')
        # make legend transparent
        plt.legend(framealpha=0.5)
        if animal_type1 == animal_type2:
            if animal_type1 =="dopamine":
                plt.title('{} activity difference between \n {} and {}'.format(plot_names[idx], task1,task2))
                plt.savefig(
                    '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_{}.pdf'.format(today, today, task1, task2, plot_names[idx]))
            else:
                plt.title('{} {} activity difference between \n {} and {}'.format(animal_type1, plot_names[idx], task1, task2))
                plt.savefig('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_{}_{}.pdf'.format(today, today, animal_type1, task1, task2, plot_names[idx]))
        else:
            plt.title('{} activity difference between \n {} and {} after {}'.format( plot_names[idx], animal_type1, animal_type2, task1,))
            plt.savefig('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_{}_{}_vs_{}.pdf'.format(today, today, animal_type1, task1, task2, plot_names[idx], animal_type2))
        plt.show()

#%%

#change the name of pkl files in /Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event
# eample /Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event/240225_PNOC_error.pkl to '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event/{}_PNOC_error.pkl'.format(today)
all_pkls = glob.glob('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event/*.pkl')
for pkl in all_pkls:
    df = pd.read_pickle(pkl)
    df.to_pickle(pkl.replace('240225', datetime.today().strftime('%y%m%d')))

#%%
# make the dopamine activity pkl files different
PNOC_pkls = glob.glob('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event/*PNOC*.pkl')
#alphabetical order sorting
PNOC_pkls.sort()

NTS_pkls = glob.glob('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event/*NTS*.pkl')
NTS_pkls.sort()

#%%

#%%
#for dopamine release, PNOC and NTS animals has same signal.
def merge_dopamine(PNOC_pkl, NTS_pkl):
    df_PNOC = pd.read_pickle(PNOC_pkl)
    df_NTS = pd.read_pickle(NTS_pkl)
    task_PNOC = PNOC_pkl.split('_')[8].split('.')[0]
    task_NTS = NTS_pkl.split('_')[8].split('.')[0]

    if task_PNOC != task_NTS:
        print('The tasks are different')
        raise ValueError
    if "exit" in PNOC_pkl:
        task_PNOC = task_PNOC + '_exit'
    print(task_PNOC)
    dopamine_merge= pd.concat([df_PNOC, df_NTS],axis =1)
    #save the dopamine_merge
    dopamine_merge.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/240226_event_by_event/{}_dopamine_{}.pkl'.format(today, task_PNOC))
    return dopamine_merge

for PNOC_pkl, NTS_pkl in zip(PNOC_pkls, NTS_pkls):

    merge_dopamine(PNOC_pkl, NTS_pkl)

#%%
# comparing the PNOC and NTS in the choice port enter and reward delivery
today = datetime.today().strftime('%y%m%d')
plot_two_and_ttest('PNOC', 'NTS','correct+rewarded', 'correct+rewarded' , boolean_event=False)
plot_two_and_ttest('PNOC', 'NTS','correct+omission', 'correct+omission' , boolean_event=False)
plot_two_and_ttest('PNOC', 'NTS','Airpuff', 'Airpuff' , boolean_event=False)
#%%
plot_two_and_ttest('dopamine', 'dopamine','Entries(even if out of task)', 'correct+rewarded' , boolean_event=False)
plot_two_and_ttest('dopamine', 'dopamine','correct+rewarded', "Airpuff" , boolean_event=False)
plot_two_and_ttest('dopamine', 'dopamine','correct+rewarded', "correct+omission" , boolean_event=False)

plot_two_and_ttest('dopamine', 'dopamine','Entries(even if out of task)', "left entry" ,  enter=False, boolean_event=True)
plot_two_and_ttest('dopamine', 'dopamine','Entries(even if out of task)', "correct+rewarded" ,  enter=True, boolean_event=False)
plot_two_and_ttest('dopamine', 'dopamine','left entry', "right entry" ,  enter=False, boolean_event=True)

#%%
plot_two_and_ttest('PNOC', 'PNOC','Airpuff', 'error' , boolean_event=False)
plot_two_and_ttest('NTS', 'NTS','Airpuff', 'error' , boolean_event=False)
plot_two_and_ttest('dopamine', 'dopamine','left correct+omission', 'left error' , boolean_event=True)
plot_two_and_ttest('dopamine', 'dopamine','right correct+omission', 'right error' , boolean_event=True)
plot_two_and_ttest('dopamine', 'dopamine','correct+omission', 'error' , boolean_event=True)

plot_two_and_ttest('PNOC', 'PNOC','correct+omission', 'error' , boolean_event=True)
plot_two_and_ttest('NTS', 'NTS','correct+omission', 'error' , boolean_event=True)
#%%
plot_two_and_ttest('PNOC', 'PNOC','left correct+omission', 'left error' , boolean_event=True)
plot_two_and_ttest('NTS', 'NTS','left correct+omission', 'left error' , boolean_event=True)
plot_two_and_ttest('PNOC', 'PNOC','right correct+omission', 'right error' , boolean_event=True)
plot_two_and_ttest('NTS', 'NTS','right correct+omission', 'right error' , boolean_event=True)
#%%
plot_two_and_ttest('PNOC', 'NTS','In Turn Area', 'In Turn Area' , enter=False,boolean_event=True)
plot_two_and_ttest('PNOC', 'NTS','Entries(even if out of task)', 'Entries(even if out of task)' , enter=False,boolean_event=True)
plot_two_and_ttest('PNOC', 'NTS','left entry', 'left entry' , enter=False,boolean_event=True)
plot_two_and_ttest('PNOC', 'NTS','right entry', 'right entry' , enter=False,boolean_event=True)


#%%

plot_two_and_ttest('PNOC', 'NTS','correct+rewarded', 'correct+rewarded' , boolean_event=False)
plot_two_and_ttest('PNOC', 'NTS','correct+omission', 'correct+omission' , boolean_event=False)
plot_two_and_ttest('PNOC', 'NTS','Airpuff', 'Airpuff' , boolean_event=False)
#%%
plot_two_and_ttest('PNOC', 'NTS','Airpuff', 'Airpuff' , boolean_event=False)
plot_two_and_ttest('PNOC', 'NTS','Airpuff', 'Airpuff', boolean_event=False)
#%%
plot_two_and_ttest('PNOC', 'PNOC','correct+rewarded', 'Airpuff' , boolean_event=False)
plot_two_and_ttest('NTS', 'NTS','correct+rewarded', 'Airpuff', boolean_event=False)
plot_two_and_ttest('PNOC', 'PNOC','correct+omission', 'Airpuff' , boolean_event=False)
plot_two_and_ttest('NTS', 'NTS','correct+omission', 'Airpuff', boolean_event=False)
plot_two_and_ttest('PNOC', 'PNOC','error', 'Airpuff' , boolean_event=False)
plot_two_and_ttest('NTS', 'NTS','error', 'Airpuff', boolean_event=False)
#%%

plot_two_and_ttest('PNOC', 'PNOC','correct+rewarded', 'correct+omission' , boolean_event=False)
plot_two_and_ttest('NTS', 'NTS','correct+rewarded', 'correct+omission', boolean_event=False)
plot_two_and_ttest('PNOC', 'PNOC','correct+omission', 'error' , boolean_event=False)
plot_two_and_ttest('NTS', 'NTS','correct+omission', 'error', boolean_event=False)


#%%
plot_two_and_ttest('PNOC', 'Entries(even if out of task)', 'left entry' )
plot_two_and_ttest('PNOC', 'Entries(even if out of task)', 'left entry', enter=False)
plot_two_and_ttest('PNOC', 'Entries(even if out of task)', 'right entry' )
plot_two_and_ttest('PNOC', 'Entries(even if out of task)', 'right entry', enter=False )
#%%
plot_two_and_ttest('NTS', 'Entries(even if out of task)', 'left entry' )
plot_two_and_ttest('NTS', 'Entries(even if out of task)', 'left entry', enter=False)
plot_two_and_ttest('NTS', 'Entries(even if out of task)', 'right entry' )
plot_two_and_ttest('NTS', 'Entries(even if out of task)', 'right entry', enter=False )

#%%
#compare reward vs omission
plot_two_and_ttest('PNOC', 'Left correct+rewarded', 'Left correct+omission',  boolean_event = False )
plot_two_and_ttest('PNOC', 'Right correct+rewarded',  'Right correct+omission', enter = True, boolean_event = False)
plot_two_and_ttest('NTS', 'Left correct+rewarded', 'Left correct+omission' , boolean_event = False)
plot_two_and_ttest('NTS', 'Right correct+rewarded',  'Right correct+omission', boolean_event = False)
#%%


#merge df1 and df2 to compare

new_df =pd.concat([df1, df2],axis =1)
#%%
today =240226

def save_merge_left_right(animal_type, task1, task2):
    if task1.split()[-1] != task2.split()[-1]:
        print('task name should be same')
        raise ValueError
    else:
        df1 = load_pkl_to_df(animal_type, task1)
        df2 = load_pkl_to_df(animal_type, task2)
        new_df =pd.concat([df1, df2],axis =1)
        new_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}.pkl'.format(today, today, animal_type, task1.split()[-1]))
Be_merged_list_left = ['left entry','Left correct+rewarded', 'Left correct+omission', 'Left error']
Be_merged_list_right = ['right entry','Right correct+rewarded', 'Right correct+omission', 'Right error']

def save_merge_left_right_exit(animal_type, task1, task2):
    if task1.split()[-1] != task2.split()[-1]:
        print('task name should be same')
        raise ValueError
    else:
        df1 = load_pkl_to_df(animal_type, task1, enter=False)
        df2 = load_pkl_to_df(animal_type, task2, enter=False)
        new_df =pd.concat([df1, df2],axis =1)
        new_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_exit.pkl'.format(today, today, animal_type, task1.split()[-1]))


for animal_type in animal_types:
    for task_left, task_right in zip(Be_merged_list_left, Be_merged_list_right):
        save_merge_left_right(animal_type, task_left, task_right)
        if task_left == 'left entry':
            save_merge_left_right_exit(animal_type, task_left, task_right)
        else:
            print("good")

#%%
for animal_type in ['NTS', 'GFAP', 'PNOC']:
    list_all = glob.glob(('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/{}_240116_fixed_arduino/*/*.csv').format(animal_type))
    list_all.sort()
    for file in tqdm(list_all):
        df= pd.read_csv(file)
        #if df[variable_names] contatins nan, then print the file
        if df[variable_names].isna().any().any():
            print(file)
#%%
significance = [x<0.05 for x in p_values]
x_values = [i for i, value in enumerate(significance) if value]
y_values = [1 for i in x_values]
plt.scatter(x_values,y_values, marker='x')
plt.show()
