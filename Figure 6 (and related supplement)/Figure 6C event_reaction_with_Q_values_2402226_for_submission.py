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

plt.rcParams['font.size']= 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'

#%%
#for event_type in tasks:
# 60 means 2s
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
        event_df = df.iloc[event_happens - 60:event_happens + 60]
        Q_action_value_per_trial = event_df['Q Action Values'].values[60]

        event_df = event_df[variable_names]

        Q_action_value_per_trial_df = pd.DataFrame(Q_action_value_per_trial, index=['Q_action_value'],
                                                   columns=variable_names)

        event_df = pd.concat([event_df, Q_action_value_per_trial_df], axis=0)
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
    list_all = glob.glob(('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/{}_30hz_updated_final/*/*.csv').format(animal_type))
    list_all.sort()
    print(list_all)
    event_all_df = pd.DataFrame()
    try:
        os.makedirs(
            '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event'.format(today))
    except FileExistsError:
        # directory already exists
        pass
    for idx in range(len(list_all)):
        try:
            os.makedirs('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}'.format(today, today, animal_type))
        except FileExistsError:
            pass
        animal_number = list_all[idx].split('/')[-2]
        all_df = get_reaction_of_the_neuron_after_event(list_all[idx], task, enter)
        all_df_10hz = all_df.iloc[::3, :]
        all_df_10hz.iloc[-1] = all_df.iloc[-1]
        all_df_10hz = all_df_10hz.reset_index(drop=True)
        event_all_df = pd.concat([event_all_df, all_df_10hz], axis=1)

    event_all_df = event_all_df.rename(index={all_df_10hz.index[-1]: 'Q_action_value'})
    #fill na with 0
    event_all_df = event_all_df.fillna(0)
    # divide the event_all_df into 3 groups using the Q_action_value
    df_transposed = event_all_df.transpose()
    Q_values_abs = np.abs(df_transposed['Q_action_value'])
    # Divide the DataFrame based on Q_action_value
    Q1 = np.percentile(Q_values_abs, 25)
    Q3 = np.percentile(Q_values_abs, 75)
    high_group = df_transposed[Q_values_abs>Q3].transpose()
    low_group = df_transposed[Q_values_abs<Q1].transpose()
    middle_group = df_transposed[(Q_values_abs>Q1) & (Q_values_abs<Q3)].transpose()
    if enter:
        event_all_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}.pkl'.format(today,today, animal_type, task))
        high_group.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_high.pkl'.format(today, today, animal_type, task))
        low_group.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_low.pkl'.format(today, today, animal_type, task))
        middle_group.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_middle.pkl'.format(today, today, animal_type, task))
    else:
        event_all_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_exit.pkl'.format(today, today, animal_type, task))
        high_group.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_high_exit.pkl'.format(today, today, animal_type, task))
        low_group.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_low_exit.pkl'.format(today, today, animal_type, task))
        middle_group.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_middle_exit.pkl'.format(today, today, animal_type, task))
#%%


tasks = boolean_tasks + counter_tasks


def load_pkl_to_df_org(animal_type, animal_number, task, enter):
    animal_type = animal_type + str(animal_number)
    if enter:
        event_all_df = pd.read_pickle(
            '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}.pkl'.format(today, today,
                                                                                                           animal_type,
                                                                                                           task))
    else:
        event_all_df = pd.read_pickle(
            '/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_exit.pkl'.format(today,
                                                                                                                today,
                                                                                                                animal_type,
                                                                                                                task))
    return event_all_df

variable_names = ['Green-L-z(Ast)', 'Green-R-z(Ast)', 'Red-L-z(DA)', 'Red-R-z(DA)']
plot_names = ['Neuron-DMS', 'Neuron-DLS', 'Dopamine-DMS', 'Dopamine-DLS']
task = 'In Turn Area'
for animal_type in animal_types:
    if animal_type == "PNOC":
        animal_numbers = [1,2,3,4,5]
    elif animal_type == "NTS":
        animal_numbers = [2,3,5,6]
    for animal_number in animal_numbers:
        mean_values = pd.DataFrame()
        df_per_animal = load_pkl_to_df_org(animal_type,animal_number,task, enter=False )
        df_per_animal_without_Q = df_per_animal.iloc[:-1]
        for variable_name in variable_names:
            current_mean = df_per_animal_without_Q[variable_name].mean(axis=1)
            mean_values = pd.concat([mean_values, current_mean], axis=1)
        mean_values.columns = plot_names
        mean_values.to_csv('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}{}_{}_exit.csv'.format(today, today, animal_type, animal_number, task))

#%%
tasks = boolean_tasks + counter_tasks
animal_types =["PNOC","NTS"]
for animal_type in animal_types:
    for task in tqdm(tasks[2:]):
        if task in boolean_tasks:
            save_event_all_df(animal_type, task, enter= True)
            save_event_all_df(animal_type, task, enter=False)
        elif task in counter_tasks:
            save_event_all_df(animal_type, task, enter=True)
#%%
#event_all_df.to_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_{}_{}.pkl'.format(today, animal_type, task))

#%%
#load_pkl to df
today = '240227'


def load_pkl_to_df(animal_type, task, value, enter):
    if enter:
        event_all_df = pd.read_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_{}.pkl'.format(today, today, animal_type, task, value))
    else:
        event_all_df = pd.read_pickle('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_event_by_event/{}_{}_{}_{}_exit.pkl'.format(today, today, animal_type, task, value))
    return event_all_df
#%%
def plot_mean_STD_basic(df, variable_name, plot_name, color, line_style, legend_name):
    mean = df[variable_name].mean(axis=1)
    std = df[variable_name].std(axis=1)
    n = df[variable_name].shape[1]
    #print(mean)
    #print(df[variable_name])
    stderr = std / np.sqrt(n)
    #confidence_interval = 1.96 * stderr

    # Convert index to a list if it's not already numeric or a simple range
    plt.plot(mean, color=color, label='{}_{}'.format(plot_name, legend_name), linestyle=line_style, alpha=0.9)
    plt.fill_between(np.arange(len(mean), mean-1.96 * stderr, mean+1.96 * stderr, alpha = 0.1, color = color))
#%%
def plot_mean_STD_basic_fixed(df, variable_name, plot_name, color, line_style, legend_name):
    mean = df[variable_name].mean(axis=1)
    std = df[variable_name].std(axis=1)
    n = df[variable_name].shape[1]

    plt.plot(mean, color=color, label='{}_{}'.format(plot_name, legend_name), linestyle=line_style, alpha=0.9)
    plt.fill_between(np.arange(len(mean)), mean - 1.96 * std / np.sqrt(n), mean + 1.96 * std / np.sqrt(n), alpha=0.1,
                     color=color)


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
#plot_two_and_ttest('PNOC', 'PNOC','In Turn Area', 'In Turn Area',"high","low", boolean_event=False)

variable_names = ['Green-L-z(Ast)', 'Green-R-z(Ast)', 'Red-L-z(DA)', 'Red-R-z(DA)']

#%%
plt.rcParams['font.size']= 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
# enter = True means entry event, enter = False means exit event
# boolean_event = True means boolean event, boolean_event = False means counter event, which means that reward delivery happens)
def plot_two_and_ttest(animal_type1,animal_type2, task1, task2,value1,value2, enter= True, boolean_event = True):
    df1 = load_pkl_to_df(animal_type1, task1, value1, enter=enter)
    df2 = load_pkl_to_df(animal_type2, task2, value2, enter=enter)
    df1 =df1.iloc[:-1, :]
    df2 =df2.iloc[:-1, :]
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
        plt.scatter(x_values, y_values, marker='$*$', color='black', label='p< 0.0001')
        if animal_type1 == animal_type2:
            plot_mean_STD_basic_fixed(df1, variable_name, plot_names[idx], colors[2*idx], line_style, value1)
            plot_mean_STD_basic_fixed(df2, variable_name, plot_names[idx], colors[2*idx+1], line_style, value2)
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
plot_two_and_ttest('NTS', 'NTS','Entries(even if out of task)', 'Entries(even if out of task)',"high","low", enter=True, boolean_event=True)

plot_two_and_ttest('NTS', 'NTS','Entries(even if out of task)', 'Entries(even if out of task)',"high","low", enter=False, boolean_event=True)

#%%
plot_two_and_ttest('PNOC', 'PNOC','Entries(even if out of task)', 'Entries(even if out of task)',"high","low", enter=True, boolean_event=True)
#%%
plot_two_and_ttest('PNOC', 'PNOC','Entries(even if out of task)', 'Entries(even if out of task)',"high","low", boolean_event=True, enter=False)
#%%
plot_two_and_ttest('PNOC', 'PNOC','Left correct+rewarded', 'Left correct+rewarded',"high","low", boolean_event=False, enter=True)
#%%
plot_two_and_ttest('PNOC', 'PNOC','Right correct+rewarded', 'Right correct+rewarded',"high","low", boolean_event=False, enter=True)

#%%
plot_two_and_ttest('PNOC', 'PNOC','In Turn Area', 'In Turn Area',"high","low", boolean_event=True, enter=True)
plot_two_and_ttest('PNOC', 'PNOC','In Turn Area', 'In Turn Area',"high","low", boolean_event=True, enter=False)
#%%
plot_two_and_ttest('PNOC', 'PNOC','In Turn Area', 'In Turn Area',"high","low", boolean_event=True, enter=True)
plot_two_and_ttest('PNOC', 'PNOC','In Turn Area', 'In Turn Area',"high","low", boolean_event=True, enter=False)
#%%
plot_two_and_ttest('NTS', 'NTS','In Turn Area', 'In Turn Area',"high","low", boolean_event=True, enter=True)
plot_two_and_ttest('NTS', 'NTS','In Turn Area', 'In Turn Area',"high","low", boolean_event=True, enter=False)
#%%
plot_two_and_ttest('NTS', 'NTS','left entry', 'left entry',"high","low", boolean_event=True, enter=True)
plot_two_and_ttest('NTS', 'NTS','left entry', 'left entry',"high","low", boolean_event=True, enter=False)
#%%
plot_two_and_ttest('PNOC', 'PNOC','right entry', 'right entry',"high","low", boolean_event=True, enter=True)
plot_two_and_ttest('PNOC', 'PNOC','right entry', 'right entry',"high","low", boolean_event=True, enter=False)
#plot_two_and_ttest('PNOC', 'PNOC','left entry', 'left entry',"high","low", boolean_event=True, enter=True)
#plot_two_and_ttest('PNOC', 'PNOC','left entry', 'left entry',"high","low", boolean_event=True, enter=False)


#plot_two_and_ttest('PNOC', 'PNOC','In Turn Area', 'In Turn Area',"middle","low", boolean_event=True, enter=False)

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
if "exit" in PNOC_pkls[-2]:
    print('The last file is exit')
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



