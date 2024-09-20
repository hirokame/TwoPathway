#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:46:40 2023

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

animal_types = ["PNOC", "NTS", "GFAP"]
tasks = ["Left correct+rewarded", "Right correct+rewarded", "Left correct+omission", "Right correct+omission", "Airpuff", "Left error", "Right error" ]

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
#%%

task_animal_type = []
df_sorted_list = []

for task in tasks:
    for animal_type in animal_types:
        task_animal_type = task + '_' + animal_type
        df_sorted_per_each = pd.read_csv('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/data/{}_{}_data.csv'.format("2023-10-04", task_animal_type), index_col=0)
        df_sorted_list.append(df_sorted_per_each)
#%%

def calculate_task_bracketing_index(number, target,ratio):
    df_one = df_sorted_list[number].T

    data_of_interest = df_one[target]

    data_of_interest = data_of_interest.T

    task_bracketing_index_per_session=[]
    for index, row in data_of_interest.iterrows():
        init_idx = row['init_list_all'].astype(int)
        exit_idx = row['side_port_exit_list_all'].astype(int)
        trial = row[init_idx+210: exit_idx+210]
        x= trial.values
        task_bracketing_index = x[:int(len(x)*ratio)].mean() + x[-int(len(x)*ratio):].mean() - x[int(len(x)*ratio):-int(len(x)*ratio)].mean()

        task_bracketing_index_per_session.append(task_bracketing_index)

    return task_bracketing_index_per_session
#%%
#print(task_animal_type)
task_bracketing_index_per_session_PNOC_DMS = []
task_bracketing_index_per_session_NTS_DMS = []
task_bracketing_index_per_session_DA_DMS = []

task_bracketing_index_per_session_PNOC_DLS = []
task_bracketing_index_per_session_NTS_DLS = []
task_bracketing_index_per_session_DA_DLS = []

for i in range(len(task_animal_types)):
    if i%3 == 0:
        #print(task_animal_types[i])
        #print(mean([x for x in calculate_task_bracketing_index(i, "Green-L-z(Ast)", 0.25) if str(x) != 'nan']))
        task_bracketing_index_per_session_PNOC_DMS += calculate_task_bracketing_index(i, "Green-L-z(Ast)", 0.25)
        task_bracketing_index_per_session_PNOC_DLS += calculate_task_bracketing_index(i, "Green-R-z(Ast)", 0.25)
        task_bracketing_index_per_session_DA_DMS += calculate_task_bracketing_index(i, "Red-L-z(DA)", 0.25)
        task_bracketing_index_per_session_DA_DLS += calculate_task_bracketing_index(i, "Red-R-z(DA)", 0.25)

    elif i%3 == 1:
        #print(task_animal_types[i])
        #print(mean([x for x in calculate_task_bracketing_index(i, "Green-L-z(Ast)", 0.25) if str(x) != 'nan']))
        task_bracketing_index_per_session_NTS_DMS += calculate_task_bracketing_index(i, "Green-L-z(Ast)", 0.25)
        task_bracketing_index_per_session_NTS_DLS += calculate_task_bracketing_index(i, "Green-R-z(Ast)", 0.25)

#%%
#plot bocc plot for task bracketing index
plt.clf()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
plt.rcParams['font.size']= 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
# Example lists
list1 = task_bracketing_index_per_session_PNOC_DMS
list2 = task_bracketing_index_per_session_NTS_DMS
list3 = task_bracketing_index_per_session_DA_DMS
#list2 np.nan delete and delte outliers
list2 = [x for x in list2 if str(x) != 'nan']
list1 = [x for x in list1 if str(x) != 'nan']
list3 = [x for x in list3 if str(x) != 'nan']
list2 = [x for x in list2 if x < 3]
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
print(mean(list1))
print(mean(list2))

# Perform an independent t-test
t_stat, p_val = ttest_ind(list1, list2)
print("P-value for t-test list1 and 2:", p_val)
t_stat, p_val = ttest_ind(list2, list3)
print("P-value for t-test list2 and 3:", p_val)
# Plot the data
sns.violinplot(data=[list1, list2, list3], palette=["lightgreen","lightblue", "indianred"])
plt.xticks([0, 1, 2], ["Pnoc", "NTS", "Da"])
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.ylabel("Task Bracketing Index")
plt.ylim([-6,6])
plt.title("Task Bracketing Index in DMS")
plt.text(0.5, 5.5, "****", horizontalalignment='center', verticalalignment='center')
plt.text(1.5, 5.5, "****", horizontalalignment='center', verticalalignment='center')

#t-stat
plt.savefig('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/Task_bracketing_index_PNOC_NTS_DA_DMS.pdf')
plt.show()

#%%
#plot bocc plot for task bracketing index
plt.clf()
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
plt.rcParams['font.size']= 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
# Example lists
list1 = task_bracketing_index_per_session_PNOC_DLS
list2 = task_bracketing_index_per_session_NTS_DLS
list3 = task_bracketing_index_per_session_DA_DLS
#list2 np.nan delete and delte outliers
list2 = [x for x in list2 if str(x) != 'nan']
list1 = [x for x in list1 if str(x) != 'nan']
list3 = [x for x in list3 if str(x) != 'nan']
list2 = [x for x in list2 if x < 3]
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
print(mean(list1))
print(mean(list2))

# Perform an independent t-test
t_stat, p_val = ttest_ind(list1, list2)
print("P-value for t-test list1 and 2:", p_val)
t_stat, p_val = ttest_ind(list2, list3)
print("P-value for t-test list2 and 3:", p_val)
# Plot the data
sns.violinplot(data=[list1, list2, list3], palette=["lightgreen","lightblue", "indianred"])
plt.xticks([0, 1, 2], ["Pnoc", "NTS", "Da"])
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.ylabel("Task Bracketing Index")
plt.ylim([-6,6])
plt.title("Task Bracketing Index in DLS")
plt.text(0.5, 5.5, "****", horizontalalignment='center', verticalalignment='center')
plt.text(1.5, 5.5, "****", horizontalalignment='center', verticalalignment='center')

#t-stat
plt.savefig('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/Task_bracketing_index_PNOC_NTS_DA_DLS.pdf')
plt.show()
