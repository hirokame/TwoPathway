import numpy as np
import pandas as pd
import tqdm
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import style
import numpy as np
import glob
from scipy.optimize import curve_fit
from datetime import datetime

#%% load the q-action value and probability

animal_types = ["PNOC", "NTS"]
animal_type= animal_types[0]
all_lists = glob.glob("/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/SessionInfo{}_Beh_Sig_updated/*".format(animal_type))

all_lists.sort()
#%%
animal_type= animal_types[1]

all_lists_NTS = glob.glob("/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/Data/SessionInfo{}_Beh_Sig_updated/*".format(animal_type))
all_lists_NTS.sort()
all_lists.extend(all_lists_NTS)
#%%
#%%
def moving_average(lst, window_size =2):
    result = []
    for i in range(len(lst) - window_size + 1):
        avg = (lst[i] + lst[i+1]) / window_size
        result.append(avg)
    return result
def sigmoid_fit(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x - x0))) + b
    return y
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_per_mouse_df(every_df, color, legend_name):
    #range_df = (np.percentile(every_df['Q Action Values'], 5), np.percentile(every_df['Q Action Values'], 95))
    #print(range_df)
    hist, bin_edges = np.histogram(every_df['Q Action Values'], bins=9)
    every_df['Mouse_Right_bin'] = pd.cut(every_df['Q Action Values'], bins=bin_edges, labels=False)
    new_bin_edges = moving_average(bin_edges)
    group_per_action_value_df = every_df.groupby('Mouse_Right_bin').mean()
    std_per_action_value_df = every_df.groupby('Mouse_Right_bin').std()
    std = [x / np.sqrt(y) for x, y in zip(std_per_action_value_df['Mouse_Right'], hist)]
    plt.errorbar(new_bin_edges, group_per_action_value_df['Mouse_Right'], yerr=std, fmt='o', capsize=0.1, elinewidth=3,
                 alpha=0.5, color=color, label = legend_name)
    y= group_per_action_value_df['Mouse_Right'].values
    # Initial guess for the parameters: [L, x0, k, b]
    initial_guess = [max(y), np.median(new_bin_edges), 1, min(y)]
        # Perform the curve fit
    popt, pcov = curve_fit(sigmoid_fit, new_bin_edges, y, p0=initial_guess)

    # Create a smooth x-axis for plotting the fitted curve
    x_smooth = np.linspace(min(new_bin_edges), max(new_bin_edges), 500)
    y_smooth = sigmoid_fit(x_smooth, *popt)
    plt.plot(x_smooth, y_smooth, color=color, alpha=0.5)
#%%

today = datetime.today().strftime('%y%m%d')

# Define a list of color hex codes
mouse_colors = [
    '#1f77b4',  # Vivid Blue
    '#ff7f0e',  # Bright Orange
    '#2ca02c',  # Leaf Green
    '#d62728',  # Strong Red
    '#9467bd',  # Deep Purple
    '#8c564b',  # Muted Brown
    '#e377c2',  # Soft Pink
    '#7f7f7f',  # Medium Gray
    '#bcbd22',  # Lime Green
    '#17becf',  # Sky Blue
    '#f7b6d2'   # Light Pink
]
plt.figure(figsize=(6,6))

x = np.linspace(-5, 5, 100)
y = sigmoid(x)  # Apply the sigmoid function to each value in x
# plot the sigmoid function with dashed line
plt.plot(x, y, color='white')
plt.rcParams['font.size']= 14
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for idx, folder_path in enumerate(all_lists):
    each_list_path = glob.glob("{}/*.csv".format(folder_path))
    every_df = pd.DataFrame()
    for path in each_list_path:
        df = pd.read_csv(path)
        every_df = pd.concat([every_df, df[['Mouse_Right', 'Q Action Values']]], axis=0)
    every_df = every_df.reset_index(drop=True)

    if idx < 5:
        label_name = "PNOC"
    else:
        label_name = "NTS"
    plot_per_mouse_df(every_df, mouse_colors[idx], 'Mouse{}_{}'.format(idx + 1,label_name))
plt.xlabel("Action value")
plt.ylabel("Fraction choice of right port")
plt.xticks([-5,0,5])
plt.yticks([0,0.5,1])
plt.xlim([-5.1,5.1])
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
#plt.legend()
plt.savefig('/Users/gunahn/Desktop/MIT/Habit_Breaking_Da_Ast/result/{}_{}_{}_without_legend.pdf'.format(today,"Two_pathway", "Q-learning"))
plt.show()



