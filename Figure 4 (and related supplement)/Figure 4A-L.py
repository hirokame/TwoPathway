#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:09:21 2022

@author: iakovos
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:45:49 2022

@author: Iakovos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
import os
import re
import sys
import scipy.stats as stats
import matplotlib.patches as mpatches
import inspect
#from scipy import stats


sys.path.append(r'/Users/iakovos/opt/anaconda3/envs/pyiak/fipIak0616122')


plt.rcParams['figure.figsize'] = [15, 5]  
plt.rcParams['font.size']= 14


#BFname = "/Volumes/iak_WD_2304/opto_230928_stim_593_GFAP_bonsai/PNOC_Flpo_DA3h_DIO_ChRmsone_CPu_opto_CPu1_CPuStim4_Mice2_40hz_160p_ITI_60_90_2023-10-02T13_33_09.csv";
#DLCname = "/Volumes/iak_WD_2304/small_open_field_Nov23-IAK-2023-11-21/videos/PNOC_Flpo_DA3h_DIO_ChRmsone_CPu_opto_CPu1_CPuStim4_Mice2_40hz_160p_ITI_60_90_2023-10-02T13_33_10DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_filtered.csv";
#DLCcall = 'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_filtered'

##### aligned_df ##########


def save_current_script(directory, script_name):
    # Get the current script's source code
    current_script = inspect.getsource(inspect.getmodule(inspect.currentframe()))

    # Define the path for the new script file
    script_path = os.path.join(directory, script_name)

    # Write the source code to the new file
    with open(script_path, 'w') as file:
        file.write(current_script)
    print(f"Script saved as: {script_path}")
    
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_ylim())
    y_vals = intercept + slope * x_vals
    plt.plot(y_vals, x_vals, '--')

def import_BF_csv(bf_file):
    BF = pd.read_csv(bf_file)

    # Identify transition points where Item2 changes from True to False
    BF['transition'] = BF['Item2'].astype(bool).ne(BF['Item2'].astype(bool).shift())
    transitions = BF[BF['transition'] & ~BF['Item2']].index

    # Handle duplicates, especially at transition points
    for idx in transitions:
        if idx > 0 and BF['Item1.Value'].iloc[idx] == BF['Item1.Value'].iloc[idx - 1]:
            BF.drop(idx - 1, inplace=True)

    BF = BF.drop(columns=['transition'])  # Drop the transition column
    BF = BF.drop_duplicates(subset='Item1.Value', keep='first')
    return BF

def import_DLC_csv(dlc_file):
    df = pd.read_csv(dlc_file, header=[0, 1])
    return df
def custom_print(*args, directory, **kwargs):
    # Construct the log file path
    log_file_path = os.path.join(directory, 'output_log.txt')
    
    # Print to console
    print(*args, **kwargs)
    
    # Write to log file
    with open(log_file_path, 'a') as f:
        print(*args, **kwargs, file=f)
        
def get_significance_marker(p_value):
    if p_value < 0.001:
        return '***'  # Highly significant
    elif p_value < 0.01:
        return '**'   # Very significant
    elif p_value < 0.05:
        return '*'    # Significant
    else:
        return ''     # Not significant



def plot_vector_directions(aligned_df, false_periods, Pdf_directory, dlc_base_name):

    def plot_length_histogram_1(axis, data_pre, data_during, data_post, title):
        bins = np.linspace(0, max(data_pre + data_during + data_post), 20)
        axis.hist(data_pre, bins=bins, color='green', alpha=0.5, label='Pre')
        axis.hist(data_during, bins=bins, color='red', alpha=0.5, label='During')
        axis.hist(data_post, bins=bins, color='blue', alpha=0.5, label='Post')
        axis.set_xlabel('Length cm')
        axis.set_ylabel('Count')
        axis.set_title(f'Histogram of {title} Vector Lengths')
        axis.legend()

    # Create a figure with subplots
    fig = plt.figure(figsize=(56, 12))
    # Add regular subplots for vector direction plots and length histograms
    axes = [fig.add_subplot(1, 6, i) for i in [1, 3, 4, 5]]
    # Add a polar subplot for the histogram of filtered angles at position 2
    axes.insert(1, fig.add_subplot(1, 6, 2, polar=True))
    # Add the sixth subplot for the box plot comparison
    ax_violin = fig.add_subplot(1, 6, 6)

    # Initialize lists to store raw data
    vectors_pre = []
    vectors_during = []
    vectors_post = []

    # Process data for each period
    for start, end in false_periods:
        pre_start = max(0, start - (end - start))
        post_end = min(len(aligned_df), end + (end - start))

        for i in range(pre_start, post_end):
            # Data extraction and processing
            # [Extracting baseTail, CBody, Neck, Snout]
            
            baseTail = np.array([(pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail'][i], errors='coerce')),
                     (pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail.1'][i], errors='coerce'))]) 

            CBody = np.array([(pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody'][i], errors='coerce')),
                               (pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody.1'][i], errors='coerce'))]) 
            
            Neck = np.array([(pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck'][i], errors='coerce')),
                              (pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck.1'][i], errors='coerce'))]) 
            
            Snout = np.array([(pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout'][i], errors='coerce')),
                               (pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout.1'][i], errors='coerce'))])
            
            baseTail_CBody = CBody - baseTail
            CBody_Neck = Neck - CBody
            Neck_Snout = Snout - Neck
            
    
            # Store raw data
            if i < start:
                vectors_pre.append((baseTail_CBody, CBody_Neck, Neck_Snout))
            elif start <= i < end:
                vectors_during.append((baseTail_CBody, CBody_Neck, Neck_Snout))
            else:
                vectors_post.append((baseTail_CBody, CBody_Neck, Neck_Snout))
    
    # Function to filter out vectors based on length
    def filter_vectors(vectors):
        # Initialize lists to store rotated vectors, their lengths, and angles
        rotated_vectors = []
        lengths_baseTail_CBody = []
        lengths_CBody_Neck = []
        lengths_Neck_Snout = []
        angles = []
    
        # Rotate all vectors and calculate lengths and angles
        for baseTail_CBody, CBody_Neck, Neck_Snout in vectors:
            angle = np.arctan2(CBody_Neck[0], CBody_Neck[1])
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                        [np.sin(angle), np.cos(angle)]])
            rotated_baseTail_CBody = np.dot(rotation_matrix, baseTail_CBody)
            rotated_CBody_Neck = np.dot(rotation_matrix, CBody_Neck)
            rotated_Neck_Snout = np.dot(rotation_matrix, Neck_Snout)
            rotated_vectors.append((rotated_baseTail_CBody, rotated_CBody_Neck, rotated_Neck_Snout))
    
            lengths_baseTail_CBody.append(np.linalg.norm(rotated_baseTail_CBody))
            lengths_CBody_Neck.append(np.linalg.norm(rotated_CBody_Neck))
            lengths_Neck_Snout.append(np.linalg.norm(rotated_Neck_Snout))
            angle_degrees_x = np.degrees(np.arctan2(rotated_Neck_Snout[0], rotated_Neck_Snout[1]))
            angles.append((angle_degrees_x + 360) % 360)  # Normalize angle to [0, 360)
            # Calculate angle in degrees from the rotated Neck_Snout vector
       
    
        # Determine bounds for filtering
        q1, q3 = np.percentile(lengths_Neck_Snout, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    
        # Filter the rotated vectors based on the length of Neck_Snout
        filtered_vectors = []
        filtered_angles = []
        filtered_lengths = []
        for i, vector in enumerate(rotated_vectors):
            if lower_bound <= lengths_Neck_Snout[i] <= upper_bound:
                filtered_vectors.append(vector)
                filtered_angles.append(angles[i])
                filtered_lengths.append(lengths_Neck_Snout[i])
    
        return (filtered_vectors, filtered_angles, filtered_lengths, 
            lengths_baseTail_CBody, lengths_CBody_Neck, lengths_Neck_Snout)
      
    # Function to calculate the average angle of vectors
    def calculate_average_angle(vectors):
        sum_sin, sum_cos = 0, 0
        for _, _, Neck_Snout in vectors:
            angle = np.arctan2(Neck_Snout[1], Neck_Snout[0])
            sum_sin += np.sin(angle)
            sum_cos += np.cos(angle)
        return np.arctan2(sum_sin, sum_cos)
    
    # Call filter_vectors for each period
    (filtered_vectors_pre, filtered_angles_pre, filtered_lengths_pre, 
     lengths_baseTail_CBody_pre, lengths_CBody_Neck_pre, lengths_Neck_Snout_pre) = filter_vectors(vectors_pre)

    (filtered_vectors_during, filtered_angles_during, filtered_lengths_during, 
     lengths_baseTail_CBody_during, lengths_CBody_Neck_during, lengths_Neck_Snout_during) = filter_vectors(vectors_during)

    (filtered_vectors_post, filtered_angles_post, filtered_lengths_post, 
     lengths_baseTail_CBody_post, lengths_CBody_Neck_post, lengths_Neck_Snout_post) = filter_vectors(vectors_post)

    
    # Plot the filtered and rotated vectors
    for period, period_vectors in zip(['Pre', 'During', 'Post'], [filtered_vectors_pre, filtered_vectors_during, filtered_vectors_post]):
        for baseTail_CBody, CBody_Neck, Neck_Snout in period_vectors:
            # Plotting the vectors
            axes[0].arrow(0, 0, Neck_Snout[0], Neck_Snout[1], color={'Pre': 'Green', 'During': 'Red', 'Post': 'Blue'}[period], alpha=0.5, head_width=0.02, head_length=0.03)
            axes[0].arrow(0, 0, CBody_Neck[0], CBody_Neck[1], color={'Pre': 'Black', 'During': 'Black', 'Post': 'Black'}[period], alpha=0.5, head_width=0.02, head_length=0.03)
    
    
    # Normalize angles for histogram plotting
    normalized_angles_pre = normalize_angles(filtered_angles_pre)
    normalized_angles_during = normalize_angles(filtered_angles_during)
    normalized_angles_post = normalize_angles(filtered_angles_post)
    
    # Step 1: Calculate the median angle for the pre period Neck-Snout vectors
    median_angle_pre = np.median(normalized_angles_pre)
    
    # Step 2: Adjust all angles by subtracting the median angle
    def adjust_angles(angles, median_angle):
        adjusted_angles = [angle - median_angle for angle in angles]
        return adjusted_angles
    
    adjusted_angles_pre = adjust_angles(normalized_angles_pre, median_angle_pre)
    adjusted_angles_during = adjust_angles(normalized_angles_during, median_angle_pre)
    adjusted_angles_post = adjust_angles(normalized_angles_post, median_angle_pre)
    
    # Step 3: Convert adjusted angles to radians for polar plotting
    radians_pre = np.radians(adjusted_angles_pre)
    radians_during = np.radians(adjusted_angles_during)
    radians_post = np.radians(adjusted_angles_post)
    # Calculate the median angles
    median_pre = np.median(radians_pre)
    median_during = np.median(radians_during)
    median_post = np.median(radians_post)

    
    # Plot the polar histogram with adjusted angles
    bin_edges = np.linspace(-np.pi, np.pi, 36)
    axes[1].hist([radians_pre, radians_during, radians_post], bins=bin_edges, color=['green', 'red', 'blue'], alpha=0.5, label=['Pre', 'During', 'Post'])
    axes[1].set_theta_zero_location('N')  # Set 0 degrees at the top
    axes[1].set_theta_direction(-1)  # Set the direction of angles to clockwise
    axes[1].set_xticks(np.linspace(0, 2 * np.pi, 9))  # Set x-ticks (angles)
    axes[1].set_xticklabels(['0°', '45°', '90°', '135°', '180°', '-135°', '-90°', '-45°', '0°'])
    axes[1].set_title('Polar Histogram of Adjusted Filtered Angles')
    # Generate the histograms and capture the output
    counts, bin_edges, patches = axes[1].hist([radians_pre, radians_during, radians_post], bins=bin_edges, color=['green', 'red', 'blue'], alpha=0.5, label=['Pre', 'During', 'Post'])
    
    # Extract the counts (which is a list of arrays, one for each dataset)
    counts_pre, counts_during, counts_post = counts
    
    # Find the maximum count across all datasets
    max_count = max(max(counts_pre), max(counts_during), max(counts_post))
    
   
    line_length = max_count + 50
    
    # Line for median_pre
    axes[1].plot([0, median_pre], [0, line_length], color='green', lw=4, linestyle='--')
    
    # Line for median_during
    axes[1].plot([0, median_during], [0, line_length], color='red', lw=4, linestyle='--')
    
    # Line for median_post
    axes[1].plot([0, median_post], [0, line_length], color='blue', lw=4, linestyle='--')
    
    # Calculate and display p-values
    t_statistic_pre_vs_during, p_value_pre_vs_during = stats.ttest_ind(radians_during, radians_pre)
    t_statistic_during_vs_post, p_value_during_vs_post = stats.ttest_ind(radians_during, radians_post)
    
    axes[1].text(0.1, 0.2, f'During vs Pre p-value: {p_value_pre_vs_during:.4f}', transform=axes[1].transAxes, color='green')
    axes[1].text(0.1, 0.1, f'During vs Post p-value: {p_value_during_vs_post:.4f}', transform=axes[1].transAxes, color='red')
    
    axes[1].legend()
    
    # Plot 2: Histogram of Adjusted Filtered Angles
    bins = np.arange(-180, 181, 10)
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=2, label='Zero Line')

    #axes[2].hist(adjusted_angles_post, bins=bins, color='blue', alpha=0.5, label='Post')
    axes[2].hist(adjusted_angles_pre, bins=bins, color='green', alpha=0.5, label='Pre')
    axes[2].hist(adjusted_angles_during, bins=bins, color='red', alpha=0.5, label='During')
   
    axes[2].set_xlabel('Angle (degrees)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Histogram of Adjusted Filtered Angles')
    axes[2].set_xlim(-180, 180)
    axes[2].set_xticks(np.arange(-180, 181, 30))
    axes[2].legend()
    
    # Plot length histograms
    plot_length_histogram(axes[3], lengths_Neck_Snout_pre, lengths_Neck_Snout_during, lengths_Neck_Snout_post, 'Neck-Snout')
    plot_length_histogram(axes[4], lengths_CBody_Neck_pre, lengths_CBody_Neck_during, lengths_CBody_Neck_post, 'CBody-Neck')
    
    # Ensure no overlapping of subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    period_colors = {'Pre': 'green', 'During': 'red', 'Post': 'blue'}
    # Prepare data for violin plot
    data_for_violinplot = {
        'baseTail_CBody': [lengths_baseTail_CBody_pre, lengths_baseTail_CBody_during, lengths_baseTail_CBody_post],
        'CBody_Neck': [lengths_CBody_Neck_pre, lengths_CBody_Neck_during, lengths_CBody_Neck_post],
        'Neck_Snout': [lengths_Neck_Snout_pre, lengths_Neck_Snout_during, lengths_Neck_Snout_post]
    }
    
    # Positions for violin plots
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    
    # Plotting violin plots
    for i, (part, lengths) in enumerate(data_for_violinplot.items()):
        # Plot violin plot
        vp = ax_violin.violinplot(lengths, positions=positions[:3], showmeans=False, showmedians=True)
        positions = positions[3:]  # Update positions for the next set of plots
    
        # Coloring each violin plot
        for pc, color in zip(vp['bodies'], period_colors.values()):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.75)
    
        # Perform t-tests
        t_val_pre, p_val_pre = stats.ttest_rel(lengths[0], lengths[1])  # 'Pre' vs 'During'
        t_val_post, p_val_post = stats.ttest_rel(lengths[1], lengths[2])  # 'During' vs 'Post'
    
        # Initialize the text for annotations
        annotation_text = "P-Values:\n"
        
        # Append p-values for each part
        for part, lengths in data_for_violinplot.items():
            # Calculate the p-values for each comparison
            t_val_pre, p_val_pre = stats.ttest_rel(lengths[0], lengths[1])  # 'Pre' vs 'During'
            t_val_post, p_val_post = stats.ttest_rel(lengths[1], lengths[2])  # 'During' vs 'Post'
        
            # Add to the annotation text
            annotation_text += f"{part}:\n"
            annotation_text += f"  Pre vs During: p={p_val_pre:.3f}\n"
            annotation_text += f"  During vs Post: p={p_val_post:.3f}\n"
        
        # Add annotations in a box in the upper left corner
        ax_violin.text(0.05, 0.95, annotation_text, transform=ax_violin.transAxes,
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.5))
        
        
            

    # Set x-tick labels
    ax_violin.set_xticks([2, 6, 10])
    ax_violin.set_xticklabels(['baseTail_CBody', 'CBody_Neck', 'Neck_Snout'])
    
    # Set y-axis label and title
    ax_violin.set_ylabel('Length cm')
    ax_violin.set_title('Comparison of Vector Lengths')
    
    # Add legend for periods
    handles = [mlines.Line2D([], [], color=period_colors[period], marker='s', linestyle='None', markersize=10, label=period) for period in period_colors]
    ax_violin.legend(handles=handles, loc='upper right')
       
    pdf_path = os.path.join(Pdf_directory, f"{dlc_base_name}_vector_directions.pdf")
    plt.tight_layout()
    fig.savefig(pdf_path, format='pdf')
    plt.close(fig)
def normalize_angles(angles):
# Adjust angles to range [-180, 180)
    return [(angle - 360) if angle > 180 else angle for angle in angles]

def plot_length_histogram(ax, lengths_pre, lengths_during, lengths_post, title):
    # Define the range and bin size for the histogram
    min_length = min(min(lengths_pre), min(lengths_during), min(lengths_post))
    max_length = max(max(lengths_pre), max(lengths_during), max(lengths_post))
    bin_size = 0.2
    bins = np.arange(min_length, max_length + bin_size, bin_size)

    # Plot histograms with defined bins
    ax.hist(lengths_pre, bins=bins, color='green', alpha=0.5, label='Pre')
    ax.hist(lengths_during, bins=bins, color='red', alpha=0.5, label='During')
    ax.hist(lengths_post, bins=bins, color='blue', alpha=0.5, label='Post')


    # Draw median lines and shaded outlier regions for each period
    for lengths, color, label in zip([lengths_pre, lengths_during, lengths_post], ['green', 'red', 'blue'], ['Pre', 'During', 'Post']):
        median = np.median(lengths)
        q1, q3 = np.percentile(lengths, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Median line
        ax.axvline(x=median, color=color, linestyle='--', label=f'{label} Median', zorder=3)

        # Shaded outlier regions
        ax.axvspan(min_length, lower_bound, color=color, alpha=0.1, zorder=2)
        ax.axvspan(upper_bound, max_length, color=color, alpha=0.1, zorder=2)

    ax.set_xlabel('Length cm')
    ax.set_ylabel('Count')
    ax.set_title(f'Histogram of {title} Vector Lengths')
    ax.legend()
    
    
def create_collective_boxplot(data_pre, data_post, Mice_colors, pdf_path, dlc_base_name):
    fig, ax = plt.subplots()
    bp = ax.boxplot([data_pre, data_post], patch_artist=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Pre to During', 'During to Post'])
    ax.set_title('Collective Comparative Changes in Lengths of CBody Trajectories')
    ax.set_ylabel('Length Change (cm)')

    # Color code for each change period (Box colors)
    box_colors = ['purple', 'orange']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    
    # Add individual data points with colors based on their source file label
    for i, data in enumerate([data_pre, data_post], 1):
        jittered_x = np.random.normal(i, 0.04, size=len(data))
        for x, y, col in zip(jittered_x, data, Mice_colors):
            ax.scatter(x, y, color=col, s=10, zorder=3)
    # Determine the new y-axis upper limit
    y_max = max(max(data_pre), max(data_post)) * 1.4  # Increasing by 40%
    y_min = min(min(data_pre), min(data_post)) * 1.2 
    # Set the new y-axis limit
    ax.set_ylim(y_min, y_max)
    # Create legend for mice colors
    unique_colors = sorted(set(Mice_colors), key=Mice_colors.index)  # Extract unique colors and preserve order
    legend_handles = [mpatches.Patch(color=color, label=f'Mouse {i+1}') for i, color in enumerate(unique_colors)]
    # Assuming p_v text is at (1.1, 0.75) in axis coordinates
    p_v_text_position = (1.2, 0.75)
    
    # Adjust the following coordinates to position the legend just below the text box
    legend_position = (p_v_text_position[0], p_v_text_position[1] - 0.15)  # Adjust 0.15 as needed
    # Create combined legend
   
    ax.legend(handles=legend_handles, bbox_to_anchor=legend_position, bbox_transform=ax.transAxes, title="Mice", loc='upper right')
    # Perform t-tests and annotate results
    t_val_pre_to_during, p_val_pre_to_during = stats.ttest_1samp(data_pre, 0)
    t_val_during_to_post, p_val_during_to_post = stats.ttest_1samp(data_post, 0)
    # Perform t-tests and annotate results
    ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    p_v = f'Pre_During p={p_val_pre_to_during:.4f}\nDuring_post p={p_val_during_to_post:.4f}'
    ax.text(1.1, 0.75, p_v, transform=ax.transAxes, fontsize=10, horizontalalignment='center')
   
    # Add t-test p-value and asterisks
    # Function to determine the appropriate number of asterisks
    def get_significance_marker(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    # Add annotations with p-value and significance marker
    significance_marker_pre_to_during = get_significance_marker(p_val_pre_to_during)
    significance_marker_during_to_post = get_significance_marker(p_val_during_to_post)
 
    ax.text(1, max(data_pre) * 1.1, f'{significance_marker_pre_to_during}', ha='center', color='purple')
    ax.text(2, max(data_post) * 1.1, f'{significance_marker_during_to_post}', ha='center', color='orange')
 
   
    plt.tight_layout()
    pdf_RT_speed_plot_path = os.path.join(pdf_path, f"{dlc_base_name}_Collective_Comparative_Changes_in_Lengths_of_CBody_Trajectories.pdf")
    fig.savefig(pdf_RT_speed_plot_path, format='pdf')
    plt.close(fig)
    
    
def create_collective_angular_plot(all_rotational_speeds_pre, all_rotational_speeds_during, all_rotational_speeds_post, pdf_path, dlc_base_name):
    # Extracting speeds and labels for each period
    speeds_pre, labels_pre = zip(*all_rotational_speeds_pre)
    speeds_during, labels_during = zip(*all_rotational_speeds_during)
    speeds_post, labels_post = zip(*all_rotational_speeds_post)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create violin plot
    vp = ax.violinplot([speeds_pre, speeds_during, speeds_post], showmeans=True, showmedians=True)
    ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1, zorder=1)
    
    # Hide the lines for distribution and T-bars
    for partname in ('cmaxes', 'cmins', 'cbars'):
        vp[partname].set_visible(False)

    # Color code for each period
    colors = ['green', 'red', 'blue']
    for pc, color in zip(vp['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('gray')
        pc.set_alpha(0.75)

    # Generating the color map for labels
    unique_labels2 = sorted(set(labels_pre + labels_during + labels_post))
    n_labels2 = len(unique_labels2)
    label_to_color2 = plt.cm.get_cmap('viridis', n_labels2)
    label_color_map = {label: label_to_color2(i) for i, label in enumerate(unique_labels2)}

    # Plotting data points with consistent colors across periods
    for i, (period_speeds, period_labels) in enumerate(zip([speeds_pre, speeds_during, speeds_post], [labels_pre, labels_during, labels_post]), 1):
        # Calculate the number of points to plot (20% of the total points)
        num_points_to_plot = int(len(period_speeds) * 0.2)
    
        # Randomly select indices for 20% of the points
        indices_to_plot = np.random.choice(range(len(period_speeds)), num_points_to_plot, replace=False)

        # Generate jittered x values for these points
        jittered_x = np.random.normal(i, 0.04, size=num_points_to_plot)
    
        # Plot only the selected points
        for index, x in zip(indices_to_plot, jittered_x):
            y = period_speeds[index]
            label = period_labels[index]
            ax.scatter(x, y, color=label_color_map[label], alpha=0.1, s=10, zorder=3)


    # Assuming p_v text is at (1.1, 0.75) in axis coordinates
    p_v_text_position = (1.2, 0.75)
    
    # Adjust the following coordinates to position the legend just below the text box
    legend_position = (p_v_text_position[0], p_v_text_position[1] - 0.15)  # Adjust 0.15 as needed
    
    # Create legend
    legend_handles = [mpatches.Patch(color=label_color_map[label], label=f'Mouse {label}') for label in unique_labels2]
    ax.legend(handles=legend_handles, title="Mice", bbox_to_anchor=legend_position, loc='upper right', bbox_transform=ax.transAxes)
    # Perform and annotate statistical tests
    t_val_pre_during, p_val_pre_during = stats.ttest_ind(speeds_pre, speeds_during)
    t_val_during_post, p_val_during_post = stats.ttest_ind(speeds_during, speeds_post)
    
    p_v = f'Pre_During p={p_val_pre_during:.4f}\nDuring_post p={p_val_during_post:.4f}'
    ax.text(1.1, 0.75, p_v, transform=ax.transAxes, fontsize=10, horizontalalignment='center')
    
    # Function to determine the appropriate number of asterisks
    def get_significance_marker(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    # Add annotations with p-value and significance marker
    significance_marker_pre_to_during = get_significance_marker(p_val_pre_during)
    significance_marker_during_to_post = get_significance_marker(p_val_during_post)
    
    # Modify the text positions based on the maximum value among the during period speeds
    max_speed_during = max(speeds_during)
    ax.text(1.5, max_speed_during * 0.2, f'{significance_marker_pre_to_during}', ha='center', color='green')
    ax.text(2.5, max_speed_during * 0.2, f'{significance_marker_during_to_post}', ha='center', color='blue')
    
    # Calculate and draw median lines for scatter plots
    mean_pre = np.mean(speeds_pre)
    mean_during = np.mean(speeds_during)
    mean_post = np.mean(speeds_post)
    ax.axhline(y=mean_pre, color='green', linestyle='dashed', linewidth=0.8, zorder=4)
    ax.axhline(y=mean_during, color='red', linestyle='dashed', linewidth=0.8, zorder=4)
    ax.axhline(y=mean_post, color='blue', linestyle='dashed', linewidth=0.8, zorder=4)
    # Set plot titles and labels
    ax.set_title('Collective Rotational Speeds of Neck-Snout Vector 20% (scatter)')
    ax.set_ylabel('Rotational Speed (rad/s)')
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(-5, 5)
    ax.set_xticklabels(['Pre', 'During', 'Post'])

    # Save the figure
    plt.tight_layout()
    pdf_plot_path = os.path.join(pdf_path, f"{dlc_base_name}_Collective_Angular_Displacement.pdf")
    fig.savefig(pdf_plot_path, format='pdf')
    plt.close(fig)

def create_collective_activity_level_plot(all_activity_levels_pre, all_activity_levels_during, all_activity_levels_post, pdf_path, dlc_base_name):
    # Extract activity levels and labels for each period
    # Assuming the structure of all_activity_levels_* lists is [(immobile_time, mobile_time, running_time, label), ...]
    immobile_pre, mobile_pre, running_pre, labels_pre = zip(*all_activity_levels_pre)
    immobile_during, mobile_during, running_during, labels_during = zip(*all_activity_levels_during)
    immobile_post, mobile_post, running_post, labels_post = zip(*all_activity_levels_post)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plot
    immobile_data = [immobile_pre, immobile_during, immobile_post]
    mobile_data = [mobile_pre, mobile_during, mobile_post]
    running_data = [running_pre, running_during, running_post]

    # Create box plot
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    bp = ax.boxplot(immobile_data + mobile_data + running_data, positions=positions, patch_artist=True)
    
    # Setting colors for each box
   
    for i, box in enumerate(bp['boxes']):
        # Determine the period based on the box index
        
        if i in [0, 3, 6]:  # Pre
            box.set_facecolor('green')
        elif i in [1, 4, 7]:  # During
            box.set_facecolor('red')
        else:  # Post
            box.set_facecolor('blue')
        
    # Proxy artists for periods
    pre_patch = mpatches.Patch(color='green', label='Pre')
    during_patch = mpatches.Patch(color='red', label='During')
    post_patch = mpatches.Patch(color='blue', label='Post')
    # Generating the color map for labels
    unique_labels = sorted(set(labels_pre + labels_during + labels_post))
  
    n_labels = len(unique_labels)
    label_to_color = plt.cm.get_cmap('viridis', n_labels)
    label_color_map = {label: label_to_color(i) for i, label in enumerate(unique_labels)}
  


   # Adding scatter plot with color coding
    for i, activity in enumerate([immobile_data, mobile_data, running_data]):
        for j, data in enumerate(activity):
            x = np.random.normal(1 + 4*i + j, 0.1, size=len(data))
            # Assuming all_activity_levels_pre/during/post are structured as [(immobile_time, mobile_time, running_time, label), ...]
            # Extract labels for the current activity and period
            if i == 0:  # Immobile
                labels = [activity_level[3] for activity_level in all_activity_levels_pre] if j == 0 else \
                         [activity_level[3] for activity_level in all_activity_levels_during] if j == 1 else \
                         [activity_level[3] for activity_level in all_activity_levels_post]
            elif i == 1:  # Mobile
                labels = [activity_level[3] for activity_level in all_activity_levels_pre] if j == 0 else \
                         [activity_level[3] for activity_level in all_activity_levels_during] if j == 1 else \
                         [activity_level[3] for activity_level in all_activity_levels_post]
            else:  # Running
                labels = [activity_level[3] for activity_level in all_activity_levels_pre] if j == 0 else \
                         [activity_level[3] for activity_level in all_activity_levels_during] if j == 1 else \
                         [activity_level[3] for activity_level in all_activity_levels_post]
    
            colors = [label_color_map[label] for label in labels]
            ax.scatter(x, data, color=colors, alpha=0.5, s=10, zorder=3)
    # Proxy artists for labels
    label_artists = [plt.Line2D([0], [0], marker='o', color=label_color_map[label], linestyle='None', markersize=10) for label in unique_labels]

    # Combine period and label artists
    combined_handles = [pre_patch, during_patch, post_patch] + label_artists
    
    # Combine period and label names
    combined_labels = ['Pre', 'During', 'Post'] + [f'Mouse {label}' for label in unique_labels]
    
    # Create combined legend
    ax.legend(handles=combined_handles, labels=combined_labels, loc='upper right', title="Legend")  
    # Assuming p_v text is at (1.1, 0.75) in axis coordinates
    p_v_text_position = (1.2, 0.75)
    
    # Adjust the following coordinates to position the legend just below the text box
    legend_position = (p_v_text_position[0], p_v_text_position[1] - 0.15)  # Adjust 0.15 as needed
    # Create combined legend
    ax.legend(handles=combined_handles, bbox_to_anchor=legend_position, loc='upper right', bbox_transform=ax.transAxes, labels=combined_labels, title="Legend") 
    # Perform and annotate statistical tests
    # Assumed activity_labels are ['Immobile', 'Mobile', 'Running']
    activity_labels = ['Immobile', 'Mobile', 'Running']
    p_values_texts = []

    for i, activity in enumerate([immobile_data, mobile_data, running_data]):
        t_stat_pre_during, p_value_pre_during = stats.ttest_rel(activity[0], activity[1], nan_policy='omit')
        t_stat_during_post, p_value_during_post = stats.ttest_rel(activity[1], activity[2], nan_policy='omit')
        
        p_values_texts.append(f"{activity_labels[i]}:\nPre-During p={p_value_pre_during:.4f}\nDuring-Post p={p_value_during_post:.4f}")

    # Combine all p-value texts into one string and add to plot
    combined_p_value_text = "\n".join(p_values_texts)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.05, 0.95, combined_p_value_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
   # Function to determine the appropriate number of asterisks
    def get_significance_marker(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''

    # Iterate over each activity level
    for i, activity in enumerate(activity_labels):
        # Find the maximum value across all periods for the current activity
        max_activity = max([max(data) for data in [immobile_data, mobile_data, running_data][i]])
    
        # Calculate significance markers
        significance_marker_pre_to_during = get_significance_marker(stats.ttest_rel([immobile_data, mobile_data, running_data][i][0], [immobile_data, mobile_data, running_data][i][1], nan_policy='omit')[1])
        significance_marker_during_to_post = get_significance_marker(stats.ttest_rel([immobile_data, mobile_data, running_data][i][1], [immobile_data, mobile_data, running_data][i][2], nan_policy='omit')[1])
        # Adjust these values as needed for proper positioning
        pre_during_x_pos = 1.5 + 4*i  # Adjust as needed
        during_post_x_pos = 2.5 + 4*i  # Adjust as needed
        # Annotate plot with significance markers
        ax.text(pre_during_x_pos, max_activity, f'{significance_marker_pre_to_during}', ha='center', color='green')
        ax.text(during_post_x_pos, max_activity, f'{significance_marker_during_to_post}', ha='center', color='blue')
    
    # Set y-axis limit to accommodate annotations
    y_max = max([ max_activity * 1.2 for max_activity in [max([max(data) for data in activity]) for activity in [immobile_data, mobile_data, running_data]]])
    ax.set_ylim(-2, y_max)
   
    # Set plot titles and labels
    ax.set_title('Activity Levels Across Periods')
    ax.set_ylabel('Time Spent (seconds)')
    ax.set_xticks([2, 6, 10])
    ax.set_xticklabels(['Immobile', 'Mobile', 'Running'])
   

    # Save the figure
    plt.tight_layout()
    pdf_plot_path = os.path.join(pdf_path, f"{dlc_base_name}_Collective_Activity_Levels.pdf")
    fig.savefig(pdf_plot_path, format='pdf')
    plt.close(fig)
    


file_counter = 1  # Initialize a counter for file labels
file_counter2 = 1  # Initialize a counter for file labels
file_counter3 = 1  # Initialize a counter for file labels
# Initialize lists to store collective data
all_changes_pre_to_during = []
all_changes_during_to_post = []
all_file_labels = [] 

all_rotational_speeds_pre = []
all_rotational_speeds_during = []
all_rotational_speeds_post = []


all_activity_levels_pre = []
all_activity_levels_during = []
all_activity_levels_post = []

#######################

#####################

def process_and_plot(bf_file, dlc_file, pdf_pages, dile_to_color):
    Pdf_directory = dir2
    print('Started process_and_plot')
    try:
        # Dynamically set BFname and DLCname
        BFname = bf_file
        DLCname = dlc_file
        
        # Import DataFrames
        BF = import_BF_csv(BFname)
        DLC = import_DLC_csv(DLCname)
        #dlc_directory = os.path.dirname(DLCname)
        dlc_base_name = os.path.splitext(os.path.basename(DLCname))[0]
    
       
        # Merge DataFrames
        BF['Item1.Value'] = BF['Item1.Value'].astype(str)
        DLC.columns = ['_'.join(col).strip() for col in DLC.columns.values]
        DLC['scorer_bodyparts'] = DLC['scorer_bodyparts'].astype(str)
        aligned_df_1 = pd.merge(BF, DLC, left_on='Item1.Value', right_on='scorer_bodyparts', how='inner')
        # Define the list of columns to be multiplied by 0.0323 for px to cm
        # List of columns to be modified
        columns_to_modify = [
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck.1',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout.1',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail.1',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody',
            'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody.1'
        ]
        
        # Multiply each column by 0.0323
        for col in columns_to_modify:
            if col in aligned_df_1.columns:
                aligned_df_1[col] = pd.to_numeric(aligned_df_1[col], errors='coerce') * pixel_size_
        
        aligned_df = aligned_df_1
        ##############
        
        ######## plot all parts and trials ##########
        # Define body parts and their corresponding columns in aligned_df
        body_parts = ['Tail', 'baseTail', 'CBody', 'neck', 'Snout', 'RFr', 'LFr', 'RBack', 'LBack']
        prefix = 'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_'
        
        # Custom color map for pre, during, and post periods
        cmap = mcolors.ListedColormap(['green', 'red', 'blue'])
        norm = plt.Normalize(0.0, 1.0)
        
        # Function to create color-mapped line segments
        def colorline(x, y, z=None, cmap=cmap, norm=norm):
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))
            z = np.asarray(z)
            segments = make_segments(x, y)
            lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=2)
            return lc
        
        # Function to create line segments from x and y coordinates
        def make_segments(x, y):
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            return segments
        
        # Function to find false periods and corresponding pre and post periods
        def find_periods(item2_column):
            false_periods = []
            start = None
            for i, value in enumerate(item2_column):
                if value == False and start is None:
                    start = i
                elif value == True and start is not None:
                    false_periods.append((start, i))
                    start = None
            if start is not None:
                false_periods.append((start, len(item2_column)))
            return false_periods
        
        # Function to calculate rotation angle to align vector from TailBase to CBody with x-axis
        def calculate_rotation_angle(x_tail, y_tail, x_body, y_body):
            dx = x_body - x_tail
            dy = y_body - y_tail
            angle = np.arctan2(dy, dx)
            return -angle  # Negative for clockwise rotation
        
        # Function to rotate points around the origin (0,0)
        def rotate_points(x, y, angle):
            cos_angle, sin_angle = np.cos(angle), np.sin(angle)
            x_rot = x * cos_angle - y * sin_angle
            y_rot = x * sin_angle + y * cos_angle
            return x_rot, y_rot
         
       
        if aligned_df.empty:
            
            # Use custom_print and pass the directory
            custom_print(f"No data to plot for files: {bf_file} and {dlc_file}", directory=Pdf_directory)
            return

        # Find false periods
        false_periods = find_periods(aligned_df['Item2'])
        
        # Filter out periods that are less than 10 frames long
        false_periods = [period for period in false_periods if period[1] - period[0] >= 10]
        
        # Check if there are enough valid false periods
        min_trials_required = 10  # Set the minimum number of trials required
        if len(false_periods) < min_trials_required:
            custom_print(f"Not enough valid trials found for files: {bf_file} and {dlc_file}", directory=Pdf_directory)
            return


        # Check if there are any false periods
        if not false_periods:
            custom_print(f"No false periods found for files: {bf_file} and {dlc_file}", directory=Pdf_directory)
            return
        #####################
        ##########prepro for all till here###########
        #####################
    
        
        # Construct PDF file path
        #pdf_file_path = os.path.join(dlc_directory, f"{dlc_base_name}_plots.pdf")
        
        
        # Plot for each body part
        n_parts = len(body_parts)
        cols = 3
        rows = n_parts // cols + (n_parts % cols > 0)
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
        # Add file names as title for the first plot page
        title = f"BF file: {os.path.basename(BFname)}\nDLC file: {os.path.basename(DLCname)}"
        fig.suptitle(title, fontsize=8)
        
       
        for i, part in enumerate(body_parts):
            ax = axes.flatten()[i]
            x_col = prefix + part
            y_col = prefix + part + '.1'

    
            if x_col in aligned_df.columns and y_col in aligned_df.columns:
                x = pd.to_numeric(aligned_df[x_col], errors='coerce') # *0.0323 px to cm
                y = pd.to_numeric(aligned_df[y_col], errors='coerce') 
    
                # Plot the full trajectory in light gray
                ax.plot(x, y, color='gray', alpha=0.5, label=f'{part} Full Trajectory')
    
                # Initialize an empty array for color values
                z = np.zeros(len(x))
    
                for start, end in false_periods:
                    
                    pre_end = max(0, start - (end - start))
                    post_start = min(len(aligned_df), end + (end - start))
                  
    
                    # Assign color values based on the period
                    z[pre_end:start] = 0.0  # pre (green)
                    z[start:end] = 0.5  # during (red)
                    z[end:post_start] = 1.0  # post (blue)
    
                    # Plot color-coded trajectory for each body part
                    lc = colorline(x[pre_end:post_start], y[pre_end:post_start], z[pre_end:post_start])
                    ax.add_collection(lc)
    
                # Set plot limits
                ax.set_xlim(np.nanmin(x), np.nanmax(x))
                ax.set_ylim(np.nanmin(y), np.nanmax(y))
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f'Trajectory of {part} \n around Item2 False')
                ax.set_xlabel('X Coordinate (cm)')
                ax.set_ylabel('Y Coordinate (cm)')

            
        pdf_body_parts_path = os.path.join(Pdf_directory, f"{dlc_base_name}_body_parts.pdf")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(pdf_body_parts_path, format='pdf')
        plt.close(fig)
        
        
        x_col = 'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody'
        y_col = 'DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody.1'
        
        # Custom color map
        cmap = mcolors.ListedColormap(['green', 'red', 'blue'])
        norm = plt.Normalize(0.0, 1.0)
        
        
        # Find false periods
        false_periods = find_periods(aligned_df['Item2'])
        
        
        # Determine the layout of subplots (rows, columns)
        n_trials = len(false_periods)
        cols = 2
        rows = n_trials // cols + (n_trials % cols > 0)
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
        axes = axes.flatten()
        
        # Plot each trial in a separate subplot
        for i, (start, end) in enumerate(false_periods):
            if i >= len(axes):
                break  # Prevents IndexError if there are more periods than subplots
        
            ax = axes[i]
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            x = pd.to_numeric(aligned_df[x_col], errors='coerce') # *0.0323 px to cm
            y = pd.to_numeric(aligned_df[y_col], errors='coerce')
        
            # Create a color array with numerical values
            z = np.full(len(x), 0.0)  # pre (0)
            z[start:end] = 0.5  # during (0.5)
            z[end:post_end] = 1.0  # post (1)
        
            # Create a line collection with color coding
            lc = colorline(x[pre_start:post_end], y[pre_start:post_end], z[pre_start:post_end])
            ax.add_collection(lc)
        
            ax.set_xlim(x[pre_start:post_end].min(), x[pre_start:post_end].max())
            ax.set_ylim(y[pre_start:post_end].min(), y[pre_start:post_end].max())
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'Trial {i+1}')
            ax.set_xlabel('X Coordinate (cm)')
            ax.set_ylabel('Y Coordinate (cm)')
        
        # Hide unused subplots if any
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Save the figure as a PDF
        pdf_trials_path = os.path.join(Pdf_directory, f"{dlc_base_name}_trials.pdf")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(pdf_trials_path, format='pdf')
        plt.close(fig)
                
       
        # Ensure x and y columns are numeric and handle non-numeric values
        aligned_df[x_col] = pd.to_numeric(aligned_df[x_col], errors='coerce') # *0.0323 px to cm
        aligned_df[y_col] = pd.to_numeric(aligned_df[y_col], errors='coerce')
        
        # Drop NaN values from x and y columns
        aligned_df.dropna(subset=[x_col, y_col], inplace=True)
        
        # Create a single plot for all trials
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each trial's trajectory
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
            
            x = aligned_df[x_col]
            y = aligned_df[y_col]
        
            # Create a color array with numerical values for pre, during, and post
            z = np.full(len(x), 0.0)  # pre (0)
            z[start:end] = 0.5  # during (0.5)
            z[end:post_end] = 1.0  # post (1)
        
            # Create and add a line collection with color coding
            lc = colorline(x[pre_start:post_end], y[pre_start:post_end], z[pre_start:post_end])
            ax.add_collection(lc)
        
        # Set plot limits and labels
        ax.set_xlim(aligned_df[x_col].min(), aligned_df[x_col].max())
        ax.set_ylim(aligned_df[y_col].min(), aligned_df[y_col].max())
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Combined Trajectories of All Trials')
        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')

        
        # Save the figure as a PDF
        pdf_body_parts_path = os.path.join(Pdf_directory, f"{dlc_base_name}_trials_c.pdf")
        plt.tight_layout()
        fig.savefig(pdf_body_parts_path, format='pdf')
        plt.close(fig)
        

        ########### all trials in 1 plot aligned to the vector of baseTale-CBody########
        ############
        
      
        # Reorient and plot each trial
        fig, ax = plt.subplots(figsize=(15, 15))  # Increased plot size
        # Initialize variables for plot limits
        max_x, min_x, max_y, min_y = 0, 0, 0, 0
        # Initialize a variable to store the duration text
        duration_text = ""
        
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
            # Calculate the duration of the false period
            duration = (end - start)/30
            # Add the duration to the duration text, along with a label for the trial
            duration_text += f"Trial {i+1}: {duration} sec\n"
             
            # Extract coordinates for Neck and Snout
            x_neck = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck'], errors='coerce') # *0.0323 px to cm
            y_neck = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck.1'], errors='coerce')
            x_snout = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout'], errors='coerce')
            y_snout = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout.1'], errors='coerce')
            # Extract coordinates for TailBase and CBody
            x_tail = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail'], errors='coerce')
            y_tail = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail.1'], errors='coerce')
            x_body = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody'], errors='coerce')
            y_body = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody.1'], errors='coerce')
            
           

            # Calculate rotation angle using Neck and Snout one frame before the first False
            one_before_false = start - 1
            angle = calculate_rotation_angle(x_neck[one_before_false], y_neck[one_before_false], x_snout[one_before_false], y_snout[one_before_false])
        
        
            # Apply rotation and translation to CBody
            x_rotated, y_rotated = rotate_points(x_body - x_body[one_before_false], y_body - y_body[one_before_false], angle)
            
            # Update plot limits
            max_x = max(max_x, x_rotated[pre_start:post_end].max(), -x_rotated[pre_start:post_end].min())
            min_x = min(min_x, x_rotated[pre_start:post_end].min(), -x_rotated[pre_start:post_end].max())
            max_y = max(max_y, y_rotated[pre_start:post_end].max(), -y_rotated[pre_start:post_end].min())
            min_y = min(min_y, y_rotated[pre_start:post_end].min(), -y_rotated[pre_start:post_end].max())
        
            # Color array for pre, during, and post
            z = np.full(len(x_body), 0.0)  # pre (0)
            z[start:end] = 0.5  # during (0.5)
            z[end:post_end] = 1.0  # post (1)
        
            # Add line collection for this trial
            lc = colorline(x_rotated[pre_start:post_end], y_rotated[pre_start:post_end], z[pre_start:post_end])
            ax.add_collection(lc)
            
            # Optionally, display the duration near the start of each trajectory
            ax.text(x_rotated[start], y_rotated[start], f"{duration} sec", fontsize=8, verticalalignment='center')
            
            # Optionally, draw an arrow representing the vector from Neck to Snout one frame before False
            arrow_dx = x_snout[one_before_false] - x_neck[one_before_false]
            arrow_dy = y_snout[one_before_false] - y_neck[one_before_false]
            arrow_dx_rot, arrow_dy_rot = rotate_points(arrow_dx, arrow_dy, angle)
            ax.arrow(x_rotated[one_before_false], y_rotated[one_before_false], arrow_dx_rot, arrow_dy_rot,
                     head_width=0.05 * max_x, head_length=0.1 * max_y, fc='black', ec='black')
            # Draw an arrow representing the vector from TailBase to CBody one frame before False
            arrow_dx = x_body[one_before_false] - x_tail[one_before_false]
            arrow_dy = y_body[one_before_false] - y_tail[one_before_false]
            arrow_dx_rot, arrow_dy_rot = rotate_points(arrow_dx, arrow_dy, angle)
            ax.arrow(x_rotated[one_before_false], y_rotated[one_before_false], arrow_dx_rot, arrow_dy_rot,
                     head_width=0.05 * max_x, head_length=0.1 * max_y, fc='gray', ec='gray')
        
        # Set plot limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Draw dashed lines for x and y axes at the origin
        ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
        ax.axvline(x=0, color='black', linestyle='dashed', linewidth=1)
        
        # Display the duration text on the plot
        # Position it in a corner or a suitable location
        ax.text(0.05, 0.95, duration_text.strip(), transform=ax.transAxes, fontsize=10, verticalalignment='top')

        
        # Set aspect, title, and labels
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Reoriented CBody Trajectories of All Trials with Neck to Snout Vectors')
        ax.set_xlabel('X Coordinate (cm)')
        ax.set_ylabel('Y Coordinate (cm)')
        
        pre_line = mlines.Line2D([], [], color='green', marker='_', markersize=15, label='Pre')
        during_line = mlines.Line2D([], [], color='red', marker='_', markersize=15, label='During')
        post_line = mlines.Line2D([], [], color='blue', marker='_', markersize=15, label='Post')
        snout_arrow = mlines.Line2D([], [], color='black', marker='>', markersize=15, label='Snout-Neck Vector')
        cbody_arrow = mlines.Line2D([], [], color='gray', marker='>', markersize=15, label='CBody-TailBase Vector')
        ax.legend(handles=[pre_line, during_line, post_line, snout_arrow, cbody_arrow], loc='upper right')
        
        
        # Save the figure as a PDF
        pdf_body_parts_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Reorient_trial_CBody.pdf")
        plt.tight_layout()
        fig.savefig(pdf_body_parts_path, format='pdf')
        plt.close(fig)
        
       ########### all trials in 1 plot aligned to the vector of Snout ########
       
        # Reorient and plot each trial
        fig, ax = plt.subplots(figsize=(15, 15))  # Increased plot size
        # Initialize variables for plot limits
        max_x, min_x, max_y, min_y = 0, 0, 0, 0
         
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            # Extract coordinates for Neck and Snout
            x_neck = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck'], errors='coerce') # *0.0323 px to cm
            y_neck = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck.1'], errors='coerce')
            x_snout = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout'], errors='coerce')
            y_snout = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout.1'], errors='coerce')
            # Extract coordinates for TailBase and CBody
            x_tail = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail'], errors='coerce')
            y_tail = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_baseTail.1'], errors='coerce')
            x_body = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody'], errors='coerce')
            y_body = pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_CBody.1'], errors='coerce')
            
        
            # Calculate rotation angle using Neck and Snout one frame before the first False
            one_before_false = start - 1
            angle = calculate_rotation_angle(x_neck[one_before_false], y_neck[one_before_false], x_snout[one_before_false], y_snout[one_before_false])
        
        
            # Apply rotation and translation to CBody
            x_rotated_S, y_rotated_S = rotate_points(x_snout - x_snout[one_before_false], y_snout - y_snout[one_before_false], angle)
            
            # Update plot limits
            max_x = max(max_x, x_rotated_S[pre_start:post_end].max(), -x_rotated_S[pre_start:post_end].min())
            min_x = min(min_x, x_rotated_S[pre_start:post_end].min(), -x_rotated_S[pre_start:post_end].max())
            max_y = max(max_y, y_rotated_S[pre_start:post_end].max(), -y_rotated_S[pre_start:post_end].min())
            min_y = min(min_y, y_rotated_S[pre_start:post_end].min(), -y_rotated_S[pre_start:post_end].max())
        
            # Color array for pre, during, and post
            z = np.full(len(x_snout), 0.0)  # pre (0)
            z[start:end] = 0.5  # during (0.5)
            z[end:post_end] = 1.0  # post (1)
        
            # Add line collection for this trial
            lc = colorline(x_rotated_S[pre_start:post_end], y_rotated_S[pre_start:post_end], z[pre_start:post_end])
            ax.add_collection(lc)
        
            # Optionally, draw an arrow representing the vector from Neck to Snout one frame before False
            arrow_dx = x_snout[one_before_false] - x_neck[one_before_false]
            arrow_dy = y_snout[one_before_false] - y_neck[one_before_false]
            arrow_dx_rot, arrow_dy_rot = rotate_points(arrow_dx, arrow_dy, angle)
            ax.arrow(x_rotated_S[one_before_false], y_rotated_S[one_before_false], arrow_dx_rot, arrow_dy_rot,
                     head_width=0.05 * max_x, head_length=0.1 * max_y, fc='black', ec='black')
            # Draw an arrow representing the vector from TailBase to CBody one frame before False
            arrow_dx = x_body[one_before_false] - x_tail[one_before_false]
            arrow_dy = y_body[one_before_false] - y_tail[one_before_false]
            arrow_dx_rot, arrow_dy_rot = rotate_points(arrow_dx, arrow_dy, angle)
            ax.arrow(x_rotated_S[one_before_false], y_rotated_S[one_before_false], arrow_dx_rot, arrow_dy_rot,
                     head_width=0.05 * max_x, head_length=0.1 * max_y, fc='gray', ec='gray')
        
        # Set plot limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Draw dashed lines for x and y axes at the origin
        ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
        ax.axvline(x=0, color='black', linestyle='dashed', linewidth=1)
        
        # Set aspect, title, and labels
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Reoriented Snout Trajectories of All Trials with Neck to Snout Vectors')
        ax.set_xlabel('X Coordinate (cm)')
        ax.set_ylabel('Y Coordinate (cm)')
        
        pre_line = mlines.Line2D([], [], color='green', marker='_', markersize=15, label='Pre')
        during_line = mlines.Line2D([], [], color='red', marker='_', markersize=15, label='During')
        post_line = mlines.Line2D([], [], color='blue', marker='_', markersize=15, label='Post')
        snout_arrow = mlines.Line2D([], [], color='black', marker='>', markersize=15, label='Snout-Neck Vector')
        cbody_arrow = mlines.Line2D([], [], color='gray', marker='>', markersize=15, label='CBody-TailBase Vector')
        ax.legend(handles=[pre_line, during_line, post_line, snout_arrow, cbody_arrow], loc='upper right')
        
        
        # Save the figure as a PDF
        pdf_body_parts_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Reorient_trial_Snout.pdf")
        plt.tight_layout()
        fig.savefig(pdf_body_parts_path, format='pdf')
        plt.close(fig)
            
        
        ################## box plot comparing the lengths of trajectories during the pre, during, and post periods across all trials#######
        # Function to calculate trajectory length (929px/30cm=30.97px=1cm, x0.0323)
        def calculate_length(x, y):
            return np.sqrt(np.diff(x)**2 + np.diff(y)**2).sum()
        
       # Initialize lists to store lengths for each period
        lengths_changes = []
        
        # Loop through each trial to calculate length changes
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            # Calculate lengths for each period
            length_pre = calculate_length(x_rotated[pre_start:start], y_rotated[pre_start:start])
            length_during = calculate_length(x_rotated[start:end], y_rotated[start:end])
            length_post = calculate_length(x_rotated[end:post_end], y_rotated[end:post_end])
        
            # Calculate changes in length from pre to during and during to post
            change_pre_to_during = length_during - length_pre
            change_during_to_post = length_post - length_during
        
            lengths_changes.append([change_pre_to_during, change_during_to_post])
        
    
        # Convert to DataFrame and extend the collective data lists
        df_changes = pd.DataFrame(lengths_changes, columns=['Pre_to_During', 'During_to_Post'])
        df_changes['File_Label'] = file_counter  # Assign current file label
        all_changes_pre_to_during.extend(df_changes['Pre_to_During'].tolist())
        all_changes_during_to_post.extend(df_changes['During_to_Post'].tolist())
        all_file_labels.extend([file_counter] * len(df_changes))  # Store the file label
        
       
        # Create a box plot
        fig, ax = plt.subplots()
        bp = ax.boxplot([df_changes['Pre_to_During'], df_changes['During_to_Post']], patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Pre to During', 'During to Post'])
        ax.set_title('Comparative Changes in Lengths of CBody Trajectories')
        ax.set_ylabel('Length Change cm')
        
        # Color code for each change period
        colors = ['purple', 'orange']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add individual data points
        for i, data in enumerate([df_changes['Pre_to_During'], df_changes['During_to_Post']], 1):
            jittered_x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(jittered_x, data, color='black', s=10, zorder=3)
        
        # Perform t-tests and annotate results
        t_val_pre_to_during, p_val_pre_to_during = stats.ttest_1samp(df_changes['Pre_to_During'], 0)
        t_val_during_to_post, p_val_during_to_post = stats.ttest_1samp(df_changes['During_to_Post'], 0)
        
        # Add t-test p-value and asterisks
        # Function to determine the appropriate number of asterisks
        def get_significance_marker(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ''
        
        # Add annotations with p-value and significance marker
        significance_marker_pre_to_during = get_significance_marker(p_val_pre_to_during)
        significance_marker_during_to_post = get_significance_marker(p_val_during_to_post)
        
        ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
        
        
        
        ax.text(1, max(df_changes['Pre_to_During']) * 1.1, f'{significance_marker_pre_to_during}\np={p_val_pre_to_during:.3f}', ha='center', color='purple')
        ax.text(2, max(df_changes['During_to_Post']) * 1.1, f'{significance_marker_during_to_post}\np={p_val_during_to_post:.3f}', ha='center', color='orange')

        plt.tight_layout()
        pdf_boxplot_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Boxplot_Changes_CBody.pdf")
        fig.savefig(pdf_boxplot_path, format='pdf')
        plt.close(fig)
                
        ##############################
        
        
        # Initialize lists to store lengths for each period
        lengths_pre = []
        lengths_during = []
        lengths_post = []
        
        # Loop through each trial to calculate lengths
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            length_pre = calculate_length(x_rotated[pre_start:start], y_rotated[pre_start:start])
            length_during = calculate_length(x_rotated[start:end], y_rotated[start:end])
            length_post = calculate_length(x_rotated[end:post_end], y_rotated[end:post_end])
        
            lengths_pre.append(length_pre)
            lengths_during.append(length_during)
            lengths_post.append(length_post)
        
        # Perform t-tests
        t_val_during_to_pre, p_val_during_to_pre = stats.ttest_ind(lengths_during, lengths_pre)
        t_val_during_to_post, p_val_during_to_post = stats.ttest_ind(lengths_during, lengths_post)
        
        # Create a violin plot
        fig, ax = plt.subplots()
        parts = ax.violinplot([lengths_pre, lengths_during, lengths_post])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Pre', 'During', 'Post'])
        ax.set_title('Comparative Lengths of CBody Trajectories')
        ax.set_ylabel('Length cm')
        ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
       
        
        # Color code for each period
        colors = ['green', 'red', 'blue']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.75)
        
        # Add individual data points
        for i, data in enumerate([lengths_pre, lengths_during, lengths_post], 1):
            jittered_x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(jittered_x, data, color='black', s=10, zorder=3)
        
        # Annotate with t-test p-values and asterisks
       
        ax.text(2, max([max(lengths_during), max(lengths_pre)]) * 0.85, f'During to Pre\np={p_val_during_to_pre:.3f}', ha='right')
        ax.text(2, max([max(lengths_during), max(lengths_post)]) * 0.7, f'During to Post\np={p_val_during_to_post:.3f}', ha='left')
        
        plt.tight_layout()
        pdf_violin_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Violin_Lengths_with_TTests_CBody.pdf")
        fig.savefig(pdf_violin_path, format='pdf')
        plt.close(fig)
                
        
        ################## speed CBody ##################
        
        # Function to calculate speed from length and frame count (30 Hz)
        def calculate_speed(length, frame_count):
            time_seconds = frame_count / 30.0  # Convert frame count to seconds
            return length / time_seconds  # Speed in cm/s
        
        # Initialize lists to store speeds for each period
        speeds_pre = []
        speeds_during = []
        speeds_post = []
        
        # Calculate speeds for each period
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            frame_count_pre = start - pre_start
            frame_count_during = end - start
            frame_count_post = post_end - end
        
            speed_pre = calculate_speed(calculate_length(x_rotated[pre_start:start], y_rotated[pre_start:start]), frame_count_pre)
            speed_during = calculate_speed(calculate_length(x_rotated[start:end], y_rotated[start:end]), frame_count_during)
            speed_post = calculate_speed(calculate_length(x_rotated[end:post_end], y_rotated[end:post_end]), frame_count_post)
        
            speeds_pre.append(speed_pre)
            speeds_during.append(speed_during)
            speeds_post.append(speed_post)
        
        # Create a DataFrame for the speeds
        df_speeds = pd.DataFrame({
            'Pre': speeds_pre,
            'During': speeds_during,
            'Post': speeds_post
        })
        
        # Perform paired t-tests for speed
        # Pre vs During
        t_stat_speed_pre_during, p_value_speed_pre_during = stats.ttest_rel(speeds_pre, speeds_during, nan_policy='omit')
        # During vs Post
        t_stat_speed_during_post, p_value_speed_during_post = stats.ttest_rel(speeds_during, speeds_post, nan_policy='omit')
        
        
        # Create a box plot for speeds
        fig, ax_speed = plt.subplots()
        bp_speed = ax_speed.boxplot([df_speeds['Pre'], df_speeds['During'], df_speeds['Post']], patch_artist=True)
        ax_speed.set_xticks([1, 2, 3])
        ax_speed.set_xticklabels(['Pre', 'During', 'Post'])
        ax_speed.set_title('Comparative Speeds in CBody Trajectories')
        ax_speed.set_ylabel('Speed (cm/s)')
        
        # Color code for each period
        colors = ['green', 'red', 'blue']
        for patch, color in zip(bp_speed['boxes'], colors):
            patch.set_facecolor(color)
            
        # Add scatter plot for individual data points
        # Creating x-coordinates for each group
        x_coords_pre = np.random.normal(1, 0.04, size=len(df_speeds['Pre']))
        x_coords_during = np.random.normal(2, 0.04, size=len(df_speeds['During']))
        x_coords_post = np.random.normal(3, 0.04, size=len(df_speeds['Post']))
        
        # Plotting the points
        ax_speed.scatter(x_coords_pre, df_speeds['Pre'], color='black', alpha=0.6, s=10, zorder=3)
        ax_speed.scatter(x_coords_during, df_speeds['During'], color='black', alpha=0.6, s=10, zorder=3)
        ax_speed.scatter(x_coords_post, df_speeds['Post'], color='black', alpha=0.6, s=10, zorder=3)

        # Add p-value texts
        ax_speed.text(1, max(max(df_speeds['Pre']), max(df_speeds['During']), max(df_speeds['Post'])) * 0.95, 
                      f"Pre vs During\np-value: {p_value_speed_pre_during:.4f}", fontsize=10, ha='center')
        ax_speed.text(2, max(max(df_speeds['During']), max(df_speeds['Post'])) * 0.95, 
                      f"During vs Post\np-value: {p_value_speed_during_post:.4f}", fontsize=10, ha='center')

        
        plt.tight_layout()
        pdf_speed_plot_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Speed_Comparison_CBody.pdf")
        fig.savefig(pdf_speed_plot_path, format='pdf')
        plt.close(fig)
                
        ############ ctivity levels CBody ###############
    
    
        # Function to calculate velocity vectors
        def calculate_average_velocity(x, y, frame_window=frame_window_activity):
            velocities = []
            if len(x) >= frame_window:
                for i in range(0, len(x) - frame_window + 1, frame_window):
                    dx = x.iloc[i + frame_window - 1] - x.iloc[i]
                    dy = y.iloc[i + frame_window - 1] - y.iloc[i]
                    time_seconds = frame_window / 30.0
                    avg_velocity_x = dx / time_seconds
                    avg_velocity_y = dy / time_seconds
                    velocities.append((avg_velocity_x, avg_velocity_y))
            return np.array(velocities)
        
        # Function to calculate speed (magnitude) and direction from velocity
        def calculate_speed_and_direction(averaged_velocity):
            speed = np.linalg.norm(averaged_velocity, axis=1)
            direction = np.arctan2(averaged_velocity[:, 1], averaged_velocity[:, 0])
            return speed, direction
        
        # Function to calculate time spent in different activity levels
        def calculate_time_in_activity_levels(speeds, immobile_threshold=2, running_threshold=6):
            frame_duration = 0.5  # Assuming each frame represents 0.5 seconds
            immobile_time = mobile_time = running_time = 0
            for speed in speeds:
                if speed < immobile_threshold:
                    immobile_time += frame_duration
                elif speed < running_threshold:
                    mobile_time += frame_duration
                else:
                    running_time += frame_duration
            return immobile_time, mobile_time, running_time
        
        
        # Resetting the indices
        x_rotated.reset_index(drop=True, inplace=True)
        y_rotated.reset_index(drop=True, inplace=True)
        
        speeds_pre, speeds_during, speeds_post = [], [], []
        activity_levels_pre, activity_levels_during, activity_levels_post = [], [], []
        
        
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(x_rotated), end + (end - start))
            
            for period, period_range in zip(['pre', 'during', 'post'], [(pre_start, start), (start, end), (end, post_end)]):
                start, end = period_range
                
                if start < end:
                    velocity = calculate_average_velocity(x_rotated[start:end], y_rotated[start:end])
                    if len(velocity) > 0:
                        speeds, _ = calculate_speed_and_direction(velocity)
                        immobile_time, mobile_time, running_time = calculate_time_in_activity_levels(speeds)
                       
                        # Append the results to the corresponding list along with the file label
                        activity_data = (immobile_time, mobile_time, running_time, file_counter3)
                        if period == 'pre':
                            speeds_pre.extend(speeds)
                            activity_levels_pre.append([immobile_time, mobile_time, running_time])
                            all_activity_levels_pre.append(activity_data)
                        elif period == 'during':
                            speeds_during.extend(speeds)
                            activity_levels_during.append([immobile_time, mobile_time, running_time])
                            all_activity_levels_during.append(activity_data)
                        else:
                            speeds_post.extend(speeds)
                            activity_levels_post.append([immobile_time, mobile_time, running_time])
                            all_activity_levels_post.append(activity_data)
        
        # Plotting
       
        fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        
        bins = np.linspace(min(speeds_pre + speeds_during + speeds_post), max(speeds_pre + speeds_during + speeds_post), 31)
        ax_hist.hist([speeds_pre, speeds_during, speeds_post], bins=bins, label=['Pre', 'During', 'Post'], alpha=0.7, color=['green', 'red', 'blue'])
        ax_hist.set_title('Average Speed Distribution (0.5s intervals)')
        ax_hist.set_xlabel('Average Speed (cm/s)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_xticks(np.arange(0, max(speeds_pre + speeds_during + speeds_post) + 2, 2))  # Set x-axis ticks at 2 cm/s intervals
        ax_hist.legend()
        
        # Set the flierprops parameter to control the size of outlier points
        flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='none', alpha=0.4)  # Adjust the markersize as needed
        
        # Box Plot for Activity Levels
        ax_box.set_title('Time Spent in Different Activity Levels')
        ax_box.set_ylabel('Time (seconds)')
        activity_labels = ['Immobile <2 cm/s', 'Mobile 2-6 cm/s', 'Running >6 cm/s']
        
                # Data preparation for box plot
        pre_data = [[], [], []]
        during_data = [[], [], []]
        post_data = [[], [], []]
        
        for activity_times in activity_levels_pre:
            for i, time in enumerate(activity_times):
                pre_data[i].append(time)
        
       
        
        for activity_times in activity_levels_during:
            for i, time in enumerate(activity_times):
                during_data[i].append(time)
        
        
        
        for activity_times in activity_levels_post:
            for i, time in enumerate(activity_times):
                post_data[i].append(time)
                
     
        data = pd.DataFrame({
            'SpeedCategory': ['Immobile <2 cm/s'] * len(pre_data[0]) + ['Mobile 2-6 cm/s'] * len(pre_data[1]) + ['Running >6 cm/s'] * len(pre_data[2]),
            'Period': ['Pre'] * len(pre_data[0]) + ['During'] * len(during_data[1]) + ['Post'] * len(post_data[2]),
            'ActivityLevels': pre_data[0] + during_data[1] + post_data[2]
        })
        data['ActivityLevels'].replace(0, 0.01, inplace=True)
        
        # Perform paired t-tests and store results
        t_test_results = []
        for i in range(len(activity_labels)):
            # Pre vs During
            t_stat_pre_during, p_value_pre_during = stats.ttest_rel(pre_data[i], during_data[i], nan_policy='omit')
            # Pre vs Post
            t_stat_during_post, p_value_during_post = stats.ttest_rel(during_data[i], post_data[i], nan_policy='omit')
        
            t_test_results.append({
                'Activity': activity_labels[i],
                'Pre-During p-value': p_value_pre_during,
                'During-Post p-value': p_value_during_post
            })
        
        # Convert results to a DataFrame for easier handling
        t_test_results_df = pd.DataFrame(t_test_results)
       
            
        # Plotting
        positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]  # Position for each boxplot
        box_colors = ['#D3D3D3', '#808080', '#404040']
        for i, label in enumerate(activity_labels):
            ax_box.boxplot([pre_data[i], during_data[i], post_data[i]], positions=positions[i::3], patch_artist=True,
                                     boxprops=dict(facecolor=box_colors[i]), labels=[label] * 3,
                                     flierprops=flierprops)  # Set flierprops to control outlier point properties
        
        # Scatter plot
        for i, data in enumerate([pre_data, during_data, post_data]):
            for j, activity_time in enumerate(data):
                x = np.random.normal(positions[j + i * 3], 0.04, size=len(activity_time))
                ax_box.scatter(x, activity_time, alpha=0.4, s=10, color='black', zorder=3)  # Set zorder to bring points to the front
        
        # Modify the plotting section to include t-test results in a box
        p_value_texts = []
        for i, speed_category in enumerate(activity_labels):
            p_value_text = f"{speed_category}:\n"
            p_value_text += f"Pre-During: {t_test_results_df.loc[i, 'Pre-During p-value']:.4f}\n"
            p_value_text += f"During-Post: {t_test_results_df.loc[i, 'During-Post p-value']:.4f}"
            p_value_texts.append(p_value_text)
        
        # Combine all p-value texts into one string
        combined_p_value_text = "\n".join(p_value_texts)
        
        # Add a text box for p-values
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Adjust position as needed
        ax_box.text(1.3, 0.5, combined_p_value_text, transform=ax_box.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax_box.set_xticks([2, 6, 10])
        ax_box.set_xticklabels(['Pre', 'During', 'Post'])
        
        # Create legend
        legend_patches = [mpatches.Patch(color=box_colors[i], label=activity_labels[i]) for i in range(len(activity_labels))]
        # Place the legend outside of the plot area
        ax_box.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        pdf_speed_plot_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Activity_levels.pdf")
        fig.savefig(pdf_speed_plot_path, format='pdf')
        plt.close(fig)
            
    
        ################## Snout_box plot comparing the lengths of trajectories during the pre, during, and post periods across all trials#######
        # Function to calculate trajectory length
        def calculate_length(x, y):
            return np.sqrt(np.diff(x)**2 + np.diff(y)**2).sum()
        
       # Initialize lists to store lengths for each period
        lengths_changes = []
        
        # Loop through each trial to calculate length changes
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            # Calculate lengths for each period
            length_pre = calculate_length(x_rotated_S[pre_start:start], y_rotated_S[pre_start:start])
            length_during = calculate_length(x_rotated_S[start:end], y_rotated_S[start:end])
            length_post = calculate_length(x_rotated_S[end:post_end], y_rotated_S[end:post_end])
        
            # Calculate changes in length from pre to during and during to post
            change_pre_to_during = length_during - length_pre
            change_during_to_post = length_post - length_during
        
            lengths_changes.append([change_pre_to_during, change_during_to_post])
        
        # Convert to DataFrame for easier manipulation
        df_changes = pd.DataFrame(lengths_changes, columns=['Pre_to_During', 'During_to_Post'])
        
        # Create a box plot
        fig, ax = plt.subplots()
        bp = ax.boxplot([df_changes['Pre_to_During'], df_changes['During_to_Post']], patch_artist=True)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Pre to During', 'During to Post'])
        ax.set_title('Comparative Changes in Lengths of Snout Trajectories')
        ax.set_ylabel('Length Change cm')
        
        # Color code for each change period
        colors = ['purple', 'orange']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add individual data points
        for i, data in enumerate([df_changes['Pre_to_During'], df_changes['During_to_Post']], 1):
            jittered_x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(jittered_x, data, color='black', s=10, zorder=3)
        
        # Perform t-tests and annotate results
        t_val_pre_to_during, p_val_pre_to_during = stats.ttest_1samp(df_changes['Pre_to_During'], 0)
        t_val_during_to_post, p_val_during_to_post = stats.ttest_1samp(df_changes['During_to_Post'], 0)
        
        # Add t-test p-value and asterisks
        # Function to determine the appropriate number of asterisks
        def get_significance_marker(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ''
        
        # Add annotations with p-value and significance marker
        significance_marker_pre_to_during = get_significance_marker(p_val_pre_to_during)
        significance_marker_during_to_post = get_significance_marker(p_val_during_to_post)
        
        ax.text(1, max(df_changes['Pre_to_During']) * 1.1, f'{significance_marker_pre_to_during}\np={p_val_pre_to_during:.3f}', ha='center', color='purple')
        ax.text(2, max(df_changes['During_to_Post']) * 1.1, f'{significance_marker_during_to_post}\np={p_val_during_to_post:.3f}', ha='center', color='orange')
        ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
        plt.tight_layout()
        pdf_boxplot_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Boxplot_Changes_Snout.pdf")
        fig.savefig(pdf_boxplot_path, format='pdf')
        plt.close(fig)
                
        ##############################
        
        
        # Initialize lists to store lengths for each period
        lengths_pre = []
        lengths_during = []
        lengths_post = []
        
        # Loop through each trial to calculate lengths
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
        
            length_pre = calculate_length(x_rotated_S[pre_start:start], y_rotated_S[pre_start:start])
            length_during = calculate_length(x_rotated_S[start:end], y_rotated_S[start:end])
            length_post = calculate_length(x_rotated_S[end:post_end], y_rotated_S[end:post_end])
        
            lengths_pre.append(length_pre)
            lengths_during.append(length_during)
            lengths_post.append(length_post)
        
        # Perform t-tests
        t_val_during_to_pre, p_val_during_to_pre = stats.ttest_ind(lengths_during, lengths_pre)
        t_val_during_to_post, p_val_during_to_post = stats.ttest_ind(lengths_during, lengths_post)
        
        # Create a violin plot
        fig, ax = plt.subplots()
        parts = ax.violinplot([lengths_pre, lengths_during, lengths_post])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Pre', 'During', 'Post'])
        ax.set_title('Comparative Lengths of Snout Trajectories')
        ax.set_ylabel('Length cm')
        
        # Color code for each period
        colors = ['green', 'red', 'blue']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.75)
        
        # Add individual data points
        for i, data in enumerate([lengths_pre, lengths_during, lengths_post], 1):
            jittered_x = np.random.normal(i, 0.04, size=len(data))
            ax.scatter(jittered_x, data, color='black', s=10, zorder=3)
        
        # Annotate with t-test p-values and asterisks
       
        ax.text(2, max([max(lengths_during), max(lengths_pre)]) * 0.85, f'During to Pre\np={p_val_during_to_pre:.3f}', ha='right')
        ax.text(2, max([max(lengths_during), max(lengths_post)]) * 0.7, f'During to Post\np={p_val_during_to_post:.3f}', ha='left')
        ax.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
        
        
        plt.tight_layout()
        pdf_violin_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Violin_Lengths_with_TTests_Snout.pdf")
        fig.savefig(pdf_violin_path, format='pdf')
        plt.close(fig)
        
        
        ############ velocity Snout ###############
    
        
        # Resetting the indices
        x_rotated.reset_index(drop=True, inplace=True)
        y_rotated.reset_index(drop=True, inplace=True)
        
        speeds_pre, speeds_during, speeds_post = [], [], []
        activity_levels_pre, activity_levels_during, activity_levels_post = [], [], []
        
        
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(x_rotated), end + (end - start))
            
            for period, period_range in zip(['pre', 'during', 'post'], [(pre_start, start), (start, end), (end, post_end)]):
                start, end = period_range
               
                if start < end:
                    velocity = calculate_average_velocity(x_rotated_S[start:end], y_rotated_S[start:end])
                    if len(velocity) > 0:
                        speeds, _ = calculate_speed_and_direction(velocity)
                        immobile_time, mobile_time, running_time = calculate_time_in_activity_levels(speeds)
                        
        
                        if period == 'pre':
                            speeds_pre.extend(speeds)
                            activity_levels_pre.append([immobile_time, mobile_time, running_time])
                        elif period == 'during':
                            speeds_during.extend(speeds)
                            activity_levels_during.append([immobile_time, mobile_time, running_time])
                        else:
                            speeds_post.extend(speeds)
                            activity_levels_post.append([immobile_time, mobile_time, running_time])
        
      

        # Plotting
        
        fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
       
        bins = np.linspace(min(speeds_pre + speeds_during + speeds_post), max(speeds_pre + speeds_during + speeds_post), 31)
        ax_hist.hist([speeds_pre, speeds_during, speeds_post], bins=bins, label=['Pre', 'During', 'Post'], alpha=0.7, color=['green', 'red', 'blue'])
        ax_hist.set_title('Average Speed Distribution (0.5s intervals)')
        ax_hist.set_xlabel('Average Speed (cm/s)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_xticks(np.arange(0, max(speeds_pre + speeds_during + speeds_post) + 2, 2))  # Set x-axis ticks at 2 cm/s intervals
        ax_hist.legend()
        
        # Set the flierprops parameter to control the size of outlier points
        flierprops = dict(marker='o', markersize=5, markeredgecolor='black', markerfacecolor='none', alpha=0.4)  # Adjust the markersize as needed
       
        # Box Plot for Activity Levels
        ax_box.set_title('Time Spent in Different Activity Levels')
        ax_box.set_ylabel('Time (seconds)')
        activity_labels = ['Immobile <2 cm/s', 'Mobile 2-6 cm/s', 'Running >6 cm/s']
        
                # Data preparation for box plot
        pre_data = [[], [], []]
        during_data = [[], [], []]
        post_data = [[], [], []]
       
        
        for activity_times in activity_levels_pre:
            for i, time in enumerate(activity_times):
                pre_data[i].append(time)
        
       
        
        for activity_times in activity_levels_during:
            for i, time in enumerate(activity_times):
                during_data[i].append(time)
      
        
        for activity_times in activity_levels_post:
            for i, time in enumerate(activity_times):
                post_data[i].append(time)
                
     
        data = pd.DataFrame({
            'SpeedCategory': ['Immobile <2 cm/s'] * len(pre_data[0]) + ['Mobile 2-6 cm/s'] * len(pre_data[1]) + ['Running >6 cm/s'] * len(pre_data[2]),
            'Period': ['Pre'] * len(pre_data[0]) + ['During'] * len(during_data[1]) + ['Post'] * len(post_data[2]),
            'ActivityLevels': pre_data[0] + during_data[1] + post_data[2]
        })
        data['ActivityLevels'].replace(0, 0.01, inplace=True)
       
        # Perform paired t-tests and store results
        t_test_results = []
        for i in range(len(activity_labels)):
            # Pre vs During
            t_stat_pre_during, p_value_pre_during = stats.ttest_rel(pre_data[i], during_data[i], nan_policy='omit')
            # Pre vs Post
            t_stat_during_post, p_value_during_post = stats.ttest_rel(during_data[i], post_data[i], nan_policy='omit')
        
            t_test_results.append({
                'Activity': activity_labels[i],
                'Pre-During p-value': p_value_pre_during,
                'During-Post p-value': p_value_during_post
            })
        
        # Convert results to a DataFrame for easier handling
        t_test_results_df = pd.DataFrame(t_test_results)
        
            
        # Plotting
        positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]  # Position for each boxplot
        box_colors = ['#D3D3D3', '#808080', '#404040']
        for i, label in enumerate(activity_labels):
            ax_box.boxplot([pre_data[i], during_data[i], post_data[i]], positions=positions[i::3], patch_artist=True,
                                     boxprops=dict(facecolor=box_colors[i]), labels=[label] * 3,
                                     flierprops=flierprops)  # Set flierprops to control outlier point properties
        
        # Scatter plot
        for i, data in enumerate([pre_data, during_data, post_data]):
            for j, activity_time in enumerate(data):
                x = np.random.normal(positions[j + i * 3], 0.04, size=len(activity_time))
                ax_box.scatter(x, activity_time, alpha=0.4, s=10, color='black', zorder=3)  # Set zorder to bring points to the front
        
        # Modify the plotting section to include t-test results in a box
        p_value_texts = []
        for i, speed_category in enumerate(activity_labels):
            p_value_text = f"{speed_category}:\n"
            p_value_text += f"Pre-During: {t_test_results_df.loc[i, 'Pre-During p-value']:.4f}\n"
            p_value_text += f"During-Post: {t_test_results_df.loc[i, 'During-Post p-value']:.4f}"
            p_value_texts.append(p_value_text)
        
        # Combine all p-value texts into one string
        combined_p_value_text = "\n".join(p_value_texts)
        
        # Add a text box for p-values
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Adjust position as needed
        ax_box.text(1.3, 0.5, combined_p_value_text, transform=ax_box.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
        
        ax_box.set_xticks([2, 6, 10])
        ax_box.set_xticklabels(['Pre', 'During', 'Post'])
        
        # Create legend
        legend_patches = [mpatches.Patch(color=box_colors[i], label=activity_labels[i]) for i in range(len(activity_labels))]
        # Place the legend outside of the plot area
        ax_box.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        pdf_speed_plot_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Activity_levels_Snout.pdf")
        fig.savefig(pdf_speed_plot_path, format='pdf')
        plt.close(fig)
            
        
        
        ############## angular displacement #######
        
        def calculate_angle(vector):
            # Ensure the vector is a numpy array
            vector = np.array(vector)
            return np.arctan2(vector[1], vector[0])  # Calculate angle and return a single value
                
        def calculate_rotational_speed(angles, time_interval=1/30):
            # Calculate angular displacement between consecutive angles
            angular_displacement = np.diff(angles)
            # Rotational speed = Angular displacement / Time interval
            return angular_displacement / time_interval
        # Initialize lists to store raw data
        vectors_pre = []
        vectors_during = []
        vectors_post = []

        # Process data for each period
        for start, end in false_periods:
            pre_start = max(0, start - (end - start))
            post_end = min(len(aligned_df), end + (end - start))
           
            for i in range(pre_start, post_end):
                # Data extraction and processing
                # [Extracting baseTail, CBody, Neck, Snout]
                
               
                Neck = np.array([(pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck'][i], errors='coerce')),
                                  (pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_neck.1'][i], errors='coerce'))]) 
                
                Snout = np.array([(pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout'][i], errors='coerce')),
                                   (pd.to_numeric(aligned_df['DLC_resnet50_small_open_field_Nov23Nov21shuffle1_910000_Snout.1'][i], errors='coerce'))])
                
              
                Neck_Snout = Snout - Neck
               
        
                # Store raw data
                if i < start:
                    vectors_pre.append(( Neck_Snout))
                elif start <= i < end:
                    vectors_during.append(( Neck_Snout))
                else:
                    vectors_post.append((Neck_Snout))
        
             
        # vectors_pre, vectors_during, vectors_post are lists of Neck-Snout vectors for each period
        angles_pre = np.array([calculate_angle(vector) for vector in vectors_pre])
        angles_during = np.array([calculate_angle(vector) for vector in vectors_during])
        angles_post = np.array([calculate_angle(vector) for vector in vectors_post])
         
        # Function to calculate average rotational speed for each trial
        
        def calculate_average_rotational_speed(angles, frame_window=frame_window_angular, frame_rate=30):
            angles = np.array(angles)
           
        
            angular_displacement = []
            try:
                for i in range(len(angles) - frame_window + 1):
                   
                    start_angle = angles[i]
                    end_angle = angles[i + frame_window - 1]
                    displacement = end_angle - start_angle
        
                    # Correct for angle wrapping
                    if displacement > np.pi:
                        displacement -= 2 * np.pi
                    elif displacement < -np.pi:
                        displacement += 2 * np.pi
        
                    angular_displacement.append(displacement)
        
            except Exception as e:
                print("Error occurred Angular displacement:", e)
        
            rotational_speed = np.array(angular_displacement) / (frame_window / frame_rate)
            
            return rotational_speed
       
        # Calculate average rotational speed over 15 frames
        avg_rotational_speed_pre = calculate_average_rotational_speed(angles_pre)
        avg_rotational_speed_during = calculate_average_rotational_speed(angles_during)
        avg_rotational_speed_post = calculate_average_rotational_speed(angles_post)
       
        rotational_speed_pre = calculate_rotational_speed(angles_pre)
        rotational_speed_during = calculate_rotational_speed(angles_during)
        rotational_speed_post = calculate_rotational_speed(angles_post)
        
        # Perform statistical tests
        t_stat_pre_during, p_val_pre_during = stats.ttest_rel(rotational_speed_pre, rotational_speed_during)
        t_stat_during_post, p_val_during_post = stats.ttest_rel(rotational_speed_during, rotational_speed_post)
        
        # Perform statistical tests
        t_stat_av_pre_during, p_val_av_pre_during = stats.ttest_rel(avg_rotational_speed_pre, avg_rotational_speed_during)
        t_stat_av_during_post, p_val_av_during_post = stats.ttest_rel(avg_rotational_speed_during, avg_rotational_speed_post)
       
        # Convert the data to 1D arrays if they are not already
        rotational_speed_pre_flat = np.squeeze(rotational_speed_pre)
        rotational_speed_during_flat = np.squeeze(rotational_speed_during)
        rotational_speed_post_flat = np.squeeze(rotational_speed_post)
        
        avg_rotational_speed_pre_flat = np.squeeze(avg_rotational_speed_pre)
        avg_rotational_speed_during_flat = np.squeeze(avg_rotational_speed_during)
        avg_rotational_speed_post_flat = np.squeeze(avg_rotational_speed_post)
        
       # Plotting
        fig, (ax_frt, ax_avg) = plt.subplots(1, 2, figsize=(18, 6))
        # Draw dashed lines for x and y axes at the origin
        ax_avg.axhline(y=0, color='black', linestyle='dashed', linewidth=1, zorder=1)
        ax_frt.axhline(y=0, color='black', linestyle='dashed', linewidth=1, zorder=1)
        # Create the violin plots
        vp_frt = ax_frt.violinplot([rotational_speed_pre_flat, rotational_speed_during_flat, rotational_speed_post_flat], showmeans=False, showmedians=True)
        vp_avg = ax_avg.violinplot([avg_rotational_speed_pre_flat, avg_rotational_speed_during_flat, avg_rotational_speed_post_flat], showmeans=False, showmedians=True)
        
        # Set zorder for violin plot bodies to be lower, ensuring scatter plots and median lines remain on top
        for pc in vp_avg['bodies'] + vp_frt['bodies']:
            pc.set_zorder(2)
        
        # Color coding
        colors = ['green', 'red', 'blue']
        alpha_val = 1.0  # Full opacity
        for pc, color in zip(vp_frt['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha_val)
        for pc, color in zip(vp_avg['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(alpha_val)
        
        # Add scatter plot for individual data points
        scatter_zorder = 3  # Scatter plots should be above violin plot bodies
        x_coords_pre_avg = np.random.normal(1, 0.04, size=len(avg_rotational_speed_pre_flat))
        x_coords_during_avg = np.random.normal(2, 0.04, size=len(avg_rotational_speed_during_flat))
        x_coords_post_avg = np.random.normal(3, 0.04, size=len(avg_rotational_speed_post_flat))


        ax_frt.scatter(np.random.normal(1, 0.04, size=len(rotational_speed_pre_flat)), rotational_speed_pre_flat, alpha=0.2, s=10, color='gray', zorder=scatter_zorder)
        ax_frt.scatter(np.random.normal(2, 0.04, size=len(rotational_speed_during_flat)), rotational_speed_during_flat, alpha=0.2, s=10, color='gray', zorder=scatter_zorder)
        ax_frt.scatter(np.random.normal(3, 0.04, size=len(rotational_speed_post_flat)), rotational_speed_post_flat, alpha=0.2, s=10, color='gray', zorder=scatter_zorder)
        ax_avg.scatter(x_coords_pre_avg, avg_rotational_speed_pre_flat, alpha=0.2, s=10, color='gray', label='Pre', zorder=scatter_zorder)
        ax_avg.scatter(x_coords_during_avg, avg_rotational_speed_during_flat, alpha=0.2, s=10, color='gray', label='During', zorder=scatter_zorder)
        ax_avg.scatter(x_coords_post_avg, avg_rotational_speed_post_flat, alpha=0.2, s=10, color='gray', label='Post', zorder=scatter_zorder)
        
        # Calculate and draw median lines for scatter plots
        mean_pre = np.mean(avg_rotational_speed_pre_flat)
        mean_during = np.mean(avg_rotational_speed_during_flat)
        mean_post = np.mean(avg_rotational_speed_post_flat)
        
        ax_avg.axhline(y=mean_pre, color='green', linestyle='dashed', linewidth=0.8, zorder=4)
        ax_avg.axhline(y=mean_during, color='red', linestyle='dashed', linewidth=0.8, zorder=4)
        ax_avg.axhline(y=mean_post, color='blue', linestyle='dashed', linewidth=0.8, zorder=4)

        # For the 'pre' period
        for speed in avg_rotational_speed_pre_flat:
            all_rotational_speeds_pre.append((speed, file_counter2))
        
        # For the 'during' period
        for speed in avg_rotational_speed_during_flat:
            all_rotational_speeds_during.append((speed, file_counter2))
        
        # For the 'post' period
        for speed in avg_rotational_speed_post_flat:
            all_rotational_speeds_post.append((speed, file_counter2))
        
        # Bias calculation and plot annotation
        def calculate_bias(rotational_speed):
            return np.sum(rotational_speed) / len(rotational_speed)
        
        bias_pre, bias_during, bias_post = map(calculate_bias, [avg_rotational_speed_pre_flat, avg_rotational_speed_during_flat, avg_rotational_speed_post_flat])
        median_pre, median_during, median_post = map(np.median, [avg_rotational_speed_pre_flat, avg_rotational_speed_during_flat, avg_rotational_speed_post_flat])
        bias_text_avg = f'Bias_Pre: {bias_pre:.2f}\nBias_During: {bias_during:.2f}\nBias_Post: {bias_post:.2f}\nM_Pre: {median_pre:.2f}\nM_During: {median_during:.2f}\nM_Post: {median_post:.2f}\nPre-During: p={p_val_av_pre_during:.4f}\nDuring-Post: p={p_val_av_during_post:.4f}'
        ax_avg.text(1.2, 0.75, bias_text_avg, transform=ax_avg.transAxes, fontsize=10, horizontalalignment='center')
        
        # Plot settings
        ax_frt.set_title('Rotational Speed of Neck-Snout Vector')
        ax_frt.set_ylabel('Rotational Speed (rad/s)')
        ax_avg.set_title('Rotational Avg_Speed of Neck-Snout Vector')
        ax_avg.set_ylabel('Rotational Avg_Speed (rad/s)')
        
        # Save the figure
        plt.tight_layout()
        pdf_RT_speed_plot_path = os.path.join(Pdf_directory, f"{dlc_base_name}_Rotational_speed_Snout.pdf")
        fig.savefig(pdf_RT_speed_plot_path, format='pdf')
        plt.close(fig)
        
        
        ##################plot_vector_directions########################

         # Call the plot_vector_directions function
        if not aligned_df.empty:
            false_periods = find_periods(aligned_df['Item2'])
            if false_periods:
                plot_vector_directions(aligned_df, false_periods, Pdf_directory, dlc_base_name)
                # Increment the file counter for the next file
                
            else:
                print(f"No false periods found for files: {bf_file} and {dlc_file}")
                
        custom_print(f"PDF file has been created: {pdf_file_path}", directory=Pdf_directory)
    except Exception as e:
        
        custom_print(f"Error processing files: {bf_file} and {dlc_file}. Error: {e}", directory=Pdf_directory)




#############Find matching files##############


def list_csv_files(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and re.search(pattern, file):
                matched_files.append(os.path.join(root, file))
    return matched_files

def find_matching_files(dir1, dir2, pattern):
    files1 = list_csv_files(dir1, pattern)
    files2 = list_csv_files(dir2, pattern)

    matching_pairs = []
    for file1 in files1:
        match1 = re.search(pattern, os.path.basename(file1))
        if not match1:
            continue

        for file2 in files2:
            match2 = re.search(pattern, os.path.basename(file2))
            if match2 and match1.group(1) == match2.group(1):
                matching_pairs.append((file1, file2))
                break

    return matching_pairs

# Doric folder
dir1 = "/Users/iakovos/Library/CloudStorage/Dropbox/opto_240122_bonsai_doric"
# DLC folder
dir2 = "/Users/iakovos/Library/CloudStorage/Dropbox/opto_240129_DLC/GFAP/New Folder With Items/New Folder With Items 2/New Folder With Items 3/"

pattern = r'(.*\d{4}-\d{2}-\d{2}T\d{2}_\d{2}).*\.csv'
frame_window_angular = 5 # for avg
frame_window_activity = 5 # for avg
pixel_size_ = 0.0323
matching_files = find_matching_files(dir1, dir2, pattern)
n_files = len(matching_files)
file_colors = plt.cm.get_cmap('hsv', n_files)  # Choose a colormap
file_to_color = {os.path.basename(file): file_colors(i) for i, (file, _) in enumerate(matching_files)}


# Process and plot data for each pair of matched files
for bf_file, dlc_file in matching_files:
    dlc_directory2 = os.path.dirname(dlc_file)
    dlc_base_name = os.path.splitext(os.path.basename(dlc_file))[0]
    pdf_file_path = os.path.join(dlc_directory2, f"{dlc_base_name}_plots.pdf")
    
    # Initialize PDF document
    pdf_pages = PdfPages(pdf_file_path)
    
    # Process and plot data
    process_and_plot(bf_file, dlc_file, pdf_pages, file_to_color)  # Pass 'pdf_pages' as the third argument
   
    file_counter += 1
    file_counter2 += 1
    file_counter3 += 1
    print('file counta', file_counter)
    # Close the PDF and save the file
    pdf_pages.close()
    
    custom_print(f"PDF file has been created: {pdf_file_path}", directory=dir2)

# Find unique labels and assign a color to each
unique_labels = sorted(set(all_file_labels))
n_labels = len(unique_labels)
label_to_color = plt.cm.get_cmap('viridis', n_labels)  # You can choose any suitable colormap

# Create a color list corresponding to all_file_labels
Mice_colors = [label_to_color(unique_labels.index(label)) for label in all_file_labels]
##########

print('collective started')

################## collective ############
create_collective_activity_level_plot(all_activity_levels_pre, all_activity_levels_during, all_activity_levels_post, dir2, dlc_base_name)
create_collective_boxplot(all_changes_pre_to_during, all_changes_during_to_post, Mice_colors, dir2, dlc_base_name)
create_collective_angular_plot(all_rotational_speeds_pre, all_rotational_speeds_during, all_rotational_speeds_post, dir2, dlc_base_name)


# Call the function with the desired directory and script name
save_current_script(dir2, "auto_DLC_Fip_doric_Iak_SWTmaze_240119.py")
