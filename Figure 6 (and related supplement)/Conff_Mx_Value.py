import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom_test
from matplotlib import rcParams

# Adjust plot settings
rcParams['figure.figsize'] = [2, 1]
rcParams['font.size'] = 6
rcParams['pdf.fonttype'] = 42
rcParams['font.family'] = 'Arial'

# Define the path where confusion matrix files are stored
conf_matrix_path = "/Users/iakovos/Library/CloudStorage/Dropbox/240527-task classification 4s all animals"

# Define the file pattern and analysis types
file_pattern = "conf_matrix_normalized_Animal"
analysis_types = ['CNN', 'RFC', 'LSTM', 'SVC']
groups = ['PNOC', 'NTS']
signals = ['Red-L', 'Red-R', 'Green-L', 'Green-R']

# Initialize a dictionary to store aggregated confusion matrices
confusion_data = {}

# Function to extract analysis type from file name
def extract_analysis_type(file_name):
    for analysis_type in analysis_types:
        if analysis_type in file_name:
            return analysis_type
    return None

# Function to extract group from file name
def extract_group(file_name):
    for group in groups:
        if group in file_name:
            return group
    return None

# Read the first file to extract the labels
labels = []
for file_name in os.listdir(conf_matrix_path):
    if file_name.startswith(file_pattern):
        file_path = os.path.join(conf_matrix_path, file_name)
        conf_matrix = pd.read_csv(file_path, index_col=0)
        labels = conf_matrix.columns.tolist()
        break

# Read each file and extract metadata
for file_name in os.listdir(conf_matrix_path):
    if file_name.startswith(file_pattern):
        parts = file_name.split("_")
        group = extract_group(file_name)
        ID = parts[3].strip()
        analysis_type = extract_analysis_type(file_name)
        signal = parts[-1].replace(".csv", "").strip()

        if group is None:
            print(f"Group not found in file: {file_name}")
            continue
        if analysis_type is None:
            print(f"Analysis type not found in file: {file_name}")
            continue

        # Load the confusion matrix
        file_path = os.path.join(conf_matrix_path, file_name)
        conf_matrix = pd.read_csv(file_path, index_col=0)

        # Normalize confusion matrix
        conf_matrix = conf_matrix.div(conf_matrix.sum(axis=1), axis=0)

        # Initialize dictionary keys if they do not exist
        if group not in confusion_data:
            confusion_data[group] = {}
        if analysis_type not in confusion_data[group]:
            confusion_data[group][analysis_type] = {}
        if signal not in confusion_data[group][analysis_type]:
            confusion_data[group][analysis_type][signal] = []

        # Append the confusion matrix and sample count
        confusion_data[group][analysis_type][signal].append({'matrix': conf_matrix, 'samples': conf_matrix.sum().sum()})

# Combine confusion matrices using weighted averaging
def weighted_average_conf_matrices(confusion_list):
    total_samples = sum(item['samples'] for item in confusion_list)
    combined_matrix = sum((item['matrix'] * item['samples'] for item in confusion_list)) / total_samples
    std_matrix = np.std([item['matrix'].values for item in confusion_list], axis=0)
    return combined_matrix, std_matrix, total_samples

# Dictionary to store combined results
combined_confusion_data = {}

# Aggregate confusion matrices
for group in confusion_data:
    combined_confusion_data[group] = {}
    for analysis_type in confusion_data[group]:
        combined_confusion_data[group][analysis_type] = {}
        for signal in confusion_data[group][analysis_type]:
            combined_matrix, std_matrix, total_samples = weighted_average_conf_matrices(confusion_data[group][analysis_type][signal])
            combined_confusion_data[group][analysis_type][signal] = {'mean': combined_matrix, 'std': std_matrix, 'samples': total_samples}

            # Print combined matrix for debugging
            print(f"Combined matrix for {group}, {analysis_type}, {signal}:")
            print(combined_matrix)

# Combine Red-L and Red-R signals across groups
combined_signals = ['Red-L', 'Red-R']
for signal in combined_signals:
    for analysis_type in analysis_types:
        combined_list = []
        for group in groups:
            if signal in confusion_data[group][analysis_type]:
                combined_list.extend(confusion_data[group][analysis_type][signal])
        
        if combined_list:
            combined_matrix, std_matrix, total_samples = weighted_average_conf_matrices(combined_list)
            if 'Combined' not in combined_confusion_data:
                combined_confusion_data['Combined'] = {}
            if analysis_type not in combined_confusion_data['Combined']:
                combined_confusion_data['Combined'][analysis_type] = {}
            combined_confusion_data['Combined'][analysis_type][signal] = {'mean': combined_matrix, 'std': std_matrix, 'samples': total_samples}

            # Print combined matrix for debugging
            print(f"Combined matrix for Combined, {analysis_type}, {signal}:")
            print(combined_matrix)

# Calculate global min and max values for the color scale
global_min = float('inf')
global_max = float('-inf')

for group in combined_confusion_data:
    for analysis_type in combined_confusion_data[group]:
        for signal in combined_confusion_data[group][analysis_type]:
            matrix = combined_confusion_data[group][analysis_type][signal]['mean']
            global_min = min(global_min, matrix.min().min())
            global_max = max(global_max, matrix.max().max())

# Function to check statistical significance
def check_significance(matrix, total_samples, alpha=0.05):
    n = matrix.shape[0]  # Number of dimensions
    expected_probability = 1 / n
    matrix_np = matrix.to_numpy()  # Convert DataFrame to NumPy array
    significance_matrix = np.zeros(matrix_np.shape, dtype=bool)
    
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            observed = matrix_np[i, j] * total_samples
           
            p_value = binom_test(observed, total_samples, expected_probability)
            if p_value < alpha and matrix_np[i, j] > expected_probability:  # Add condition for higher than expected
                significance_matrix[i, j] = True
                
    return significance_matrix

# Function to calculate performance score
def calculate_performance_score(matrix, significance_matrix):
    diagonal_accuracy = np.diag(matrix).mean()
    off_diagonal_elements = matrix.values[np.triu_indices_from(matrix.values, k=1)]
    off_diagonal_accuracy = off_diagonal_elements.mean()
    num_significant = significance_matrix.sum()
    
    # Penalize high off-diagonal accuracy above chance
    off_diagonal_penalty = off_diagonal_accuracy - 0.25 if off_diagonal_accuracy > 0.25 else 0
    score = diagonal_accuracy - off_diagonal_penalty + num_significant  
    return score, diagonal_accuracy, off_diagonal_accuracy, num_significant

# Plotting function with standard deviation, significance, and green outline for significant cells
def plot_confusion_matrix(matrix, std_matrix, title, labels, significance_matrix, save_path, vmin, vmax):
    
    std_matrix_np = std_matrix  # No conversion needed, already a NumPy array
    plt.figure(figsize=(2, 1.5))
    ax = sns.heatmap(matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=labels, yticklabels=labels, cbar=True, vmin=vmin, vmax=vmax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            # Get the background color for the current cell
            facecolors = ax.collections[0].get_facecolors()
            rgba_color = facecolors[i * len(labels) + j]
            luminance = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
            text_color = 'white' if luminance < 0.5 else 'black'

            # Add the standard deviation text
            ax.text(j + 0.5, i + 0.8, f"±{std_matrix_np[i, j]:.2f}", color=text_color, ha='center', va='center', fontsize=5)

            # Add the asterisk for significance
            if significance_matrix[i, j]:
                #ax.text(j + 0.5, i + 0.3, '*', color=text_color, ha='center', va='center', fontsize=10)
                # Add green outline
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='lightgreen', linewidth=1.5)
                ax.add_patch(rect)

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.show()
    plt.close()

# Function to aggregate significant dimensions across all signals and groups for each method
def aggregate_significant_dimensions(confusion_data, analysis_types, combined_confusion_data, groups, signals):
    aggregated_significant_matrices = {}
    for analysis_type in analysis_types:
        for signal in signals:
            for group in groups:
                key = f"{analysis_type}_{signal}_{group}"
                if analysis_type in combined_confusion_data.get(group, {}):
                    if signal in combined_confusion_data[group][analysis_type]:
                        combined_matrix = combined_confusion_data[group][analysis_type][signal]['mean']
                        total_samples = combined_confusion_data[group][analysis_type][signal]['samples']
                        significance_matrix = check_significance(combined_matrix, total_samples)
                        
                        # Ensure consistency in matrix dimensions
                        if key not in aggregated_significant_matrices:
                            aggregated_significant_matrices[key] = np.zeros(significance_matrix.shape, dtype=int)
                        
                        if aggregated_significant_matrices[key].shape == significance_matrix.shape:
                            # Aggregate significant cells
                            aggregated_significant_matrices[key] += significance_matrix
                        else:
                            print(f"Shape mismatch for key {key}: aggregated {aggregated_significant_matrices[key].shape}, new {significance_matrix.shape}")
    return aggregated_significant_matrices

# Example usage:
performance_scores = []
aggregated_significant_matrices = aggregate_significant_dimensions(confusion_data, analysis_types, combined_confusion_data, groups, signals)

# Plot the combined confusion matrices for each group and analysis type
for analysis_type in analysis_types:
    for group in combined_confusion_data:
        for signal in combined_confusion_data[group][analysis_type]:
            combined_matrix = combined_confusion_data[group][analysis_type][signal]['mean']
            std_matrix = combined_confusion_data[group][analysis_type][signal]['std']
            title = f"{group} - {analysis_type} - {signal}"
            save_path = os.path.join(conf_matrix_path, f"{title}.pdf")
            
            # Check and print significance (optional)
            total_samples = combined_confusion_data[group][analysis_type][signal]['samples']
            significance_matrix = check_significance(combined_matrix, total_samples)
            
            plot_confusion_matrix(combined_matrix, std_matrix, title, labels, significance_matrix, save_path, global_min, global_max)
            print(f"Significance matrix for {title}:")
            print(significance_matrix)
            
            # Calculate performance score
            score, diagonal_accuracy, off_diagonal_accuracy, num_significant = calculate_performance_score(combined_matrix, significance_matrix)
            performance_scores.append((analysis_type, signal, score, diagonal_accuracy, off_diagonal_accuracy, num_significant))
            print(f"Performance score for {title}: Score={score}, Diagonal Accuracy={diagonal_accuracy}, Off-diagonal Accuracy={off_diagonal_accuracy}, Num. Significant={num_significant}")

# Create a summary table
summary_data = []

for analysis_type in analysis_types:
    analysis_scores = [(signal, score, diagonal_accuracy, off_diagonal_accuracy, num_significant) for at, signal, score, diagonal_accuracy, off_diagonal_accuracy, num_significant in performance_scores if at == analysis_type]
    total_score = sum(score for _, score, _, _, _ in analysis_scores)
    avg_diagonal_accuracy = np.mean([diagonal_accuracy for _, _, diagonal_accuracy, _, _ in analysis_scores])
    avg_off_diagonal_accuracy = np.mean([off_diagonal_accuracy for _, _, _, off_diagonal_accuracy, _ in analysis_scores])
    total_num_significant = sum(num_significant for _, _, _, _, num_significant in analysis_scores)
    
    summary_data.append({
        "Analysis Type": analysis_type,
        "Total Score": total_score,
        "Avg. Diagonal Accuracy": avg_diagonal_accuracy,
        "Avg. Off-diagonal Accuracy": avg_off_diagonal_accuracy,
        "Total Num. Significant": total_num_significant
    })

summary_df = pd.DataFrame(summary_data)
print("\nSummary of Method Performance:")
print(summary_df)

# Plot the summary table
fig, ax = plt.subplots(figsize=(10, 6))
summary_df.set_index('Analysis Type').plot(kind='bar', ax=ax)
ax.set_title("Summary of Method Performance")
ax.set_xlabel("Analysis Type")
ax.set_ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
summary_table_path = os.path.join(conf_matrix_path, "method_performance_summary.pdf")
plt.savefig(summary_table_path)
plt.show()
plt.close()

# Plot the aggregated significant dimensions for each signal and group separately
fig, axes = plt.subplots(len(analysis_types), len(groups) * len(signals), figsize=(20, 15))
for i, analysis_type in enumerate(analysis_types):
    for j, signal in enumerate(signals):
        for k, group in enumerate(groups):
            key = f"{analysis_type}_{signal}_{group}"
            if key in aggregated_significant_matrices:
                ax = axes[i, j * len(groups) + k]
                sns.heatmap(aggregated_significant_matrices[key], annot=True, cmap="Greens", xticklabels=labels, yticklabels=labels, cbar=True, ax=ax, vmin=global_min, vmax=global_max)
                ax.set_title(f"{analysis_type} - {signal} - {group}")
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')

plt.tight_layout()
aggregated_path = os.path.join(conf_matrix_path, "aggregated_significant_dimensions.pdf")
plt.savefig(aggregated_path)
plt.show()
plt.close()

# Optionally, save the summary table to a CSV file
summary_df.to_csv(os.path.join(conf_matrix_path, "method_performance_summary.csv"), index=False)
