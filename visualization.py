import numpy as np
import matplotlib as plt
import os 
import pdb
import json
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_single_condition_line(data, train_label, test_label, plot_basepath,  model_name, layer_name, metric_name):
    marker = '*'
    color = 'green'
    if train_label == "fl":
        marker = 'o'
    if test_label == "fl":
        color = 'purple'

    plt.scatter(data, np.zeros_like(data), color=color, marker=marker)
    plt.xlabel(metric) #label the plot by metric
    plt.xlim(-1, 1)
    plt.ylim(-0.1, 0.1)  
    plt.yticks([])  # Remove y-axis ticks

    save_path = os.path.join(plot_basepath, metric_name, model_name ,f"{layer_name}_{train_label}_{test_label}_plot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists

    # Save the plot
    plt.savefig(save_path)
    plt.close() 

    print(f"Saved plot: {save_path}")

def visualize_all_conditions_line(data_dict, plot_basepath, model_name,layer_name, metric_name):
    # Define y-offsets for each data type to stack line plots
    y_offsets = {
        'll_ll': 0.3,
        'll_fl': 0.1,
        'fl_ll': -0.1,
        'fl_fl': -0.3
    }

    # Define marker and color mappings for each key
    #ll (train) = 'o', fl(train) = '*', ll (test) = green, fl(test) = purple
    markers = {'ll_ll': 'o', 'll_fl': 'o', 'fl_ll': '*', 'fl_fl': '*'}
    colors = {'ll_ll': 'green', 'll_fl': 'purple', 'fl_ll': 'green', 'fl_fl': 'purple'}

    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size

    for key, y_offset in y_offsets.items():
        if key in data_dict:  # Check if the key exists in the data_dict
            plt.scatter(data_dict[key], 
                        np.full_like(data_dict[key], y_offset),  # Apply vertical offset
                        color=colors[key], marker=markers[key], label=key)

    #Labels, legend, axis limits
    plt.xlabel(metric_name)
    plt.xlim(-1, 1)
    plt.ylim(-0.5, 0.5)  # Adjust y-axis range to fit all offsets
    plt.legend(title="Data Type: train_test")
    plt.title(f"Layer: {layer_name}")
    plt.yticks([])  # Remove y-axis ticks

    # Save the plot
    save_path = os.path.join(plot_basepath,metric_name,model_name, 'line_plots', f"{layer_name}_conditions.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path)
    plt.close() 

    print(f"Saved combined plot: {save_path}")

def visualize_all_conditions_violin(data_dict, plot_basepath, model_name, layer_name, metric_name):
    # Define color mappings for each condition
    #colors from accessible color-palette: True
    colors = {'ll_ll': '#DC267F', 'll_fl': '#FFB000', 'fl_ll': '#785EF0', 'fl_fl': '#FE6100'}
    
    # Prepare data for Seaborn violin plot
    all_data = []
    all_conditions = []
    for key, values in data_dict.items():
        all_data.extend(values)  # Flatten the data into a single list
        all_conditions.extend([key] * len(values))  # Track condition labels

    
    # # Create a vertical violin plot
    # sns.violinplot(x=all_conditions, y=all_data, palette=colors, cut=0)

    import pandas as pd
    df = pd.DataFrame({'Condition': all_conditions, 'Values': all_data})

    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Create a violin plot
    ax = sns.violinplot(
        x='Condition',
        y='Values',
        data=df,
        palette=colors,
        cut=0,        # Do not extend beyond data range
        linewidth=0 # Keep the median and IQR lines
    )

    # Add horizontal lines for median and ends of violin plot
    for idx, key in enumerate(data_dict.keys()):
        condition_data = np.array(data_dict[key])
        
        # Compute statistics
        median = np.median(condition_data)
        min_line = min(condition_data)
        max_line = max(condition_data)
        
        # X-coordinate for this condition
        x_coord = idx
        
        # Plot median and whisker lines
        plt.hlines(y=median, xmin=x_coord - 0.2, xmax=x_coord + 0.2, color='black', linewidth=2)  # Median
        plt.hlines(y=min_line, xmin=x_coord - 0.15, xmax=x_coord + 0.15, color='black', linewidth=1)   # Lower end
        plt.hlines(y=max_line, xmin=x_coord - 0.15, xmax=x_coord + 0.15, color='black', linewidth=1)   # Upper end


    # Overlay outliers as stars
    for key in data_dict.keys():
        condition_data = data_dict[key]
        q1 = np.percentile(condition_data, 25)
        q3 = np.percentile(condition_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identify outliers
        outliers = [val for val in condition_data if val < lower_bound or val > upper_bound]
        
        if outliers:
            # Get x-coordinate of the condition
            x_coord = list(data_dict.keys()).index(key)
            # Plot outliers as stars
            plt.scatter([x_coord] * len(outliers), outliers, color='black', marker='*', zorder=3, label='Outliers' if key == list(data_dict.keys())[0] else "")

    for violin in ax.collections:
        violin.set_alpha(0.5)
    
    # Add labels, title, and styling
    plt.xlabel("Conditions")
    plt.ylabel(metric_name)
    plt.title(f"Layer: {layer_name} - {metric_name}")
    plt.ylim(-0.75, 0.75)
    
    sns.despine() 

    # Save the plot
    save_path = os.path.join(plot_basepath, metric_name, model_name, "violin_plots", f"{layer_name}_conditions.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved violin plot: {save_path}")


def all_model_visualizations(directory, plot_basepath, metric_name, visualization):
    '''
    given a parent model directory, obtain visualizations for layer sub-directories
    args: directory (path)
    '''
    # List all files in the directory
    model_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    for mf in model_folders:
        mf_path = os.path.join(directory, mf)
        all_layer_visualizations(mf_path,plot_basepath, mf, metric_name, visualization)


def all_layer_visualizations(directory, plot_basepath, model_name, metric_name, visualization):
    '''
    given a parent layer directory, obtain visualizations for layer sub-directories
    args: directory (path)
    '''
    # List all files in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    for lf in folders:
        layer_name = lf
        lf = os.path.join(directory, lf)
        layer_data_dictionary = get_data_dictionary(lf)
        visualization(layer_data_dictionary, plot_basepath, model_name, layer_name, metric_name)



def get_data_dictionary(layer_file):
    ''' 
    given a particular directory, produce processed data for all 4 test cases 
    args:
        layer_file (path): path to directory where layer representations are housed
    returns:
        data_dictionary (dict): data for all 4 test cases
    '''
    layer_name = os.path.basename(layer_file)
    layer_data = [f for f in os.listdir(layer_file) if os.path.isfile(os.path.join(layer_file, f))]

    #keyname: train_test
    data_dict = {
                    'll_ll': [], 
                    'll_fl': [], 
                    'fl_ll': [], 
                    'fl_fl': []
                }
    
    
    for filename in layer_data:
        data_path = os.path.join(layer_file, filename)
        try:
            preprocessed_file_data = preprocess_input(data_path) #should be a dictionary 
            left_out_index = preprocessed_file_data["left_out_index"]
            opposite_set_keys = set(preprocessed_file_data['test_indices'])
            opposite_set_keys.remove(left_out_index)  #remove same set key from opposite set
            
            if preprocessed_file_data['train_type'] == 'literal':
                #ll_fl should be 90 values, ll_ll should be 1 value
                data_dict['ll_fl'].append([preprocessed_file_data['similarities'].get(str(key)) for key in opposite_set_keys])
                data_dict['ll_ll'].append(preprocessed_file_data['similarities'][str(left_out_index)])
            elif preprocessed_file_data['train_type'] == 'idiom':
                data_dict['fl_ll'].append([preprocessed_file_data['similarities'].get(str(key)) for key in opposite_set_keys])
                data_dict['fl_fl'].append(preprocessed_file_data['similarities'][str(left_out_index)])
            
        except:
            #if empty file, exclude it
            continue

    #need to mean over each column in fl_ll and ll_fl (opposite-case) files
    data_dict['ll_fl'] = np.mean(data_dict['ll_fl'], axis=0) #mean over columns to get average for each stimulus
    data_dict['fl_ll'] = np.mean(data_dict['fl_ll'], axis=0)

    return data_dict
    


def preprocess_input(data_path):
    '''
    obtains file from path and returns extracted json file
    args: data_path (path) 
    returns: data (dictionary)
    '''

    data = {}
    with open(data_path, 'r') as file:
        data = json.load(file)

    return data
        

if __name__ == "__main__":
    directory = "/Users/kaitlinzareno/USC/CS-626-Project/neural_cosine_v2"
    plot_basepath = "/Users/kaitlinzareno/USC/CS-626-Project/visualizations"
    metric_name = "Neural_Similarity"
    visualization  = visualize_all_conditions_violin #pass in visualization method
    all_model_visualizations(directory, plot_basepath, metric_name, visualization)
