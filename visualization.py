import numpy as np
import matplotlib as plt
import os 
import pdb
import json
import matplotlib.pyplot as plt


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
    save_path = os.path.join(plot_basepath,metric_name,model_name ,f"{layer_name}_combined_plot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the directory exists
    plt.savefig(save_path)
    plt.close() 

    print(f"Saved combined plot: {save_path}")


def all_model_visualizations(directory, plot_basepath, metric_name):
    '''
    given a parent model directory, obtain visualizations for layer sub-directories
    args: directory (path)
    '''
    # List all files in the directory
    model_folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    for mf in model_folders:
        mf_path = os.path.join(directory, mf)
        all_layer_visualizations(mf_path,plot_basepath, mf, metric_name)


def all_layer_visualizations(directory, plot_basepath, model_name, metric_name):
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
        visualize_all_conditions_line(layer_data_dictionary, plot_basepath, model_name, layer_name, metric_name)



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
    all_model_visualizations(directory, plot_basepath, metric_name)
