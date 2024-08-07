import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_folder_contents(base_path):
    results_path = os.path.join(base_path, 'results')
    subfolders = [f for f in glob.glob(results_path + "/*") if os.path.isdir(f)]
    
    if not subfolders:
        print("No subfolders found in 'results'.")
        return None, []
    
    latest_subfolder = max(subfolders, key=os.path.getmtime)
    contents = os.listdir(latest_subfolder)
    excel_files = [f for f in contents if f.endswith('.xlsx') and f != 'summary_data.xlsx']
    if not excel_files:
        print("No relevant Excel files found in the newest folder.")
        return latest_subfolder, []
    
    excel_file_paths = [os.path.join(latest_subfolder, f) for f in excel_files]
    print(f"Excel files found: {excel_file_paths}")
    return latest_subfolder, excel_file_paths

def read_and_plot_metrics(file_path, save_path, metrics):
    df = pd.read_excel(file_path)
    
    # Filter dataframe to include only the columns that exist in the metrics list
    existing_metrics = [metric for metric in metrics if metric in df.columns]
    if not existing_metrics:
        print(f"No relevant metrics found in the file: {file_path}")
        return
    
    # Save the figure to the specified path
    file_name = os.path.basename(file_path).split('_')[0].split('.')[0]
    subfolder_path = os.path.join(save_path, file_name)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Create a boxplot for each metric in the same figure
    plt.figure(figsize=(15, 10))
    box = df.boxplot(column=existing_metrics, patch_artist=True, return_type='dict')
    
    # Add colors
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#B266FF']  # Add more colors if needed
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Boxplot of ' + file_name)
    plt.ylabel('Values')

    metric_save_path = os.path.join(subfolder_path, f'{file_name}_all_metrics_boxplot.png')
    plt.savefig(metric_save_path)
    plt.close()
    print(f"Image of all metrics saved at: {metric_save_path}")

base_path = '.'
latest_folder_path, excel_file_paths = get_latest_folder_contents(base_path)

if excel_file_paths:
    boxplot_path = os.path.join(latest_folder_path, 'boxplot')
    os.makedirs(boxplot_path, exist_ok=True)
    
    image_save_path = boxplot_path
    metrics = ['Accuracy', 'Precision Micro', 'Recall Micro', 'F1 Score Micro']
    
    for file_path in excel_file_paths:
        read_and_plot_metrics(file_path, image_save_path, metrics)
