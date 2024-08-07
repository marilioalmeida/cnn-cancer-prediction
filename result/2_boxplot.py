import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_folder_contents(base_path):
    # Navigate to the 'results' folder
    results_path = os.path.join(base_path, 'results')
    
    # Get all subfolders within 'results'
    subfolders = [f for f in glob.glob(results_path + "/*") if os.path.isdir(f)]
    
    # Check if there are subfolders
    if not subfolders:
        print("No subfolders found in 'results'.")
        return None, []
    
    # Find the newest subfolder
    latest_subfolder = max(subfolders, key=os.path.getmtime)
    
    # List the contents of the newest subfolder
    contents = os.listdir(latest_subfolder)
    
    # Find all Excel files in the newest subfolder, ignoring 'summary_data.xlsx'
    excel_files = [f for f in contents if f.endswith('.xlsx') and f != 'summary_data.xlsx']
    if not excel_files:
        print("No relevant Excel files found in the newest folder.")
        return latest_subfolder, []
    
    excel_file_paths = [os.path.join(latest_subfolder, f) for f in excel_files]
    print(f"Excel files found: {excel_file_paths}")
    return latest_subfolder, excel_file_paths

def read_and_plot_metrics(file_path, save_path, metrics):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    for metric in metrics:
        # Check if the metric exists in the DataFrame
        if metric not in df.columns:
            print(f"The column '{metric}' was not found in the file.")
            continue
        
        # Generate the boxplot for the metric
        plt.figure(figsize=(10, 6))
        df.boxplot(column=metric)
        plt.title(f'Boxplot of the column {metric}')
        plt.ylabel(metric)
        
        # Save the image to the specified path
        file_name = os.path.basename(os.path.basename(file_path).split('_')[0]).split('.')[0]
 
        subfolder_path = os.path.join(save_path, file_name)
        os.makedirs(subfolder_path, exist_ok=True)

        metric_save_path = os.path.join(subfolder_path, f'{file_name}_{metric.lower().replace(" ", "_")}.png')
        plt.savefig(metric_save_path)
        plt.close()
        print(f"Image of the metric '{metric}' saved at: {metric_save_path}")

# Call the function with the base path where the 'results' folder is located
base_path = '.'  # You can adjust the base path as necessary
latest_folder_path, excel_file_paths = get_latest_folder_contents(base_path)

if excel_file_paths:
    # Create the boxplot folder
    boxplot_path = os.path.join(latest_folder_path, 'boxplot')
    os.makedirs(boxplot_path, exist_ok=True)
    
    # Complete path to save the images
    image_save_path = boxplot_path
    
    # Metrics to generate the plots
    metrics = ['Accuracy', 'Precision Micro', 'Recall Micro', 'F1 Score Micro']
    
    # Read the data and save the plots for each Excel file
    for file_path in excel_file_paths:
        read_and_plot_metrics(file_path, image_save_path, metrics)
