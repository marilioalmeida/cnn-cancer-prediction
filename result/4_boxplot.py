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

def read_and_plot_metrics(file_paths, save_path, metrics):
    all_data = {metric: pd.DataFrame() for metric in metrics}
    
    for file_path in file_paths:
        file_name = os.path.basename(file_path).split('_')[0].split('.')[0]
        df = pd.read_excel(file_path)
        
        for metric in metrics:
            if metric in df.columns:
                df_metric = df[[metric]].copy()
                df_metric['Source'] = os.path.basename(file_name)
                all_data[metric] = pd.concat([all_data[metric], df_metric], axis=0)
    
    for metric, data in all_data.items():
        if data.empty:
            print(f"No relevant data found for metric: {metric}")
            continue
        
        # Calcular as medianas e ordenar os dados
        medians = data.groupby('Source')[metric].median().sort_values(ascending=False)
        sorted_sources = medians.index
        data['Source'] = pd.Categorical(data['Source'], categories=sorted_sources, ordered=True)
        data = data.sort_values('Source')
        
        fig, ax = plt.subplots(figsize=(15, 10))
        box = data.boxplot(column=[metric], by='Source', patch_artist=True, return_type='dict', ax=ax)
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#B266FF', '#FF66B2', '#FF9966', '#66FFB2', '#B2FF66', '#66FF66']
        for patch, color in zip(box[metric]['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title(f'Boxplot of {metric}')
        plt.suptitle('')
        plt.ylabel('Values')
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.97, bottom=0.05, left=0.03, right=0.99)
        
        metric_save_path = os.path.join(save_path, f'{metric}_combined_boxplot.png')
        plt.savefig(metric_save_path, dpi=400)
        plt.close()
        print(f"Image for {metric} saved at: {metric_save_path}")

base_path = '.'
latest_folder_path, excel_file_paths = get_latest_folder_contents(base_path)

if excel_file_paths:
    boxplot_path = os.path.join(latest_folder_path, 'boxplot')
    os.makedirs(boxplot_path, exist_ok=True)
    
    image_save_path = boxplot_path
    metrics = ['Accuracy', 'Precision Micro', 'Recall Micro', 'F1 Score Micro']
    
    read_and_plot_metrics(excel_file_paths, image_save_path, metrics)
