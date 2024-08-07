import os
import glob
import pandas as pd

def get_latest_folder_contents(base_path):
    # Navigate to the 'results' folder
    results_path = os.path.join(base_path, 'results')
    
    # Get all subfolders within 'results'
    subfolders = [f for f in glob.glob(results_path + "/*") if os.path.isdir(f)]
    
    # Check if there are subfolders
    if not subfolders:
        print("No subfolders found in 'results'.")
        return None, None
    
    # Find the newest subfolder
    latest_subfolder = max(subfolders, key=os.path.getmtime)
    
    # List the contents of the newest subfolder
    contents = [f for f in glob.glob(latest_subfolder + "/*.xlsx") if os.path.isfile(f)]
    
    if contents:
        return latest_subfolder, contents
    else:
        print(f"No Excel files found in the newest folder.")
        return None, None

# Call the function with the base path where the 'results' folder is located
base_path = '.'  # You can adjust the base path as necessary
latest_folder_path, latest_files = get_latest_folder_contents(base_path)

if latest_files:
    # Combine data from all Excel files
    combined_data = pd.DataFrame()
    for file in latest_files:
        data = pd.read_excel(file)
        combined_data = pd.concat([combined_data, data], ignore_index=True)
    
    # Modify the 'file' column to show only the first name
    combined_data['file'] = combined_data['file'].apply(lambda x: x.split('\\')[0])
    
    # Create a new Excel file with multiple sheets
    output_path = os.path.join(latest_folder_path, 'summary_data.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        # Create a sheet for each metric
        # metrics = ['Accuracy', 'Precision Micro', 'Recall Micro', 'F1 Score Micro', 'MCC', 'ROC AUC Score', 'Log Loss']
        metrics = ['Accuracy', 'Precision Micro', 'Recall Micro', 'F1 Score Micro']
        
        for metric in metrics:
            if metric in combined_data.columns:
                # Group the data by the 'file' field and calculate statistics
                df_grouped = combined_data.groupby('file')[metric].agg(['mean', 'median', 'min', 'max', 'var']).reset_index()
                df_grouped.columns = ['file', 'Mean', 'Median', 'Min', 'Max', 'Variance']
                
                # Save to a new sheet
                df_grouped.to_excel(writer, sheet_name=metric, index=False)

    print(f"New file created: {output_path}")
else:
    print("No Excel files were found and the new Excel file was not generated.")
