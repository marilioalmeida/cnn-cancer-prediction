import os
import pandas as pd
from datetime import datetime
import re
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook

def list_folders_in_directory(directory):
    """Lists all folders in a given directory."""
    folder_paths = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            folder_paths.append(folder_path)
    folder_paths.sort()
    return folder_paths

def get_relative_path(url, depth=4):
    """Gets a relative path with a specified depth."""
    parts = []
    while True:
        url, tail = os.path.split(url)
        if tail:
            parts.append(tail)
        else:
            if url:
                parts.append(url)
            break
    relative_path = os.path.join(*reversed(parts[:depth]))
    return relative_path

def parse_confusion_matrix(matrix_str):
    lines = matrix_str.strip().split('\n')
    headers = re.split(r'\s{2,}', lines[1].strip())[1:]
    matrix_data = {'CancerTypes': []}

    for header in headers:
        matrix_data[header] = []

    for line in lines[3:]:  # Ignoring the "Actual" line
        elements = re.split(r'\s{2,}', line.strip())
        actual_label = elements[0]
        row_data = [int(e) for e in elements[1:]]
        matrix_data['CancerTypes'].append(actual_label)
        for label, value in zip(headers, row_data):
            matrix_data[label].append(value)

    return pd.DataFrame(matrix_data)

def read_metrics_and_confusion_matrix_from_txt_file(file_path):
    """Reads metrics and confusion matrix from a single txt file."""
    metrics = {'file': get_relative_path(file_path)}
    confusion_matrix_str = ""
    with open(file_path, 'r') as f:
        capture_metrics = False
        capture_confusion_matrix = False
        for line in f:
            if "Confusion Matrix:" in line:
                capture_confusion_matrix = True
                capture_metrics = False
            if capture_confusion_matrix: 
                if "Accuracy:" in line:
                    capture_confusion_matrix = False
                    capture_metrics = True
                else:
                    confusion_matrix_str += line
            if capture_metrics:
                if "Accuracy:" in line:
                    metrics['Accuracy'] = float(line.split(': ')[1])
                elif "Precision Macro:" in line:
                    metrics['Precision Macro'] = float(line.split(': ')[1])
                elif "Precision Micro:" in line:
                    metrics['Precision Micro'] = float(line.split(': ')[1])
                elif "Recall Macro:" in line:
                    metrics['Recall Macro'] = float(line.split(': ')[1])
                elif "Recall Micro:" in line:
                    metrics['Recall Micro'] = float(line.split(': ')[1])
                elif "F1 Score Macro:" in line:
                    metrics['F1 Score Macro'] = float(line.split(': ')[1])
                elif "F1 Score Micro:" in line:
                    metrics['F1 Score Micro'] = float(line.split(': ')[1])
                elif "MCC:" in line:
                    metrics['MCC'] = float(line.split(': ')[1])
                elif "ROC AUC Score:" in line:
                    metrics['ROC AUC Score'] = float(line.split(': ')[1])
                elif "Log Loss:" in line:
                    metrics['Log Loss'] = float(line.split(': ')[1])
                elif "Average Precision Macro:" in line:
                    metrics['Average Precision Macro'] = float(line.split(': ')[1])
                elif "Average Precision Micro:" in line:
                    metrics['Average Precision Micro'] = float(line.split(': ')[1])
                if "Average Precision Micro:" in line:
                    break
    return metrics, confusion_matrix_str

def read_metrics_from_folder(folder):
    """Reads metrics and confusion matrices from all txt files in the validation folder of a given folder."""
    validation_folder = os.path.join(folder, 'validation')
    metrics_data = []
    confusion_matrices = []
    if os.path.exists(validation_folder) and os.path.isdir(validation_folder):
        for root, dirs, files in os.walk(validation_folder):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file) 
                    metrics, confusion_matrix_str = read_metrics_and_confusion_matrix_from_txt_file(file_path)
                    metrics_data.append(metrics)
                    if confusion_matrix_str:
                        confusion_matrices.append({'File': get_relative_path(file_path), 'Matrix': confusion_matrix_str})
    return metrics_data, confusion_matrices

def append_confusion_matrices(writer, matrices):
    """Appends confusion matrices to the Excel writer."""
    ws = writer.book.create_sheet(title='Confusion Matrix')
    for matrix_dict in matrices:
        file_name = matrix_dict.pop('File', 'Unknown File')
        matrix_str = matrix_dict.pop('Matrix')

        ws.append(['File:', file_name, '', '', '', '', '', '', '', '', '', '', '', '', ''])
        ws.append(['', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])

        matrix_df = parse_confusion_matrix(matrix_str)
        for r in dataframe_to_rows(matrix_df, index=False, header=True):
            ws.append(r)
        for _ in range(2):
            ws.append([])

def create_excel_from_metrics_and_confusion_matrix(metrics_data, confusion_matrices, output_file):
    """Creates an Excel file from metrics data and confusion matrices."""
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write metrics data to the first sheet
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

        # Append confusion matrices to the second sheet
        append_confusion_matrices(writer, confusion_matrices)

    print(f"\nExcel file created: {output_file}")

def main():
    """Main function to orchestrate the processing."""
    analyze_directory = os.path.join(os.getcwd(), 'analyze')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_directory = os.path.join(os.getcwd(), 'results', timestamp)

    os.makedirs(results_directory, exist_ok=True)

    folders = list_folders_in_directory(analyze_directory)
    
    metrics_data_by_root_folder = {}

    for folder in folders:
        root_folder_name = os.path.basename(os.path.dirname(folder))
        metrics_data, confusion_matrices = read_metrics_from_folder(folder)
        if metrics_data:
            if root_folder_name not in metrics_data_by_root_folder:
                metrics_data_by_root_folder[root_folder_name] = {'metrics': [], 'confusion_matrices': []}
            metrics_data_by_root_folder[root_folder_name]['metrics'].extend(metrics_data)
            metrics_data_by_root_folder[root_folder_name]['confusion_matrices'].extend(confusion_matrices)

    for root_folder_name, data in metrics_data_by_root_folder.items():
        output_file = os.path.join(results_directory, f'{root_folder_name}_metrics_data.xlsx')
        create_excel_from_metrics_and_confusion_matrix(data['metrics'], data['confusion_matrices'], output_file)

if __name__ == "__main__":
    main()
