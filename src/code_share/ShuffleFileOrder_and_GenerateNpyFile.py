import numpy as np
import os
import random
from datetime import datetime as DT
import shutil

def clear_output_directory(output_directory):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    print(f"Contents of the folder {output_directory} successfully deleted.")

def process_samples(path, output_file_name):
    dirs = os.listdir(path)

    time = DT.utcnow()
    seed = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute) + str(time.microsecond)
    random.seed(seed)
    random.shuffle(dirs)  # shuffle input order
    print(seed)

    sample_titles = np.array(dirs)
    samples = []
    labels = []
    elements = []
    O_names = []

    for name in dirs:
        O_names.append(name)
        data = name.split('-')
        labels.append(data[2])
        elements_name = []
        with open(os.path.join(path, name)) as file:
            for line in file:
                line = line.strip().split()
                elements_name.append(line)
        elements.append(elements_name)

    x_samples = np.array(elements).astype(np.float32)
    y_labels = np.array(labels).astype(np.int32)
    np.save(output_file_name + ".npy", x_samples)
    np.save(output_file_name + "_label.npy", y_labels)
    np.save(output_file_name + "_title.npy", sample_titles)

    print(x_samples.shape, y_labels.shape)
    print(f"Processing of {output_file_name} completed successfully.")

# Clear output directory
output_directory = './datasets/output/'
clear_output_directory(output_directory)

# Process training samples
training_path = './datasets/training_samples/'
training_output_file_name = './datasets/output/Example_training_1228_TCGA_samples'
process_samples(training_path, training_output_file_name)

# Process validation samples
validation_path = './datasets/validation_samples/'
validation_output_file_name = './datasets/output/Example_validation_4908_TCGA_samples'
process_samples(validation_path, validation_output_file_name)
