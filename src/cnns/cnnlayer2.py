import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  code_share.core import (
    configure_paths, train_model, validate_model, get_file_name
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
 
def build_cnn_model(input_shape):
    return Sequential([
        Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(12, activation='softmax')
    ])
 
def main():
    #Configure paths and environment
    model_name = get_file_name(__file__)
    model = build_cnn_model((100, 100, 1))
    config = configure_paths(model_name)
  
    # Train the model
    model = train_model(config, model)

    # Validate the model
    validate_model(model, config) 
   
if __name__ == "__main__":
    main()
