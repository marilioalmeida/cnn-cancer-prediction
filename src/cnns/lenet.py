import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  code_share.core import (
    configure_paths, train_model, validate_model, get_file_name
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
 
def build_cnn_model(input_shape):
    return Sequential([
        Conv2D(6, (5, 5), activation='relu', input_shape=input_shape, padding='same'),
        AveragePooling2D(),
        Conv2D(16, (5, 5), activation='relu', padding='same'),
        AveragePooling2D(),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
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
