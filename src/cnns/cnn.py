import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  code_share.core import (
    configure_paths, train_model, validate_model, get_file_name
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
 
def build_cnn_model(input_shape):
    return Sequential([
        Conv2D(64, kernel_size=(5, 5), padding="same", input_shape=input_shape, activation='relu'),
        BatchNormalization(axis=-1, epsilon=1e-5),
        MaxPooling2D(pool_size=(2, 2), padding="same"),
        Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
        BatchNormalization(axis=-1, epsilon=1e-5),
        MaxPooling2D(pool_size=(2, 2), padding="same"),
        Conv2D(64, kernel_size=(3, 3), padding="same", activation='relu'),
        BatchNormalization(axis=-1, epsilon=1e-5),
        MaxPooling2D(pool_size=(2, 2), padding="same"),
        Flatten(),
        Dense(1000, activation='relu'),
        Dense(600, activation='relu'),
        Dense(80, activation='relu'),
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
