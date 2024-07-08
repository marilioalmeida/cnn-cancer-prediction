import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from  code_share.core import (
    configure_paths, train_model, validate_model, get_file_name
)
from keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet121 
from tensorflow.keras.layers import Dense, Dropout, Flatten
  
def build_cnn_model(input_shape):
    img_input = Input(shape=input_shape)
    img_conc = Concatenate()([img_input, img_input, img_input])   
    base_model = DenseNet121(include_top=False, input_tensor=img_conc, weights='imagenet')  
    x = base_model.output
    x = GlobalAveragePooling2D()(x) 
    predictions = Dense(12, activation='softmax')(x)  
    model = Model(inputs=base_model.input, outputs=predictions)  
    return model

 
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
