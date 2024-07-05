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
 
# def build_cnn_model(input_shape):
#     img_input = Input(shape=input_shape)
#     img_conc = Concatenate()([img_input, img_input, img_input])   
#     base_model = DenseNet121(include_top=False,  input_tensor=img_conc, weights='imagenet')  
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
#     predictions = Dense(12, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#     return model

def build_cnn_model(input_shape):
    img_input = Input(shape=input_shape)
    img_conc = Concatenate()([img_input, img_input, img_input])  # Concatenando a entrada três vezes para criar uma entrada de 3 canais
    base_model = DenseNet121(include_top=False, input_tensor=img_conc, weights='imagenet')  # Usando DenseNet121 com pesos do ImageNet
    
    # Adicionando camadas para ajustar a saída do modelo
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    predictions = Dense(12, activation='softmax')(x)  # Camada de saída com 12 unidades e ativação softmax
    
    model = Model(inputs=base_model.input, outputs=predictions)  # Criando o modelo
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
