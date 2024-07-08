import os
import io
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime as DT 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import seaborn as sns
import tensorflow.keras.backend as K
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, classification_report, roc_auc_score, 
    log_loss, average_precision_score 
)
import pandas as pd

def configure_paths(model_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    matplotlib.use("Agg")
    MODEL_NAME = model_name
    timestamp = DT.now().strftime("%Y-%m-%d-%H-%M") 
    TRAINING_OUTPUT_DIR = f"./result/{model_name}/{timestamp}/training"
    VALIDATION_OUTPUT_DIR = f"./result/{model_name}/{timestamp}/validation"
    BASE_INPUT_PATH = './datasets/output/' 
    TRAINING_FILE_BASE = 'training_1228_TCGA'
    VALIDATION_FILE_BASE = 'validation_4908_TCGA'
    TRAINING_OUTPUT_FILE = f"{TRAINING_OUTPUT_DIR}/{TRAINING_FILE_BASE}"
    TRAINING_SET_PATH = f"{BASE_INPUT_PATH}{TRAINING_FILE_BASE}.npy"
    TRAINING_LABEL_PATH = f"{BASE_INPUT_PATH}{TRAINING_FILE_BASE}_label.npy"
    VALIDATION_SET_PATH = f"{BASE_INPUT_PATH}{VALIDATION_FILE_BASE}.npy"
    VALIDATION_LABEL_PATH = f"{BASE_INPUT_PATH}{VALIDATION_FILE_BASE}_label.npy"
    FILENAME_LIST_PATH = f"{BASE_INPUT_PATH}{VALIDATION_FILE_BASE}_title.npy"
    MODEL_WEIGHTS_PATH = f"{BASE_INPUT_PATH}{TRAINING_FILE_BASE}.weights.h5"
    BEST_MODEL_WEIGHTS_PATH = f"{BASE_INPUT_PATH}{TRAINING_FILE_BASE}_best_weights.keras"
    NUM_EPOCHS = 100
    
    os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
    
    return {
        "MODEL_NAME": MODEL_NAME,
        "TRAINING_OUTPUT_DIR": TRAINING_OUTPUT_DIR,
        "VALIDATION_OUTPUT_DIR": VALIDATION_OUTPUT_DIR,
        "BASE_INPUT_PATH": BASE_INPUT_PATH, 
        "TRAINING_FILE_BASE": TRAINING_FILE_BASE,
        "VALIDATION_FILE_BASE": VALIDATION_FILE_BASE,
        "TRAINING_OUTPUT_FILE": TRAINING_OUTPUT_FILE,
        "TRAINING_SET_PATH": TRAINING_SET_PATH,
        "TRAINING_LABEL_PATH": TRAINING_LABEL_PATH,
        "VALIDATION_SET_PATH": VALIDATION_SET_PATH,
        "VALIDATION_LABEL_PATH": VALIDATION_LABEL_PATH,
        "FILENAME_LIST_PATH": FILENAME_LIST_PATH,
        "MODEL_WEIGHTS_PATH": MODEL_WEIGHTS_PATH,
        "BEST_MODEL_WEIGHTS_PATH": BEST_MODEL_WEIGHTS_PATH,
        "NUM_EPOCHS": NUM_EPOCHS
    }

def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def save_train_history(history, train_metric, val_metric, filename):
    plt.plot(history.history[train_metric])
    plt.plot(history.history[val_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(filename + ".png", bbox_inches="tight")
    plt.close()

class Metrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        print(f"F1 score: {_val_f1:.4f} - Recall: {_val_recall:.4f} - Precision: {_val_precision:.4f}")
        logs['val_f1s'] = _val_f1
        logs['val_recalls'] = _val_recall
        logs['val_precisions'] = _val_precision
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

def remove_existing_files(config):
    if os.path.exists(config["MODEL_WEIGHTS_PATH"]):
        os.remove(config["MODEL_WEIGHTS_PATH"])
    if os.path.exists(config["BEST_MODEL_WEIGHTS_PATH"]):
        os.remove(config["BEST_MODEL_WEIGHTS_PATH"])

def get_file_name(file_path):
    # Obter o nome do arquivo atual
    file_name = os.path.basename(file_path)
    
    # Remover a extensão .py
    file_name_without_extension = os.path.splitext(file_name)[0]
    
    # Retornar a mensagem em inglês
    return file_name_without_extension


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision Macro": precision_score(y_true, y_pred, average='macro'),
        "Precision Micro": precision_score(y_true, y_pred, average='micro'),
        "Recall Macro": recall_score(y_true, y_pred, average='macro'),
        "Recall Micro": recall_score(y_true, y_pred, average='micro'),
        "F1 Score Macro": f1_score(y_true, y_pred, average='macro'),
        "F1 Score Micro": f1_score(y_true, y_pred, average='micro'),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC AUC Score": roc_auc_score(y_true, y_prob, multi_class='ovr'),
        "Log Loss": log_loss(y_true, y_prob),
        "Average Precision Macro": average_precision_score(y_true, y_prob, average='macro'),
        "Average Precision Micro": average_precision_score(y_true, y_prob, average='micro'),
    }

def plot_confusion_matrix(config, df, row_range=None, col_range=None):
    subset = df.iloc[row_range[0]:row_range[1], col_range[0]:col_range[1]] if row_range and col_range else df
    plt.figure(figsize=(18, 8))
    ax = sns.heatmap(subset, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=subset.columns, yticklabels=subset.index)
    ax.hlines([i for i in range(1, subset.shape[0])], *ax.get_xlim(), colors='black', linewidths=0.5)
    ax.vlines([i for i in range(1, subset.shape[1])], *ax.get_ylim(), colors='black', linewidths=0.5)
    plt.xlabel('Predicted classes')
    plt.ylabel('Truth classes')
    plt.savefig(f"{config['VALIDATION_OUTPUT_DIR']}/confusion_matrix.png", dpi=500, bbox_inches='tight')
    plt.close()

def evaluate_model(model, config, x_validation, validation_filenames, y_validation, y_validation_compare): 
    result_file_path = os.path.join(config["VALIDATION_OUTPUT_DIR"], f"{config['MODEL_NAME']}_{os.path.splitext(os.path.basename(config['VALIDATION_SET_PATH']))[0]}_predicted_result.txt")

    with open(result_file_path, "w") as out:                
        model.load_weights(config["MODEL_WEIGHTS_PATH"])
        adam = Adam(learning_rate=1e-4)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', mcc])

        # Evaluate model
        scores = model.evaluate(x_validation, y_validation, verbose=1)
       
        # Predictions
        predictions = model.predict(x_validation)
        predicted_labels = predictions.argmax(axis=-1)
 
        num = 0
        start = 1
        print("Predict-result:", file=out)
        for predicted_label in predicted_labels:
            actual_label = int(str(validation_filenames[num]).split("-")[2])
            if actual_label != predicted_label:
                print(f"{start}\tSample Name:\t{validation_filenames[num]}\tActual:\t{actual_label}\tPredict:\t{predicted_label}", file=out)
                start += 1
            num += 1
        
        new_labels = {
            0: 'Normal', 1: 'BRCA', 2: 'KIRC', 3: 'LUAD', 4: 'THCA', 5: 'PRAD', 
            6: 'LUSC', 7: 'LIHC', 8: 'HNSC', 9: 'COAD', 10: 'STAD', 11: 'KIRP'
            }
        
        cm = pd.crosstab(y_validation_compare, predicted_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)
        cm = cm.rename(index=new_labels, columns=new_labels)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_columns', None)
        print(f"\n\nConfusion Matrix:\n{cm}", file=out)
        print(f"\n\nConfusion Matrix:\n{cm}")
        plot_confusion_matrix(config, cm, (0, 11), (0, 11))

        val_metrics = calculate_metrics(y_validation_compare, predicted_labels, predictions)      
        out.write("\n")
        for metric, value in val_metrics.items():
             print(f"{metric}: {value:.3f}\n", file=out) 
             print(f"{metric}: {value:.3f}\n") 
        report = classification_report(y_validation_compare, predicted_labels)
        out.write("Report: \n")
        out.write(report)   

def save_training_log(config, history, train_metrics, y_train_labels, train_pred):
    log_file = f"{config['TRAINING_OUTPUT_FILE']}_training_log.txt"
    with open(log_file, 'w') as out:
        for step, (i, mcc_val) in enumerate(zip(history.history['accuracy'], history.history['mcc']), 1):
            out.write(f'Epoch:\t{step}/{config["NUM_EPOCHS"]}\tLoss:\t{history.history["loss"][step-1]:.6f}\tAcc:\t{i:.6f}\tMCC:\t{mcc_val:.6f}\tVal_Loss:\t{history.history["val_loss"][step-1]:.6f}\tVal_Acc:\t{history.history["val_accuracy"][step-1]:.6f}\tVal_MCC:\t{history.history["val_mcc"][step-1]:.6f}\n')
        for metric, value in train_metrics.items():
            out.write(f"{metric}: {value:.3f}\n")
        out.write("Report: \n")
        out.write(classification_report(y_train_labels, train_pred))

def save_model_summary(config, model):
    # Capturar o resumo do modelo
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    # Definir o caminho do arquivo para salvar o resumo do modelo
    summary_file = f"{config['TRAINING_OUTPUT_FILE']}_training_modelsummary.txt"

    # Salvar o resumo do modelo no arquivo
    with open(summary_file, 'w') as out:
        out.write(summary_string)

def train_model(config, model):
    remove_existing_files(config)

    # Load and preprocess training data
    x_train = np.load(config["TRAINING_SET_PATH"]).reshape((-1, 100, 100, 1))
    y_train = to_categorical(np.load(config["TRAINING_LABEL_PATH"]))

    # Build and compile model 
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy', mcc])
   

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor="val_mcc", patience=60, mode="max"),
        ModelCheckpoint(config["BEST_MODEL_WEIGHTS_PATH"], monitor="val_mcc", save_best_only=True, mode="max")
    ]
 
    # Train model
    history = model.fit(x_train, y_train, validation_split=0.25, epochs=config["NUM_EPOCHS"], batch_size=24, callbacks=callbacks)

    train_pred = model.predict(x_train).argmax(axis=-1)
    y_train_labels = y_train.argmax(axis=1)

    train_metrics = calculate_metrics(y_train_labels, train_pred, model.predict(x_train))
    save_training_log(config, history, train_metrics, y_train_labels, train_pred)
    save_model_summary(config, model) 

    # Save final model weights
    model.save_weights(config["MODEL_WEIGHTS_PATH"])

    # Plot training history
    save_train_history(history, "accuracy", "val_accuracy", f"{config['TRAINING_OUTPUT_FILE']}_accuracy")
    save_train_history(history, "loss", "val_loss", f"{config['TRAINING_OUTPUT_FILE']}_loss")
    
    return model

def validate_model(model, config):
    # Load validation data
    x_validation = np.load(config["VALIDATION_SET_PATH"]).reshape((-1, 100, 100, 1))
    y_validation = to_categorical(np.load(config["VALIDATION_LABEL_PATH"]))
    y_validation_compare = np.load(config["VALIDATION_LABEL_PATH"])
    validation_filenames = np.load(config["FILENAME_LIST_PATH"])

    # Evaluate model
    evaluate_model(model, config, x_validation, validation_filenames, y_validation, y_validation_compare)
    remove_existing_files(config)
 