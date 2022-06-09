import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import glob
import librosa as lbr
from librosa.feature import mfcc

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, SimpleRNN, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.optimizers import  Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Precision, Recall

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, top_k_accuracy_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


data_loso = pd.read_csv('/home/mishra.g/spring2022/hci/project/all_data_with_subject_info.csv', index_col=0)

subjects = data_loso.index.unique().values
labels = ['No Pathology','Bruxism','Insomnia','Narcolepsy','NFLE','PLM','RBD','SDB']
metrics_header = ['Subject Name', 'Train Samples','Val Samples','Tr:Accuracy', 'Tr:Precision','Tr:Recall','Tr:F1', 'Val:Accuracy', 'Val:Precision','Val:Recall','Val:F1']

metrics = []
for one_subject in subjects:
    #if(one_subject == 'ins3'):
    #    continue;
    print(f"Validating on subject:{one_subject}");
    data_train = data_loso[data_loso.index != one_subject]
    data_val = data_loso[data_loso.index==one_subject]
    print(f"Training on {len(data_train)} data points and validating on {len(data_val)}({(len(data_val)*100/len(data_loso))}%)data points")
    
    x_train = data_train.iloc[:,:-1]
    y_train = data_train.iloc[:,-1]
    x_val = data_val.iloc[:,:-1]
    y_val = data_val.iloc[:,-1]
    print('Training Started')
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', n_jobs=-1).fit(x_train, y_train);
    print('Training Done!')
    print('Training Metrics')
    y_pred_tr = rf.predict(x_train)
    y_pred_proba = rf.predict_proba(x_train)
    print(classification_report(y_pred_tr,y_train.values, target_names= labels))

    
    print('\nValidation Metrics')
    y_pred_val = rf.predict(x_val)
    print(classification_report(y_pred_val,y_val.values))#, target_names= ['Insomnia']))
    
    
    metrics.append([one_subject,len(x_train) ,len(x_val), accuracy_score(y_pred_tr, y_train.values), precision_score(y_pred_tr, y_train.values, average='macro'), recall_score(y_pred_tr, y_train.values,average='macro'), f1_score(y_pred_tr, y_train.values,average='macro'), accuracy_score(y_pred_val, y_val.values), precision_score(y_pred_val, y_val.values,average='macro'), recall_score(y_pred_val, y_val.values, average='macro'), f1_score(y_pred_val, y_val.values,average='macro')])
    
    #break;
    print('Training and validation done! Moving on to next...')
    print('\n')
    
    
metrics = pd.DataFrame(metrics)

metrics.to_csv('/home/mishra.g/spring2022/hci/project/loso_metrics.csv', header=metrics_header, index=False)