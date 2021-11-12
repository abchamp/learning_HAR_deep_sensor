import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
import tensorflow as tf
from models import Models

featureMappingIndex = { 'accX': 0, 'accY': 1, 'accZ': 2, }
samplingRate = { 'realWorld': 50, 'mHealth': 50, 'shoaib': 50, 'wisdm': 20}
totalFold = { 'realWorld': 10, 'mHealth': 10, 'shoaib': 10, 'wisdm': 45}
windowSizeMultiply = 2
# mhealth 30720 - 200 => 30320 => 2668 / 100 => 26 * 8 * 10 => 280 per act

def slidingWindow(data):
    _trueLabels = list()
    _featureStack = [[], [], []] 
    windowSize = samplingRate[dataset] * windowSizeMultiply
    for act in data['act'].unique():
        for user in data['user'].unique():
            selected_act_data = data[(data['act'] == act) & (data['user'] == user)]
            for window in range(0, selected_act_data.shape[0], windowSize):
                windows = selected_act_data.iloc[window: window+windowSize]
                if(windows.shape[0] != windowSize):
                    break
                        
                _trueLabels.append(act)
                for ssAxis in selected_axis:
                    _featureStack[featureMappingIndex[ssAxis]].append(windows[ssAxis].values.copy())
                        
    return np.dstack(np.array(_featureStack)), np.array(_trueLabels)

def myLabelEncode(y_train, y_test):
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

if __name__ == '__main__':
    load_model = False
    ###
    datasetName = ['mHealth', 'shoaib', 'realWorld', 'wisdm']
    selectedAxis = ['accX', 'accY', 'accZ']
    epochs, batch_size = 15, 64
    # verbose, n_steps, n_length = 1, 4, 32
    verbose, n_steps, n_length = 1, 4, 25
    
    # rawResult = {
    #     'tLabel': [],
    #     'pLabel': [],
    #     'fold': [],
    # }

    selected_axis = ['accX', 'accY', 'accZ']
    for dataset in datasetName:
        for kfold in range(1, totalFold[dataset]+1):
            rawResult = {
                'tLabel': [],
                'pLabel': []
            }

            # example shape => # (7352, 128, 9) (row, windowsize, column)
            df_data = pd.read_csv(f"./data/nvm0/{dataset}-training-{kfold}.csv")
            trStackFeatures, trTrueLabels = slidingWindow(df_data)
            df_data = pd.read_csv(f"./data/nvm0/{dataset}-testing-{kfold}.csv")
            teStackFeatures, teTrueLabels = slidingWindow(df_data)
            print(f"testing size: {teTrueLabels.shape}")
            encodedTrTrueLabels, encodedTeTrueLabels = myLabelEncode(trTrueLabels, teTrueLabels)
            cTrTrueLabels = to_categorical(encodedTrTrueLabels)
            cTeTrueLabels = to_categorical(encodedTeTrueLabels)
            
            n_timesteps, n_features, n_outputs = trStackFeatures.shape[1], trStackFeatures.shape[2], cTrTrueLabels.shape[1]
            
            model_name = 'cnnlstm'
            model = None
    
            if load_model == False:
                if model_name == 'cnnlstm':
                    # (row, windowsize, column) => 
                    # (width, height, color_depth) =>
                    # (sample_size, width, height, color_depth) = 4D
                    trStackFeatures = trStackFeatures.reshape((trStackFeatures.shape[0], n_steps, n_length, n_features))
                    teStackFeatures = teStackFeatures.reshape((teStackFeatures.shape[0], n_steps, n_length, n_features))
                elif model_name == 'convlstm':
                    print(trStackFeatures.shape)
                    trStackFeatures = trStackFeatures.reshape((trStackFeatures.shape[0], n_steps, 1, n_length, n_features))
                    teStackFeatures = teStackFeatures.reshape((teStackFeatures.shape[0], n_steps, 1, n_length, n_features))
                
                model = Models(model_name, n_timesteps, n_features, n_outputs, n_steps, n_length)
                model.model.fit(trStackFeatures, cTrTrueLabels, epochs=epochs, batch_size=batch_size, verbose=verbose)
                # model.model.save(f'./model/{dataset}_{kfold}')
            else:
                model = tf.keras.models.load_model(f'./model/{dataset}_{kfold}')
            
            raw_pred = model.model.predict(teStackFeatures, batch_size=batch_size)
            rawResult['tLabel'] = encodedTeTrueLabels
            rawResult['pLabel'] = raw_pred.argmax(axis=1)
            result_df = pd.DataFrame(rawResult)
            result_df.to_csv(f'./raw_results/{dataset}_{model_name}_{kfold}.csv', index=False)
     
          
            
         