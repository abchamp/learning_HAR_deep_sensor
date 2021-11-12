from sklearn import metrics
import pandas as pd

user_counts = {
    'mHealth': 10,
    'realWorld': 10,
    'shoaib': 10,
    'wisdm': 45,
}

model_name = 'lstm' # convlstm

if __name__ == '__main__':
    for dataset in ['realWorld','shoaib', 'mHealth']:
        result = {
            'pLabel': [],
            'tLabel': [],
        }
      
        for fold in range(1, user_counts[dataset]+1):
            data = pd.read_csv(f"./raw_results/{dataset}_{model_name}_{fold}.csv")
            result['pLabel'].extend(data['pLabel'].tolist())
            result['tLabel'].extend(data['tLabel'].tolist())
        print(metrics.confusion_matrix(result['tLabel'], result['pLabel']))