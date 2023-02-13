import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from pathlib import Path
import argparse

import sys
sys.path.append('python')
from decisionTree.decision_tree_valid import metricsAndPlots

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=True)
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=True)
    parser.add_argument('--model_name', '-mn', type=str, default='data/models/',
                        required=True)
    parser.add_argument('--argument_pool', '-ap', type=str, default='motor-axis-currents',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    argument_pool = args.argument_pool
    model_name = args.model_name

    metrics_dir = 'metrics/{}/{}_metrics.json'.format(
        argument_pool, model_name
    )

    cm_plot_dir = 'plots/{}/{}_CM.png'.format(
        argument_pool, model_name
    )

    Path('metrics/' + argument_pool).mkdir(exist_ok=True, parents=True)
    Path('plots/' + argument_pool).mkdir(exist_ok=True, parents=True)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)
    le = preprocessing.LabelEncoder()
    le.fit(np.ravel(y_val))

    classifier = lgb.Booster(model_file=input_model)

    y_pred = classifier.predict(X_val)
    class_index = np.argmax(y_pred, axis=1)
    pred = le.inverse_transform(class_index)


    print('Score: ', accuracy_score(y_val, pred))

    metricsAndPlots(model_name, y_val, pred, metrics_dir, cm_plot_dir)