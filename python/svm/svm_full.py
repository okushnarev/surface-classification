import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.svm import SVC


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/only-motor-currents',
                        required=True)
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/only-motor-currents',
                        required=True)
    parser.add_argument('--best_params_dir', '-bpd', type=str, default='python/svm/', required=True)
    parser.add_argument('--model_name', '-mn', type=str, default='svm', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    best_params_dir = Path(args.best_params_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')
    best_params_path = best_params_dir / (args.model_name + '_best.json')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)

    with open(best_params_path, 'r') as file:
        BEST_PARAMETERS = json.load(file)

    reg = SVC(**BEST_PARAMETERS)
    reg = reg.fit(X_train, np.ravel(y_train))

    dump(reg, output_model_joblib_path)
