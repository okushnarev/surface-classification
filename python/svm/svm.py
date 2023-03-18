import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.svm import SVC
from joblib import dump, load
import random
from sklearn.model_selection import GridSearchCV
import json


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=True)
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=True)
    parser.add_argument('--best_params_dir', '-bpd', type=str, default='python/svm/', required=True)
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=True)
    parser.add_argument('--model_name', '-mn', type=str, default='svm', required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['svm']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    best_params_dir = Path(args.best_params_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')
    best_params_path = best_params_dir / (args.model_name + '_best.json')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    scorer = make_scorer(accuracy_score)

    random.seed(42)

    model = SVC(verbose=False)

    classifier = GridSearchCV(model, params, scoring=scorer, n_jobs=-1)
    classifier = classifier.fit(X_train, np.ravel(y_train))

    print("Score: ", classifier.best_estimator_.score(X_test, y_test))
    print(classifier.best_params_)

    with open(best_params_path, 'w') as file:
        json.dump(classifier.best_params_, file, indent=4)

    dump(classifier.best_estimator_, output_model_joblib_path)
