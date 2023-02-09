import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeClassifier,
                       'RandomForest': RandomForestClassifier}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='Decision Tree', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['decision_tree']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

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
    decision_tree_model = TREES_MODELS_MAPPER.get(args.model_name)()
    decision_tree_classifier = GridSearchCV(decision_tree_model, params[args.model_name], scoring=scorer, n_jobs=-1)

    if isinstance(decision_tree_model, RandomForestClassifier):
        y_train = np.ravel(y_train.values)
        y_test = np.ravel(y_test.values)
    decision_tree_classifier = decision_tree_classifier.fit(X=X_train, y=y_train)

    # create dummy classifier
    dummy_clf = DummyClassifier(strategy='stratified', random_state=42)
    dummy_clf.fit(X_train, y_train)

    print('Score: ', decision_tree_classifier.best_estimator_.score(X_test, y_test))
    print('Baseline Score: ', dummy_clf.score(X_test, y_test))
    print()
    print(decision_tree_classifier.best_params_)
    print()

    dump(decision_tree_classifier.best_estimator_, output_model_joblib_path)