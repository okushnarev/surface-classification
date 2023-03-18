import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeClassifier,
                       'RandomForest': RandomForestClassifier}




def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--argument_pool', '-ap', type=str, default='motor-axis-currents',
                        required=False, help='path to save prepared data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    argument_pool = str(Path(args.argument_pool))

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.jpg')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    y_train_cols = y_train.columns


    pools = {
        'motor-axis-currents': {
            'DecisionTree': {'max_depth': 15, 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'best'},
            'RandomForest': {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 25}
        },

        'motorCurrent-motorVelocities': {
            'DecisionTree': {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 3, 'splitter': 'random'},
            'RandomForest': {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 35}
        },

        'only-motor-currents': {
            'DecisionTree': {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 5, 'splitter': 'best'},
            'RandomForest': {'max_depth': 25, 'min_samples_leaf': 15, 'min_samples_split': 25, 'n_estimators': 100}
        },

        'pure-motor-currents': {
            'DecisionTree': {'max_depth': 10, 'min_samples_leaf': 15, 'min_samples_split': 10, 'splitter': 'best'},
            'RandomForest': {'max_depth': 20, 'min_samples_leaf': 15, 'min_samples_split': 15, 'n_estimators': 25}
        },

        'fuzzy-input-motCur-commandedVel': {
            'DecisionTree': {'max_depth': 10, 'min_samples_leaf': 35, 'min_samples_split': 10, 'splitter': 'best'},
            'RandomForest': {'max_depth': 10, 'min_samples_leaf': 15, 'min_samples_split': 35, 'n_estimators': 25}
        },

        'fuzzy-input-motCur': {
            'DecisionTree': {'max_depth': 15, 'min_samples_leaf': 10, 'min_samples_split': 10, 'splitter': 'best'},
            'RandomForest': {'max_depth': 20, 'min_samples_leaf': 15, 'min_samples_split': 25, 'n_estimators': 25}
        },
    }

    if argument_pool in pools:
        BEST_PARAMETERS = pools[argument_pool]
    else:
        raise Exception('No such argument pool')

    best_params = BEST_PARAMETERS.get(args.model_name)
    reg = TREES_MODELS_MAPPER.get(args.model_name)(**best_params)
    if isinstance(reg, RandomForestClassifier):
        y_train = np.ravel(y_train.values)
    reg = reg.fit(X_train, y_train)

    dump(reg, output_model_joblib_path)