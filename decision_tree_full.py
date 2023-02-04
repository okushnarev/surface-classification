import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
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

    if argument_pool == 'motor-axis-currents':
        TREES_MODELS_BEST_PARAMETERS = {
            'DecisionTree': {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 4, 'splitter': 'random'},
            'RandomForest': {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 25}}

    elif argument_pool == 'motorCurrent-motorVelocities':
        TREES_MODELS_BEST_PARAMETERS = {
            'DecisionTree': {'max_depth': 15, 'min_samples_leaf': 3, 'min_samples_split': 3, 'splitter': 'random'},
            'RandomForest': {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 35}}

    elif argument_pool == 'only-motor-currents':
        TREES_MODELS_BEST_PARAMETERS = {
            'DecisionTree': {'max_depth': 20, 'min_samples_leaf': 3, 'min_samples_split': 4, 'splitter': 'random'},
            'RandomForest': {'max_depth': 15, 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 35}}

    else:
        print("ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR")

    best_params = TREES_MODELS_BEST_PARAMETERS.get(args.model_name)
    reg = TREES_MODELS_MAPPER.get(args.model_name)(**best_params)
    if isinstance(reg, RandomForestClassifier):
        y_train = np.ravel(y_train.values)
    reg = reg.fit(X_train, y_train)

    dump(reg, output_model_joblib_path)