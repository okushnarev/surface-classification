import argparse
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier as Cat
from joblib import dump


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--argument_pool', '-ap', type=str, default='motor-axis-currents',
                        required=False, help='path to save prepared data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    argument_pool = str(Path(args.argument_pool))

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / 'CatClassifier_prod.joblib'

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)

    if argument_pool == 'motor-axis-currents':
        BEST_PARAMETERS = {'max_depth': 12, 'n_estimators': 40}

    elif argument_pool == 'motorCurrent-motorVelocities':
        BEST_PARAMETERS = {'max_depth': 13, 'n_estimators': 40}

    elif argument_pool == 'only-motor-currents':
        BEST_PARAMETERS = {'max_depth': 10, 'min_data_in_leaf': 5, 'n_estimators': 100}

    elif argument_pool == 'pure-motor-currents':
        BEST_PARAMETERS = {'max_depth': 8, 'min_data_in_leaf': 50, 'n_estimators': 40}

    else:
        print("ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR")

    reg = Cat(**BEST_PARAMETERS)
    reg = reg.fit(X_train, y_train, silent=True)

    dump(reg, output_model_joblib_path)
