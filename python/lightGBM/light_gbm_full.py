import argparse
from pathlib import Path
import lightgbm as lgb
import pandas as pd


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False)
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False)
    parser.add_argument('--argument_pool', '-ap', type=str, default='motor-axis-currents',
                        required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    argument_pool = str(Path(args.argument_pool))

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / 'lightGBM_prod.joblib'

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)


    if argument_pool == 'motor-axis-currents':
        BEST_PARAMETERS = {'boosting': 'dart', 'learning_rate': 0.1, 'max_bin': 255, 'num_iterations': 150, 'num_leaves': 30}

    elif argument_pool == 'motorCurrent-motorVelocities':
        BEST_PARAMETERS = {'boosting': 'dart', 'learning_rate': 0.1, 'max_bin': 355, 'num_iterations': 150, 'num_leaves': 40}

    elif argument_pool == 'only-motor-currents':
        BEST_PARAMETERS = {'boosting': 'dart', 'learning_rate': 0.05, 'max_bin': 255, 'num_iterations': 200, 'num_leaves': 40}

    else:
        print("ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR-ERROR")

    lgbm_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbose=-1, n_jobs=4, **BEST_PARAMETERS)
    
    lgbm_model.fit(X_full, y_full)

    lgbm_model.booster_.save_model(output_model_joblib_path)
