import argparse
import random
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=True)
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=True)
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['light_gbm']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / 'lightGBM.joblib'

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    y_train = np.ravel(y_train.values)
    y_test = np.ravel(y_test.values)

    random.seed(42)

    scorer = make_scorer(accuracy_score)

    lgbm_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, verbose=-1, n_jobs=4)
    lgbm_classifier = GridSearchCV(lgbm_model, params, scoring=scorer, n_jobs=-1)

    lgbm_classifier = lgbm_classifier.fit(X_train, y_train)

    best = lgbm_classifier.best_estimator_

    print("Score: ", best.score(X_test, y_test))
    # print("Baseline Score: ", baseline_model.score(X_test, y_test))
    print(lgbm_classifier.best_params_)



    best.booster_.save_model(output_model_joblib_path)
    # dump(best, output_model_joblib_path)