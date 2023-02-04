import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    X_val_name = input_dir / 'X_full.csv'
    y_val_name = input_dir / 'y_full.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    reg = load(input_model)

    print('Score on NEW DATA: ', reg.score(X_val, y_val))