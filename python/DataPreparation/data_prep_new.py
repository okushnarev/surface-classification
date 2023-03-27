import pandas as pd
import argparse
from pathlib import Path

from data_prep import to_categorical, selectFeatures


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--argument_pool', '-ap', type=str, default='motor-axis-currents',
                        required=False, help='path to save prepared data')
    return parser.parse_args()


def cols_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for x in list(df.columns.values):
        df[x] = pd.to_numeric(df[x], errors='coerce').fillna(0)
    return df


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    argument_pool = str(Path(args.argument_pool))

    output_dir.mkdir(exist_ok=True, parents=True)

    surf_types = ['grey', 'green', 'table']

    for surf in surf_types:
        sheet = pd.read_csv(str(input_dir / surf) + ".csv", sep=';')
        sheet = cols_to_numeric(sheet)
        sheet['sum_motor_current'] = sheet[['m1cur', 'm2cur', 'm3cur']].abs().sum(axis=1)
        sheet['sum_axis_current'] = sheet[['xcur', 'ycur', 'rotcur']].abs().sum(axis=1)
        sheet[['xsetspeed', 'ysetspeed']] *= 1000
        sheet['rotational'] = 0
        sheet['surface_type'] = surf
        sheet.to_csv(str(output_dir / surf) + "_surf.csv", index=False)

    df_list = (pd.read_csv(str(output_dir / file) + '_surf.csv') for file in surf_types)

    df = pd.concat(df_list, ignore_index=True)
    df = to_categorical(df, surf_types)
    df.to_csv(output_dir / "merged.csv", index=False)

    df = selectFeatures(argument_pool, df)

    X = df.drop('surface_type', axis=1)
    y = df['surface_type']

    X_full_name = output_dir / 'X_full.csv'
    y_full_name = output_dir / 'y_full.csv'

    X.to_csv(X_full_name, index=False)
    y.to_csv(y_full_name, index=False)
