import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def to_categorical(df: pd.DataFrame):
    df['surface_type'] = pd.Categorical(df['surface_type'])
    df = df.assign(surface_type=df.surface_type.cat.codes)
    return df

def exportDF(df: pd.DataFrame, dir):
    X = df.drop('surface_type', axis=1)
    y = df['surface_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=params.get('train_test_ratio'),
                                                        random_state=params.get('random_state'))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=params.get('train_val_ratio'),
                                                      random_state=params.get('random_state'))
    concatDir = output_dir / dir
    concatDir.mkdir(exist_ok=True, parents=True)

    X_full_name = concatDir / 'X_full.csv'
    y_full_name = concatDir / 'y_full.csv'
    X_train_name = concatDir / 'X_train.csv'
    y_train_name = concatDir / 'y_train.csv'
    X_test_name = concatDir / 'X_test.csv'
    y_test_name = concatDir / 'y_test.csv'
    X_val_name = concatDir / 'X_val.csv'
    y_val_name = concatDir / 'y_val.csv'

    X.to_csv(X_full_name, index=False)
    y.to_csv(y_full_name, index=False)
    X_train.to_csv(X_train_name, index=False)
    y_train.to_csv(y_train_name, index=False)
    X_test.to_csv(X_test_name, index=False)
    y_test.to_csv(y_test_name, index=False)
    X_val.to_csv(X_val_name, index=False)
    y_val.to_csv(y_val_name, index=False)


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_prep']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    surf_types = ['grey', 'green', 'table']

    for surf in surf_types:
        sheet = pd.read_csv(str(input_dir / surf) + ".csv", sep=';')
        sheet['sum_axis_current'] = sheet['Current_X_A'].abs() + sheet['Current_Y_A'].abs() + sheet['Current_Z_A'].abs()
        sheet['rotational'] = np.where(sheet['Rotational_speed_deg_s'] == 0, 0, 1)
        sheet['surface_type'] = surf
        sheet.to_csv(str(output_dir / surf) + "_surf.csv", index=False)

    df_list = (pd.read_csv(str(output_dir / file) + '_surf.csv') for file in surf_types)

    df = pd.concat(df_list, ignore_index=True)
    df = to_categorical(df)

    old_names = ['X_axis_speed_mm_s', 'Y_axis_speed_mm_s', 'Motor_1_current_A', 'Motor_2_current_A',
                 'Motor_3_current_A', 'Current_X_A', 'Current_Y_A', 'Current_Z_A', 'Sum_motor_current_A',
                 'Motor_1_velocity_rpm', 'Motor_2_velocity_rpm', 'Motor_3_velocity_rpm']
    new_names = ['xsetspeed', 'ysetspeed', 'm1cur', 'm2cur',
                 'm3cur', 'xcur', 'ycur', 'rotcur', 'sum_motor_current',
                 'm1vel', 'm2vel', 'm3vel']

    name_dict = dict(zip(old_names, new_names))
    df = df.rename(columns=name_dict)

    df.to_csv(output_dir / "merged.csv", index=False)

    types = ['motor-axis-currents', 'motorCurrent-motorVelocities', 'only-motor-currents']

    for type in types:
        prepDF = df.copy()
        if type == 'motor-axis-currents':
            prepDF = prepDF[['xsetspeed', 'ysetspeed', 'm1cur', 'm2cur', 'm3cur', 'sum_motor_current', 'xcur', 'ycur',
                             'rotcur', 'sum_axis_current', 'rotational', 'surface_type']]
        elif type == 'motorCurrent-motorVelocities':
            prepDF = prepDF[['surface_type', 'rotational', 'xsetspeed', 'ysetspeed',
                             'm1cur', 'm2cur', 'm3cur', 'm1vel', 'm2vel', 'm3vel']]
        elif type == 'only-motor-currents':
            prepDF = prepDF[['surface_type', 'rotational', 'xsetspeed', 'ysetspeed', 'm1cur', 'm2cur',
                             'm3cur']]
        exportDF(prepDF, type)



