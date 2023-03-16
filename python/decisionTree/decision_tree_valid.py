import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
import seaborn as sns
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix

def metricsAndPlots(model_name, y_val, y_pred, metrics_dir, cm_plot_dir):

    with open(metrics_dir, 'w') as fd:
        json.dump(
            {
                'accuracy': accuracy_score(y_val, y_pred),
                'balanced accuracy': balanced_accuracy_score(y_val, y_pred),
                'precision_macro': precision_score(y_val, y_pred, average='macro'),
                'recall_macro': recall_score(y_val, y_pred, average='macro')

            },
            fd,
            indent=4
        )

    sns.set_theme(context='paper', style='white', font='Serif', font_scale=2,
                  rc={'savefig.dpi': 800, 'axes.titlepad': 33,
                      'figure.figsize': (6, 4.5), 'axes.titlesize': 25,
                      'axes.labelpad': 25, 'xtick.labeltop': True,
                      'xtick.major.pad': 10, 'ytick.major.pad': 15,
                      'xtick.labelbottom': False, 'font.weight': 'bold'})

    categories = [x.upper() for x in ['grey', 'green', 'table']]
    ConfMatrix = confusion_matrix(y_val, y_pred)

    lineWidth = 1.0

    heatmap = sns.heatmap(ConfMatrix, annot=True, cbar=False, cmap='Blues',
                          linewidths=lineWidth, linecolor='#000', fmt="d",
                          xticklabels=categories, yticklabels=categories)

    heatmap.set_title('Confusion matrix for {}'.format(model_name))
    heatmap.set_xlabel("Predicted labels")
    # heatmap.set_ylabel("True labels")
    plt.yticks(rotation=0)


    for _, spine in heatmap.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(lineWidth)

    fig = heatmap.get_figure()

    fig.savefig(cm_plot_dir, bbox_inches='tight', pad_inches=0.5)


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=True)
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=True)
    parser.add_argument('--model_name', '-mn', type=str, default='data/models/',
                        required=True)
    parser.add_argument('--argument_pool', '-ap', type=str, default='motor-axis-currents',
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    argument_pool = args.argument_pool
    model_name = args.model_name

    metrics_dir = 'metrics/{}/{}_metrics.json'.format(
        argument_pool, model_name
    )

    cm_plot_dir = 'plots/{}/{}_CM.png'.format(
        argument_pool, model_name
    )

    Path('metrics/'+argument_pool).mkdir(exist_ok=True, parents=True)
    Path('plots/'+argument_pool).mkdir(exist_ok=True, parents=True)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    reg = load(input_model)

    y_pred = np.squeeze(reg.predict(X_val))

    print('Score: ', reg.score(X_val, y_val))

    metricsAndPlots(model_name, y_val, y_pred, metrics_dir, cm_plot_dir)