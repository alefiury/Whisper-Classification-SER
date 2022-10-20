from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_conf_matrix(
    targets: List[int],
    preds: List[int],
    classes: List[str],
    output_path: str
) -> None:
    """
    Saves a confusion matrix given the true labels and the predicted outputs.
    """
    cm = confusion_matrix(
        y_true=targets,
        y_pred=preds
    )

    df_cm = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    plt.figure(figsize=(24,12))
    plot = sns.heatmap(df_cm, annot=True,  fmt='g')
    figure1 = plot.get_figure()
    plt.tight_layout()
    figure1.savefig(output_path, format='png')