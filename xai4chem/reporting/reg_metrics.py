import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def regression_metrics(smiles_valid, y_valid, y_pred, output_folder):
    # A dataFrame with the predictions
    evaluation_data = pd.DataFrame({
        'SMILES': smiles_valid,
        'Actual Value': y_valid,
        'Predicted Value': y_pred
    })

    # Save as a CSV file
    evaluation_data.to_csv(os.path.join(output_folder, "evaluation_data.csv"), index=False)

    plt.scatter(y_valid, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs. Predicted Values')
    plt.savefig(os.path.join(output_folder, 'evaluation_scatter_plot.png'))
    plt.close()

    # Metrics
    evaluation_metrics = {
        "Mean Squared Error": round(metrics.mean_squared_error(y_valid, y_pred), 4),
        "Root Mean Squared Error": round(np.sqrt(metrics.mean_squared_error(y_valid, y_pred)), 4),
        "Mean Absolute Error": round(metrics.mean_absolute_error(y_valid, y_pred), 4),
        "R-squared Score": round(metrics.r2_score(y_valid, y_pred), 4),
        "Explained Variance Score": round(metrics.explained_variance_score(y_valid, y_pred), 4)
    }
    with open(os.path.join(output_folder, 'evaluation_metrics.json'), 'w') as f:
        json.dump(evaluation_metrics, f, indent=4)

    return evaluation_metrics
