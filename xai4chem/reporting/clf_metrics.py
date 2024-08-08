import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

def classification_metrics(smiles_valid, y_valid, y_proba, output_folder):
    # Calculate ROC curve and optimal thresholds
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_proba)
    roc_auc = metrics.auc(fpr, tpr)

    optimal_threshold = thresholds[np.argmax(tpr - fpr)]  # Youden's J statistic
    optimal_idx = np.argmax(tpr - fpr)
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    optimal_threshold_fpr_5 = thresholds[fpr <= 0.05][-1]  # 0.05 is max_fpr 

    # Predictions using default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)

    # Predictions using optimal threshold (Youden's J)
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

    # Predictions using optimal threshold (Max FPR 0.05)
    y_pred_fpr_5 = (y_proba >= optimal_threshold_fpr_5).astype(int)

    # Save results for all thresholds
    evaluation_data = pd.DataFrame({
        'SMILES': smiles_valid,
        'Actual Value': y_valid,
        'Prob_Pred': y_proba,
        'Predicted Value (Default Threshold)': y_pred_default,
        'Predicted Value (Optimal Threshold - Youden\'s J)': y_pred_optimal,
        'Predicted Value (Optimal Threshold - Max FPR 0.05)': y_pred_fpr_5
    })

    evaluation_data.to_csv(os.path.join(output_folder, "evaluation_data.csv"), index=False)

    # Confusion Matrix for all thresholds
    cm_display_default = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_default))
    cm_display_default.plot()
    cm_display_default.ax_.set_title("Confusion Matrix - Default Threshold")
    cm_display_default.figure_.savefig(os.path.join(output_folder, 'confusion_matrix_default.png'))
    plt.close(cm_display_default.figure_)

    cm_display_optimal = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_optimal))
    cm_display_optimal.plot()
    cm_display_optimal.ax_.set_title(
        f"Confusion Matrix - Optimal Threshold (Youden's J({round(optimal_threshold, 4)}))")
    cm_display_optimal.figure_.savefig(os.path.join(output_folder, 'confusion_matrix_optimal.png'))
    plt.close(cm_display_optimal.figure_)

    cm_display_fpr_5 = ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_valid, y_pred_fpr_5))
    cm_display_fpr_5.plot()
    cm_display_fpr_5.ax_.set_title(
        f"Confusion Matrix - Optimal Threshold (Max FPR_0.05({round(optimal_threshold_fpr_5, 4)}))")
    cm_display_fpr_5.figure_.savefig(os.path.join(output_folder, 'confusion_matrix_fpr_5%.png'))
    plt.close(cm_display_fpr_5.figure_)

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter([optimal_fpr], [optimal_tpr], color='red',
                label=f'Youden Index({round(optimal_fpr, 4)}, {round(optimal_tpr, 4)})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
    plt.close()

    # Evaluation metrics for all thresholds
    metrics_data = {
        "Threshold Type": ["Default", "Optimal (Youden's J)", "Optimal (Max FPR 0.05)"],
        "Accuracy": [
            round(metrics.accuracy_score(y_valid, y_pred_default), 4),
            round(metrics.accuracy_score(y_valid, y_pred_optimal), 4),
            round(metrics.accuracy_score(y_valid, y_pred_fpr_5), 4)
        ],
        "Precision": [
            round(metrics.precision_score(y_valid, y_pred_default, average='macro'), 4),
            round(metrics.precision_score(y_valid, y_pred_optimal, average='macro'), 4),
            round(metrics.precision_score(y_valid, y_pred_fpr_5, average='macro'), 4)
        ],
        "Recall": [
            round(metrics.recall_score(y_valid, y_pred_default, average='macro'), 4),
            round(metrics.recall_score(y_valid, y_pred_optimal, average='macro'), 4),
            round(metrics.recall_score(y_valid, y_pred_fpr_5, average='macro'), 4)
        ],
        "F1 Score": [
            round(metrics.f1_score(y_valid, y_pred_default, average='macro'), 4),
            round(metrics.f1_score(y_valid, y_pred_optimal, average='macro'), 4),
            round(metrics.f1_score(y_valid, y_pred_fpr_5, average='macro'), 4)
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_folder, 'performance_metrics.csv'), index=False)

    return metrics_df
