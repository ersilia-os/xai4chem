import os
import joblib
import pandas as pd
from xai4chem.supervised import Regressor, Classifier
import matplotlib.pyplot as plt
import seaborn as sns

def infer(args):
    # Load model and descriptor
    model_path = os.path.join(args.model_dir, "model.pkl")
    descriptor_path = os.path.join(args.model_dir, "descriptor.pkl")
    descriptor = joblib.load(descriptor_path)
    
    # Determine model type
    temp_model = Regressor(args.output_dir)
    temp_model.load_model(model_path)
    if hasattr(temp_model.model, 'predict_proba'):
        model = Classifier(args.output_dir)
        model_type = 'clf'
    else:
        model = Regressor(args.output_dir)
        model_type = 'reg'
    
    model.load_model(model_path)
    
    # Load and transform data
    data = pd.read_csv(args.input_file)
    smiles = data["smiles"]
    features = descriptor.transform(smiles)
    
    # Make predictions and save results
    if model_type == 'reg':
        predictions = model.model_predict(features)
        pd.DataFrame({"smiles": smiles, "pred": predictions}).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
        
        # Plot score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions, kde=True)
        plt.title('Score Distribution Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(args.output_dir, 'score_distribution_plot.png'))
        plt.close()
    else:
        proba, pred = model.model_predict(features)
        pd.DataFrame({"smiles": smiles, "proba": proba, "pred": pred}).to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)
        
        # Plot score violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=pred, y=proba)
        plt.title('Score Violin Plot')
        plt.xlabel('Predicted Class')
        plt.ylabel('Predicted Probability')
        plt.savefig(os.path.join(args.output_dir, 'score_violin_plot.png'))
        plt.close()
        
        # Plot score strip plot
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=pred, y=proba, hue=pred)
        plt.title('Score Strip Plot')
        plt.xlabel('Predicted Class')
        plt.ylabel('Predicted Probability')
        plt.legend(title='Predicted Class', bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(args.output_dir, 'score_strip_plot.png'))
        plt.close()
    
    model.explain(features, smiles_list=smiles)
