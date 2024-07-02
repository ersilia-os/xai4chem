import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "xai4chem"))

from datamol_desc import DatamolDescriptor
from rdkitclassical_desc import RDkitClassicalDescriptor
from mordred_desc import MordredDescriptor
from regressor import Regressor
from morgan_desc import MorganFingerprint

output_folder = os.path.join(root, "..", "results")

if __name__ == "__main__":
    # Read data from CSV file into a DataFrame
    data = pd.read_csv(os.path.join(root, "..", "data", "plasmodium_falciparum_3d7_ic50.csv"))

    # Extract SMILES and target values
    smiles = data["smiles"]
    target = data["pchembl_value"]

    # Split data into training and test sets
    smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)

    # Reset indices
    smiles_train.reset_index(drop=True, inplace=True)
    smiles_valid.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)

    # Instantiate the descriptor class
    descriptor = MorganFingerprint()

    descriptor.fit(smiles_train)

    # Transform the data 
    train_features = descriptor.transform(smiles_train)
    valid_features= descriptor.transform(smiles_valid)

    # Instantiate the regressor
    regressor = Regressor(output_folder, algorithm='xgboost')
    
    # Train the model 
    regressor.fit(train_features, y_train) 

    # Evaluate model
    regressor.evaluate(valid_features, smiles_valid, y_valid)

    # Explain the model     
    regressor.explain(train_features, smiles_list=smiles_train, use_fingerprints=True)
    
    