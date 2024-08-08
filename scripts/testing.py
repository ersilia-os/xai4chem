import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "xai4chem"))

from representations import DatamolDescriptor, RDKitDescriptor, MordredDescriptor, MorganFingerprint, RDKitFingerprint
from supervised import Regressor, Classifier

output_folder = os.path.join(root, "..", "results")

if __name__ == "__main__":
    # Read data from CSV file into a DataFrame
    data = pd.read_csv(os.path.join(root, "..", "data", "plasmodium_falciparum_3d7_ic50.csv"))

    # Extract SMILES and target values
    smiles = data["smiles"]
    target = data["pchembl_value"]#regression #(data["uM_value"] <= 2).astype(int)#classification
    # print(target.value_counts())#classification
    
    # Split data into training and test sets
    smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)

    # Reset indices
    smiles_train.reset_index(drop=True, inplace=True)
    smiles_valid.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)

    # Instantiate the descriptor class
    descriptor = RDKitFingerprint()

    descriptor.fit(smiles_train)

    # Transform the data 
    train_features = descriptor.transform(smiles_train)
    valid_features= descriptor.transform(smiles_valid)

    # Instantiate the Regressor/Classifier
    trainer = Regressor(output_folder, fingerprint='rdkit', k=100)#fingerprints='morgan' if MorganFingerprint
    
    # Train the model 
    trainer.fit(train_features, y_train) 

    # Evaluate model
    trainer.evaluate(valid_features, smiles_valid, y_valid)

    # Explain the model     
    trainer.explain(train_features, smiles_list=smiles_train)    
    