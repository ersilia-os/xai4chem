# train.py
import os
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from xai4chem.representations import DatamolDescriptor, RDKitDescriptor, MordredDescriptor, MorganFingerprint, RDKitFingerprint
from xai4chem.supervised import Regressor, Classifier

def _get_descriptor(representation):
    '''
    Returns the descriptor/fingerprint class, 
    the fingerprints used(None for descriptor features), 
    and maximum features to be selected
    '''
    if representation == 'datamol_descriptor':
        return DatamolDescriptor(), None, None
    elif representation == 'rdkit_descriptor':
        return RDKitDescriptor(), None, 64
    elif representation == 'mordred_descriptor':
        return MordredDescriptor(), None, 100
    elif representation == 'morgan_fingerprint':
        return MorganFingerprint(), 'morgan', 100
    elif representation == 'rdkit_fingerprint':
        return RDKitFingerprint(), 'rdkit', 100
    else:
        raise ValueError("Invalid representation type")

def train(args):
    # Load data
    data = pd.read_csv(args.input_file)
    smiles = data["smiles"]
    target = data["activity"]
    
    # Check if the problem is binary classification
    is_binary_classification = target.nunique() == 2 and set(target.unique()) <= {0, 1}
    
    # Split data
    smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)
    
    # Reset indices
    smiles_train.reset_index(drop=True, inplace=True)
    smiles_valid.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)

    # Choose feature representation
    descriptor, fingerprints, max_features = _get_descriptor(args.representation)
    
    # Fit and transform
    descriptor.fit(smiles_train)
    train_features = descriptor.transform(smiles_train)
    valid_features = descriptor.transform(smiles_valid)
    
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Choose appropriate model
    if is_binary_classification: 
        print('...Classification.....\n', target.value_counts())
        model = Classifier(reports_dir, fingerprints=fingerprints, algorithm='catboost', k=max_features)
    else: 
        print('...Regression.....')
        model = Regressor(reports_dir, fingerprints=fingerprints, algorithm='catboost', k=max_features)
        
    # Train model
    model.fit(train_features, y_train)
    
    # Generate reports
    model.evaluate(valid_features, smiles_valid, y_valid)
    model.explain(train_features, smiles_list=smiles_train)
    
    # Retrain final model on all data
    print('.........Training Final Model.................')
    descriptor.fit(smiles)
    all_features = descriptor.transform(smiles)
    model.fit(all_features, target)

    # Save final model 
    model_filename = os.path.join(args.output_dir, "model.pkl")
    model.save_model(model_filename)
    
    #Save the descriptor used
    descriptor.save(os.path.join(args.output_dir, "descriptor.pkl"))    
