# XAI4Chem

Explainable AI for Chemistry

## Installation
For the environment,
```bash
conda create -n xai4chem python=3.10 -y
conda activate xai4chem
```
Then install from GitHub:
```bash
python -m pip install git+https://github.com/ersilia-os/xai4chem.git 
```
 
## Usage
### Data
Read data file and split:
```python
import pandas as pd  
from sklearn.model_selection import train_test_split

data = pd.read_csv("plasmodium_falciparum_3d7_ic50.csv") #data path

# Extract SMILES and target values
smiles = data["smiles"]
target = data["pchembl_value"] #target value's column_name

# Split data into training and test sets
smiles_train, smiles_valid, y_train, y_valid = train_test_split(smiles, target, test_size=0.2, random_state=42)
```
Calculate and transform descriptors:
```python
from xai4chem import DatamolDescriptor

descriptor = DatamolDescriptor(discretize=False)

# Fit the descriptor to training data
descriptor.fit(smiles_train)

# Transform the data
smiles_train_transformed = descriptor.transform(smiles_train)
smiles_valid_transformed = descriptor.transform(smiles_valid)
```

### Model Training and Evaluation
The tool provides a `Regressor` class for training and evaluating regression models. It supports both XGBoost and CatBoost algorithms. You can train the model with default parameters or perform hyperparameter optimization using Optuna.
```python
from xai4chem import Regressor

# use xgboost or catboost
regressor = Regressor(algorithm='xgboost', n_trials=200)

# Train the model
regressor.train(smiles_train_transformed, y_train, default_params=False)

#you can save the trained model
#regressor.save('model.joblib') #pass the filename

# Evaluate the model
regressor.evaluate(smiles_valid_transformed, y_valid, output_folder) #the output folder is where evaluation metrics will be saved
```
### Model Interpretation
The `Regressor` class also provides functionality for interpreting model predictions. You can generate plots by;
```python
regressor.explain(smiles_train_transformed, descriptor.feature_names, output_folder) #the output folder is where interpretability plots will be saved.
```