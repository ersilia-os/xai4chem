import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import shap
import optuna
import xgboost
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import sys

sys.path.append('.')
from xai4chem import MorganFingerprint, DatamolDescriptor, AccFgFingerprint
from xai4chem.reporting import Explainer
from xai4chem.reporting import regression_metrics, shapley_raw_total_per_atom, highlight_and_draw_molecule, plot_waterfall, draw_top_features


class Regressor:
    def __init__(self, output_folder, fingerprints="morgan", n_trials=300, k=None):
        self.n_trials = n_trials
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.model = None
        self.explainer = None
        self.max_features = k
        self.selected_features = None
        self.fingerprints = fingerprints

    def _select_features(self, X_train, y_train):
        if self.max_features is None:
            print("No maximum feature limit specified. Using all features.")
            self.selected_features = list(X_train.columns)
        elif X_train.shape[1] <= self.max_features:
            print(f"Number of input features is less than or equal to {self.max_features}. Using all features.")
            self.selected_features = list(X_train.columns)
        else:
            print(f"Features in the dataset are more than {self.max_features}. Using KBest selection.")
            selector = SelectKBest(score_func=mutual_info_regression, k=self.max_features)
            selector.fit(X_train, y_train)
            # Get the indices of the selected features
            selected_indices = np.argsort(selector.scores_)[::-1][:self.max_features]
            self.selected_features = X_train.columns[selected_indices]

        return self.selected_features

    def _optimize_xgboost(self, trial, X, y):
        params = {
            'lambda': trial.suggest_int('lambda', 0, 1.0),
            'alpha': trial.suggest_int('alpha', 0, 1.0),
            'gamma': trial.suggest_int('gamma', 0, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
            'random_state': trial.suggest_categorical('random_state', [0, 42]),
        }

        model = xgboost.XGBRegressor(**params)
        return self._train_and_evaluate_optuna_model(model, X, y)

    def _train_and_evaluate_optuna_model(self, model, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

        preds = model.predict(X_valid)
        mae = metrics.mean_absolute_error(y_valid, preds)
        return mae
        
    def _featurize_smiles(self, smiles_list):
        X = self.descriptor.transform(smiles_list)
        X = X[self.selected_features]
        return X

    def fit(self, X_train, y_train):
        self._select_features(X_train, y_train)
        X_train = X_train[self.selected_features]
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._optimize_xgboost(trial, X_train.values, y_train), n_trials=self.n_trials, timeout=1200)
        best_params = study.best_params
        print('Best parameters for XGBoost:', best_params)
        self.model = xgboost.XGBRegressor(**best_params)
        self.model.fit(X_train.values, y_train)

        y_pred = self.model.predict(X_train.values)
        assert max(y_pred) - min(y_pred) > 0.05, "Insufficicient variation between highest and lowest prediction scores."

    def evaluate(self, X_valid_features, smiles_valid, y_valid):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        y_pred = self.model_predict(X_valid_features)
        evaluation_metrics = regression_metrics(smiles_valid, y_valid, y_pred, self.output_folder)

        return evaluation_metrics

    def model_predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        X = X[list(self.selected_features)]
        return self.model.predict(X)

    def explain(self, smiles_list):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        if self.fingerprints == "morgan":
            self.descriptor = MorganFingerprint()
        elif self.fingerprints == "datamol":
            self.descriptor = DatamolDescriptor()
        elif self.fingerprints == "accfg":
            self.descriptor = AccFgFingerprint()
        self.descriptor.fit(smiles_list)
        self.explainer = Explainer(self, smiles_list, self.output_folder, self.fingerprints)
        self.explainer.explain_model()

    def explain_preds(self, smiles_list, output_folder):
        self.explainer.explain_predictions(smiles_list, output_folder)

    def explain_mol_atoms(self, smiles, atomInfo=False, file_prefix=None):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        if self.explainer is None:
            raise ValueError("The model has not yet been explained.")
        X = self._featurize_smiles([smiles])
        X_cols = X.columns

        shap_values = self.explainer.explainer(X)
        if file_prefix:
            plot_waterfall(shap_values, 0, smiles, self.output_folder, file_prefix + f"_waterfall_{self.fingerprints}.png", self.fingerprints)
        else:
            plot_waterfall(shap_values, 0, smiles, self.output_folder, smiles + f"_waterfall_{self.fingerprints}.png", self.fingerprints)
        
        if self.fingerprints == "morgan" or self.fingerprints == "accfg":
            bit_info, valid_top_bits, bit_shap_values = self.explainer.explain_mol_features(smiles, X_cols, fingerprints=self.fingerprints)
            raw_atom_values = shapley_raw_total_per_atom(bit_info, bit_shap_values, smiles, fingerprints=self.fingerprints)
            scaled_shapley_values = self.explainer.scaler.transform(np.array(list(raw_atom_values.values())).reshape(-1, 1)).flatten()
            atom_shapley_values = {k: scaled_shapley_values[i] for i, k in enumerate(raw_atom_values)}
        
            if file_prefix:
                highlight_and_draw_molecule(atom_shapley_values, smiles, os.path.join(self.output_folder, file_prefix + f"_highlights_{self.fingerprints}.png"))
                draw_top_features(bit_info, valid_top_bits, smiles, os.path.join(self.output_folder, file_prefix + f"_top_features_{self.fingerprints}.png"), self.fingerprints) 
            else:
                highlight_and_draw_molecule(atom_shapley_values, smiles, os.path.join(self.output_folder, smiles + f"_highlights_{self.fingerprints}.png"))
                draw_top_features(bit_info, valid_top_bits, smiles, os.path.join(self.output_folder, smiles + f"_top_features_{self.fingerprints}.png"), self.fingerprints)
        
        if atomInfo:
            return atom_shapley_values
        else:
            return None

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("The model has not been trained.")
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        model_data = {
            'model': self.model,
            'selected_features': self.selected_features,
            'fingerprints': self.fingerprints, 
            'describer': self.descriptor, 
        }
        if self.explainer is not None:
            model_data['shapley_explainer'] = self.explainer
        joblib.dump(model_data, filename)

    def load_model(self, filename):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.selected_features = model_data['selected_features']
        self.fingerprints = model_data["fingerprints"]
        self.descriptor = model_data["describer"]
        if 'shapley_explainer' in model_data:
            self.explainer = model_data['shapley_explainer']
